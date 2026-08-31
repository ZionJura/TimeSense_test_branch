"""Run a TimeSense training stage with the released ChatTS trainer.

The source checkpoint is never modified. This command copies it to ``tmp/``,
replaces the TimeSense model append files in that copy, and gives ChatTS an
isolated dataset registry in the same temporary directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


APPEND_FILES = (
    "configuration_qwen2.py",
    "modeling_qwen2.py",
    "processing_qwen2_ts.py",
    "chat_template.jinja",
)
DATASET_NAME = "timesense_chrongen"


def load_stage(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("stage configuration requires PyYAML") from exc
    stage = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {"freeze_llm", "ts_loss_weight", "max_steps"}
    if set(stage) != required:
        raise ValueError(f"{path} must contain exactly: {', '.join(sorted(required))}")
    return stage


def prepare_model(source: Path, destination: Path, stage: dict, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"temporary model already exists: {destination}. "
                "Use --overwrite-tmp to rebuild it."
            )
        shutil.rmtree(destination)

    # A local model checkout can retain Git LFS metadata. It is not needed by
    # Transformers and can be much larger than the checkpoint itself.
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    append_dir = Path(__file__).resolve().parent
    for filename in APPEND_FILES:
        source_file = append_dir / filename
        if not source_file.is_file():
            raise FileNotFoundError(f"missing TimeSense append file: {source_file}")
        shutil.copy2(source_file, destination / filename)

    config_path = destination / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["freeze_model"] = stage["freeze_llm"]
    config["ts_loss_weight"] = stage["ts_loss_weight"]
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def prepare_dataset(data: Path, temp_root: Path) -> Path:
    dataset_dir = temp_root / "data"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    staged_data = dataset_dir / "chron_gen_train.jsonl"
    if staged_data.exists() or staged_data.is_symlink():
        staged_data.unlink()
    staged_data.symlink_to(data.resolve())
    registry = {
        DATASET_NAME: {
            "file_name": staged_data.name,
            "columns": {"prompt": "input", "response": "output", "timeseries": "timeseries"},
        }
    }
    (dataset_dir / "dataset_info.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return dataset_dir


def export_inference_checkpoint(source: Path, destination: Path) -> None:
    """Save a generation checkpoint without reconstruction-decoder parameters."""
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("inference export requires Transformers") from exc
    model = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True, torch_dtype="auto")
    if not hasattr(model, "ts_decoder"):
        raise RuntimeError(f"{source} is not a TimeSense checkpoint")
    model.ts_decoder = None
    model.config.use_ts_decoder = False
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    AutoProcessor.from_pretrained(source, trust_remote_code=True).save_pretrained(destination)


def build_command(args: argparse.Namespace, stage: dict, prepared_model: Path, dataset_dir: Path) -> list[str]:
    trainer = args.chatts_training / "src" / "train.py"
    command = [
        str(trainer),
        "--stage", "sft",
        "--model_name_or_path", str(prepared_model),
        "--dataset", DATASET_NAME,
        "--dataset_dir", str(dataset_dir),
        "--do_train",
        "--template", "chatts",
        "--finetuning_type", "full",
        "--output_dir", str(args.output),
        "--overwrite_output_dir",
        "--trust_remote_code", "True",
        "--report_to", "none",
        "--seed", str(args.seed),
        "--per_device_train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--timeseries_sft_lr", str(args.learning_rate),
        "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.02",
        "--logging_steps", "1",
        "--save_steps", str(args.save_steps),
        "--num_train_epochs", "0",
        "--max_steps", str(stage["max_steps"]),
        "--cutoff_len", str(args.cutoff_len),
        "--preprocessing_num_workers", str(args.preprocessing_workers),
        "--fp16",
        "--plot_loss",
        "--save_only_model",
        "--save_safetensors", "False",
    ]
    if args.launcher == "deepspeed":
        command = [
            "deepspeed", "--num_gpus", str(args.num_gpus), "--master_port", str(args.master_port),
            *command,
            "--deepspeed", str(args.chatts_training / "ds_config" / "ds_config_3.json"),
        ]
    else:
        command = [args.python, *command]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chatts-training", type=Path, default=Path("chatts_training"))
    parser.add_argument("--tmp-dir", type=Path, default=Path("tmp"))
    parser.add_argument("--overwrite-tmp", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--export-inference", action="store_true")
    parser.add_argument("--inference-output", type=Path)
    parser.add_argument("--launcher", choices=("deepspeed", "python"), default="deepspeed")
    parser.add_argument("--python", default="python")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--master-port", type=int, default=19901)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=600)
    parser.add_argument("--cutoff-len", type=int, default=12000)
    parser.add_argument("--preprocessing-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=66)
    args = parser.parse_args()

    args.config = args.config.resolve()
    args.model = args.model.resolve()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    args.chatts_training = args.chatts_training.resolve()
    args.tmp_dir = args.tmp_dir.resolve()
    stage = load_stage(args.config)
    for path, label in (
        (args.data, "training data"),
        (args.model, "model"),
        (args.chatts_training / "src" / "train.py", "ChatTS trainer"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    temp_root = args.tmp_dir / args.config.stem
    prepared_model = temp_root / "model"
    prepare_model(args.model, prepared_model, stage, args.overwrite_tmp)
    dataset_dir = prepare_dataset(args.data, temp_root)
    command = build_command(args, stage, prepared_model, dataset_dir)
    (temp_root / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    if args.prepare_only:
        print("Prepared temporary model and dataset registry:", temp_root)
        print("Run:", " ".join(command))
        return
    subprocess.run(command, check=True, cwd=args.chatts_training)
    if args.export_inference:
        inference_output = (args.inference_output or args.output.with_name(f"{args.output.name}_inference")).resolve()
        export_inference_checkpoint(args.output, inference_output)


if __name__ == "__main__":
    main()
