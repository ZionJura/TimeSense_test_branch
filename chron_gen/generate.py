"""Generate ChronGen training data or the complete EvalTS task suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm import LLM, parse_json

from .io import write_jsonl
from .tasks import EVALTS_TASKS, TASKS, generate_eval_sample, generate_sample


FORMATTER_PROMPT = """You format synthetic time-series examples for supervised fine-tuning.
Rewrite the input and answer in clear, natural English while preserving every fact, index, value, label, and time-series placeholder.
Do not solve the task again. Do not add analysis or unsupported details.
Return one JSON object with exactly two string fields: input and output."""


def training_record(sample: dict[str, Any], formatter: LLM | None = None) -> dict[str, Any]:
    input_text = sample["question"]
    output_text = json.dumps(sample["answer"], ensure_ascii=True, sort_keys=True)
    if formatter is not None:
        request = {
            "input": input_text,
            "output": output_text,
        }
        response = formatter.generate(
            [{"role": "system", "content": FORMATTER_PROMPT}, {"role": "user", "content": json.dumps(request)}],
            json_mode=True,
        )
        formatted = parse_json(response)
        if set(formatted) != {"input", "output"} or not all(isinstance(formatted[key], str) and formatted[key].strip() for key in formatted):
            raise ValueError("formatter must return non-empty input and output strings")
        if formatted["input"].count("<ts><ts/>") != input_text.count("<ts><ts/>"):
            raise ValueError("formatter changed the number of time-series placeholders")
        input_text, output_text = formatted["input"].strip(), formatted["output"].strip()
    return {"input": input_text, "output": output_text, "timeseries": sample["timeseries"], "type": sample["task"]}


def generate_train(args: argparse.Namespace) -> None:
    total = args.samples if args.full else 5
    output = args.output or Path("data/train/chron_gen_train.jsonl" if args.full else "data/train/demo.jsonl")
    formatter = None
    if args.format:
        formatter = LLM.from_config(
            args.llm_config,
            "training_formatter",
            base_url=args.api_base,
            model=args.api_model,
            api_key_env=args.api_key_env,
        )
    records = []
    for index in range(total):
        task = TASKS[index % len(TASKS)]
        sample = generate_sample(task, length=args.length, dimensions=args.dimensions, seed=args.seed + index)
        records.append(training_record(sample, formatter))
    write_jsonl(output, records)


def generate_eval(args: argparse.Namespace) -> None:
    tasks = EVALTS_TASKS if args.full else ("uni_trend", "segment", "root_cause_analysis")
    per_task = args.samples_per_task if args.full else 1
    output_dir = args.output or Path("data/eval/full" if args.full else "data/eval/demo")
    for task_index, task in enumerate(tasks):
        records = []
        for sample_index in range(per_task):
            sample = generate_eval_sample(task, seed=args.seed + task_index * 1000, index=sample_index)
            record = training_record(sample)
            record["id"] = f"{task}-{sample_index:05d}"
            records.append(record)
        write_jsonl(output_dir / f"{task}.jsonl", records)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--full", action="store_true", help="Generate the full release-scale split instead of the tracked demo.")
    parser.add_argument("--output", type=Path, help="Output file for training or output directory for EvalTS task files.")
    parser.add_argument("--seed", type=int, default=20260831)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="purpose", required=True)
    train = subparsers.add_parser("train", help="Generate ChronGen training records.")
    _common(train)
    train.add_argument("--samples", type=int, default=100000)
    train.add_argument("--length", type=int, default=128)
    train.add_argument("--dimensions", type=int, default=1)
    train.add_argument("--format", action="store_true", help="Use llm_config.yaml to format natural-language training pairs.")
    train.add_argument("--llm-config", type=Path, default=Path("llm_config.yaml"))
    train.add_argument("--api-base", help="Temporary OpenAI-compatible formatter endpoint override.")
    train.add_argument("--api-model", help="Temporary formatter model override.")
    train.add_argument("--api-key-env", help="Environment variable holding the temporary formatter API key.")
    evaluation = subparsers.add_parser("eval", help="Generate every EvalTS task.")
    _common(evaluation)
    evaluation.add_argument("--samples-per-task", type=int, default=300)
    args = parser.parse_args()
    if args.purpose == "train":
        generate_train(args)
    else:
        generate_eval(args)


if __name__ == "__main__":
    main()
