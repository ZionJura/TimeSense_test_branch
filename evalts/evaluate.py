"""Run a local checkpoint or an OpenAI-compatible API model on EvalTS."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from llm import load_config

from .backends import ModelSpec, build_backend
from .dataset import expected_answer as _expected
from .dataset import load_records as _load
from .dataset import model_samples
from .dataset import question_text as _question
from .extract import extract_answer
from .protocol import validate_generations
from .task_specs import CHANGE_POINT_TOLERANCE, PERIOD_TOLERANCE, POSITION_TOLERANCE, SEGMENT_TOLERANCE, TASK_SCORING
from .task_specs import base_task as _base_task

def _model_spec(args: argparse.Namespace) -> ModelSpec:
    """Build a unified backend specification, preserving legacy CLI flags."""
    if args.model_config:
        spec = ModelSpec.from_file(args.model_config)
    elif args.api:
        values = load_config(args.llm_config, "evaluation_api")
        spec = ModelSpec(
            backend="openai_compatible",
            model=args.api_model or values["model"],
            base_url=args.api_base or values["base_url"],
            api_key_env=args.api_key_env or values["api_key_env"],
            max_tokens=args.api_max_tokens or values.get("max_tokens", 8192),
        )
    else:
        spec = ModelSpec(
            backend="huggingface",
            model=str(args.model or Path("results/stage2_inference")),
            input_mode="serialized" if args.text_only else "auto",
            max_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            device=args.device,
        )
    if args.model:
        spec = replace(spec, model=str(args.model))
    if args.text_only:
        spec = replace(spec, input_mode="serialized")
    if args.device:
        spec = replace(spec, device=args.device)
    if args.batch_size != 1:
        spec = replace(spec, batch_size=args.batch_size)
    if args.max_new_tokens != 512:
        spec = replace(spec, max_tokens=args.max_new_tokens)
    return spec


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.lower().replace("_", " ").split())
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in sorted(value.items())}
    return value


def _atoms(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        return [atom for key, item in sorted(value.items()) for atom in _atoms(item, f"{prefix}/{key}")]
    if isinstance(value, list):
        return [atom for index, item in enumerate(value) for atom in _atoms(item, f"{prefix}/{index}")]
    return [f"{prefix}={_normalize(value)}"]


def _f1(matches: int, predicted_count: int, expected_count: int) -> float:
    if not predicted_count and not expected_count:
        return 1.0
    if not predicted_count or not expected_count:
        return 0.0
    precision, recall = matches / predicted_count, matches / expected_count
    return 2 * precision * recall / (precision + recall) if matches else 0.0


def _maximum_matches(predicted: list[Any], expected: list[Any], compatible) -> int:
    """Maximum one-to-one bipartite matching; a prediction cannot match two labels."""
    assigned: dict[int, int] = {}

    def visit(predicted_index: int, seen: set[int]) -> bool:
        for expected_index, target in enumerate(expected):
            if expected_index in seen or not compatible(predicted[predicted_index], target):
                continue
            seen.add(expected_index)
            if expected_index not in assigned or visit(assigned[expected_index], seen):
                assigned[expected_index] = predicted_index
                return True
        return False

    return sum(visit(index, set()) for index in range(len(predicted)))


def _point_entities(value: Any, path: tuple[int, ...] = ()) -> list[tuple[tuple[int, ...], int]]:
    if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
        return [(path, int(item)) for item in value]
    if isinstance(value, list):
        return [entity for index, item in enumerate(value) for entity in _point_entities(item, path + (index,))]
    return []


def _point_f1(predicted: Any, expected: Any, tolerance: int) -> float:
    predicted_entities, expected_entities = _point_entities(predicted), _point_entities(expected)
    matches = _maximum_matches(predicted_entities, expected_entities, lambda left, right: left[0] == right[0] and abs(left[1] - right[1]) <= tolerance)
    return _f1(matches, len(predicted_entities), len(expected_entities))


def _event_f1(predicted: list[tuple[tuple[Any, ...], tuple[int, ...]]], expected: list[tuple[tuple[Any, ...], tuple[int, ...]]]) -> float:
    matches = _maximum_matches(predicted, expected, lambda left, right: left[0] == right[0] and len(left[1]) == len(right[1]) and all(abs(a - b) <= POSITION_TOLERANCE for a, b in zip(left[1], right[1])))
    return _f1(matches, len(predicted), len(expected))


def _spike_events(answer: dict[str, Any]) -> list[tuple[tuple[Any, ...], tuple[int, ...]]]:
    return [((series + 1,), (int(item["point"]),)) for series, values in enumerate(answer.get("spikes", [])) for item in values if isinstance(item, dict) and "point" in item]


def _interval_events(answer: dict[str, Any]) -> list[tuple[tuple[Any, ...], tuple[int, ...]]]:
    return [((), tuple(int(point) for point in interval)) for interval in answer.get("intervals", []) if isinstance(interval, list) and len(interval) == 2]


def _root_cause_f1(predicted: dict[str, Any], expected: dict[str, Any]) -> float:
    root_score = 0.5 if predicted.get("root_cause") == expected.get("root_cause") else 0.0
    predicted_order = predicted.get("propagation_order", [])
    expected_order = expected.get("propagation_order", [])
    if not isinstance(predicted_order, list) or not isinstance(expected_order, list):
        return root_score
    set_f1 = _f1(len(set(predicted_order) & set(expected_order)), len(set(predicted_order)), len(set(expected_order)))
    ordered = (
        sum(left == right for left, right in zip(predicted_order, expected_order)) / len(expected_order)
        if len(predicted_order) == len(expected_order) and expected_order else 0.0
    )
    return root_score + 0.5 * max(set_f1, ordered)


def _anomaly_f1(predicted: dict[str, Any], expected: dict[str, Any]) -> float:
    def events(answer: dict[str, Any]) -> set[tuple[int, str]]:
        return {
            (int(item["series"]), _normalize(item["direction"]))
            for item in answer.get("series_status", [])
            if isinstance(item, dict)
            and item.get("is_anomalous")
            and isinstance(item.get("series"), (int, float))
            and isinstance(item.get("direction"), str)
        }
    predicted_events, expected_events = events(predicted), events(expected)
    return _f1(len(predicted_events & expected_events), len(predicted_events), len(expected_events))


def _extreme_score(predicted: dict[str, Any], expected: dict[str, Any]) -> float:
    predicted_values, expected_values = predicted.get("extrema", []), expected.get("extrema", [])
    if not isinstance(predicted_values, list) or not isinstance(expected_values, list) or len(predicted_values) != len(expected_values):
        return 0.0
    scores = []
    for predicted_item, expected_item in zip(predicted_values, expected_values):
        if not isinstance(predicted_item, dict) or not isinstance(expected_item, dict):
            scores.append(0.0)
            continue
        score = 0.0
        for key in ("min", "max"):
            try:
                score += 0.5 * (round(float(predicted_item[key]), 2) == round(float(expected_item[key]), 2))
            except (KeyError, TypeError, ValueError):
                pass
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def _period_score(predicted: dict[str, Any], expected: dict[str, Any]) -> float:
    if not isinstance(predicted, dict) or not isinstance(expected, dict):
        return 0.0
    predicted_periods, expected_periods = predicted.get("periods", []), expected.get("periods", [])
    if not isinstance(predicted_periods, list) or not isinstance(expected_periods, list):
        return 0.0
    if len(predicted_periods) != len(expected_periods):
        return 0.0
    for predicted_period, expected_period in zip(predicted_periods, expected_periods):
        if expected_period is None:
            if predicted_period is not None:
                return 0.0
            continue
        try:
            if abs(int(predicted_period) - int(expected_period)) > PERIOD_TOLERANCE:
                return 0.0
        except (TypeError, ValueError):
            return 0.0
    return 1.0


def _score(task: str, expected: Any, predicted: Any) -> float:
    base_task = _base_task(task)
    mode = TASK_SCORING[base_task]
    if mode == "accuracy":
        if base_task == "extreme":
            return _extreme_score(predicted, expected)
        if base_task == "period":
            return _period_score(predicted, expected)
        return float(_normalize(expected) == _normalize(predicted))
    if mode == "legacy_relative":
        first = 0.3 * (_normalize(expected.get("first_trend")) == _normalize(predicted.get("first_trend")))
        relative = 0.7 * (_normalize(expected.get("relative_change")) == _normalize(predicted.get("relative_change")))
        return float(first + relative)
    if base_task == "change_point":
        return _point_f1(predicted.get("change_points", []), expected.get("change_points", []), CHANGE_POINT_TOLERANCE)
    if base_task == "segment":
        return _point_f1(predicted.get("change_points", []), expected.get("change_points", []), SEGMENT_TOLERANCE)
    if base_task == "spike":
        return _event_f1(_spike_events(predicted), _spike_events(expected))
    if base_task == "comparison":
        return _event_f1(_interval_events(predicted), _interval_events(expected))
    if base_task == "root_cause_analysis":
        return _root_cause_f1(predicted, expected)
    if base_task == "anomaly_detection":
        return _anomaly_f1(predicted, expected)
    predicted_atoms, expected_atoms = set(_atoms(predicted)), set(_atoms(expected))
    return _f1(len(predicted_atoms & expected_atoms), len(predicted_atoms), len(expected_atoms))


def _write_results(
    result_path: Path,
    predictions: list[dict[str, Any]],
    details: list[dict[str, Any]],
    shard_index: int,
) -> None:
    """Write one task file per shard beneath a single shard directory."""
    shard_path = result_path / "shard"
    shard_path.mkdir(parents=True, exist_ok=True)

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction, detail in zip(predictions, details):
        by_task[detail["type"]].append(prediction)
    for task, rows in by_task.items():
        path = shard_path / f"{task}-{shard_index:05d}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    grouped: dict[str, list[float]] = defaultdict(list)
    for item in details:
        grouped[item["type"]].append(item["score"])
    overall = sum(item["score"] for item in details) / len(details) if details else 0.0
    per_task = {task: sum(values) / len(values) for task, values in sorted(grouped.items())}
    metrics = {
        "overall_score": overall,
        "per_task_score": per_task,
        "overall_accuracy": overall,
        "per_task_accuracy": per_task,
        "num_samples": len(details),
        "shard_index": shard_index,
    }
    (shard_path / f"metrics-{shard_index:05d}.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--api", action="store_true", help="Evaluate an OpenAI-compatible API model instead of a local checkpoint.")
    parser.add_argument("--data", type=Path, help="An EvalTS task directory or a legacy JSONL file.")
    parser.add_argument("--results", type=Path, help="Experiment result directory; task records are written beneath results/shard/.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split the input deterministically across this many workers.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index to evaluate.")
    parser.add_argument("--model", type=Path, help="Local model override for a Hugging Face backend.")
    parser.add_argument("--model-config", type=Path, help="Unified model backend configuration.")
    parser.add_argument("--text-only", action="store_true", help="Legacy alias for --input-mode serialized in a model configuration.")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=1, help="Local text-only generation batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--llm-config", type=Path, default=Path("llm_config.yaml"))
    parser.add_argument("--api-base")
    parser.add_argument("--api-model")
    parser.add_argument("--api-key-env")
    parser.add_argument("--api-max-tokens", type=int, help="Temporary API completion-budget override.")
    parser.add_argument("--parser-api-base")
    parser.add_argument("--parser-model")
    parser.add_argument("--parser-api-key-env")
    parser.add_argument("--parser-max-tokens", type=int, help="Temporary parser completion-budget override.")
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    data_path = args.data or Path("data/eval/full" if args.full else "data/eval/demo")
    result_path = args.results or Path("results/full_api" if args.full and args.api else "results/full" if args.full else "results/demo_api" if args.api else "results/demo")
    records = _load(data_path)[args.shard_index::args.num_shards]
    samples = model_samples(records)
    raw_outputs = validate_generations(samples, build_backend(_model_spec(args)).generate(samples))
    predictions = []
    details = []
    for index, (record, response) in enumerate(zip(records, raw_outputs)):
        expected = _expected(record)
        task = _base_task(record["type"])
        parsed, value = None, 0.0
        last_error: Exception | None = None
        for _ in range(3):
            try:
                parsed = extract_answer(
                    response,
                    llm_config=args.llm_config,
                    task=task,
                    base_url=args.parser_api_base,
                    model=args.parser_model,
                    api_key_env=args.parser_api_key_env,
                    max_tokens=args.parser_max_tokens,
                )
                value = _score(record["type"], expected, parsed)
                break
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None and parsed is None:
            print(f"answer parsing failed for record {index}: {last_error}", file=sys.stderr)
        prediction = {
            "score": value,
            "question": _question(record),
            "answer": expected,
            "output": response,
            "parsed": parsed,
        }
        predictions.append(prediction)
        details.append({"id": index, "type": record["type"], "score": value})
    _write_results(result_path, predictions, details, args.shard_index)


if __name__ == "__main__":
    main()
