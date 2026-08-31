"""Dataset and prompt utilities shared by EvalTS model backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .protocol import EvalSample


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load a task directory or a legacy JSONL file in deterministic order."""
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"no task JSONL files found in {path}")
        return [record for file in files for record in load_records(file)]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_answer(record: dict[str, Any]) -> Any:
    return json.loads(record["output"])


def question_text(record: dict[str, Any]) -> str:
    """Remove legacy answer-shape hints to elicit a natural-language answer."""
    return record["input"].split("The answer format is:", maxsplit=1)[0].strip()


def serialized_prompt(sample: EvalSample) -> str:
    """Represent a time series for backends without a native TS processor."""
    return f"{sample.question}\n\nTime series values:\n{json.dumps(sample.timeseries)}"


def sample_id(record: dict[str, Any], index: int) -> str:
    """Read a persisted EvalTS ID, accepting legacy files only for compatibility."""
    value = record.get("id")
    if isinstance(value, str) and value:
        return value
    task = record.get("type", "sample")
    return f"{task}-{index:05d}"


def model_samples(records: list[dict[str, Any]]) -> list[EvalSample]:
    """Convert labelled records to the label-free backend contract."""
    return [
        EvalSample(
            id=sample_id(record, index),
            task=str(record["type"]),
            question=question_text(record),
            timeseries=record["timeseries"],
        )
        for index, record in enumerate(records)
    ]
