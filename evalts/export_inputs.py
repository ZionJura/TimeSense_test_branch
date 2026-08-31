"""Export label-free EvalTS inputs for an external model runner.

The exported JSONL rows form the stable runner contract.  An external runner
reads ``id``, ``task``, ``question``, and ``timeseries`` and writes a JSONL
file containing exactly ``id`` and its natural-language ``output``.  EvalTS
then loads that output file through ``backend: external_jsonl`` and applies the
same task-specific parser and metrics used by built-in backends.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .dataset import model_samples, load_records


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def export_inputs(data_path: Path, output_dir: Path) -> None:
    """Write one label-free input JSONL file per EvalTS task."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample in model_samples(load_records(data_path)):
        grouped[sample.task].append(
            {
                "id": sample.id,
                "task": sample.task,
                "question": sample.question,
                "timeseries": sample.timeseries,
            }
        )
    for task, rows in grouped.items():
        _write_jsonl(output_dir / f"{task}.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/eval/demo"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_inputs(args.data, args.output)


if __name__ == "__main__":
    main()
