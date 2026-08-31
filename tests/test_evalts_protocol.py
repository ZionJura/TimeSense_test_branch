"""Dependency-light regression checks for the public EvalTS protocol."""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from evalts.backends import ExternalJSONLBackend, ModelSpec
from evalts.dataset import load_records, model_samples
from evalts.evaluate import _score
from evalts.protocol import Generation, validate_generations


class EvalTSProtocolTest(unittest.TestCase):
    def test_demo_split_is_task_partitioned(self) -> None:
        records = load_records(Path("data/eval/demo"))
        self.assertEqual(len(records), 3)
        self.assertEqual(
            Counter(record["type"] for record in records),
            {"uni_trend": 1, "segment": 1, "root_cause_analysis": 1},
        )
        self.assertTrue(all(isinstance(record.get("id"), str) for record in records))

    def test_generation_ids_define_the_runner_contract(self) -> None:
        samples = model_samples(load_records(Path("data/eval/demo")))
        outputs = validate_generations(
            samples,
            [Generation(sample.id, f"answer for {sample.id}") for sample in reversed(samples)],
        )
        self.assertEqual(outputs, [f"answer for {sample.id}" for sample in samples])
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_generations(samples, [Generation(samples[0].id, "only one")])

    def test_external_jsonl_backend_uses_ids(self) -> None:
        samples = model_samples(load_records(Path("data/eval/demo")))
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "outputs.jsonl"
            output_path.write_text(
                "\n".join(
                    f'{{"id": "{sample.id}", "output": "answer"}}' for sample in samples
                ) + "\n",
                encoding="utf-8",
            )
            backend = ExternalJSONLBackend(
                ModelSpec(backend="external_jsonl", output_path=str(output_path))
            )
            self.assertEqual(validate_generations(samples, backend.generate(samples)), ["answer"] * len(samples))

    def test_legacy_tolerance_boundaries(self) -> None:
        answer = {"change_points": [[10]]}
        self.assertEqual(_score("uni_change_point", answer, {"change_points": [[15]]}), 1.0)
        self.assertEqual(_score("uni_change_point", answer, {"change_points": [[16]]}), 0.0)
        self.assertEqual(_score("uni_period", {"periods": [20]}, {"periods": [25]}), 1.0)
        self.assertEqual(_score("uni_period", {"periods": [20]}, {"periods": [26]}), 0.0)


if __name__ == "__main__":
    unittest.main()
