"""Stable contracts between EvalTS data, model runners, and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalSample:
    """Label-free model input with a stable identifier."""

    id: str
    task: str
    question: str
    timeseries: list[list[float]]


@dataclass(frozen=True)
class Generation:
    """A model response associated with exactly one EvalTS sample."""

    id: str
    output: str


def validate_generations(samples: list[EvalSample], generations: list[Generation]) -> list[str]:
    """Return outputs in sample order and reject missing, duplicate, or foreign IDs."""
    expected = {sample.id for sample in samples}
    by_id: dict[str, str] = {}
    for generation in generations:
        if generation.id not in expected:
            raise ValueError(f"backend returned an unknown sample id: {generation.id}")
        if generation.id in by_id:
            raise ValueError(f"backend returned a duplicate sample id: {generation.id}")
        if not isinstance(generation.output, str):
            raise TypeError(f"backend output for {generation.id} must be text")
        by_id[generation.id] = generation.output
    missing = expected - set(by_id)
    if missing:
        raise ValueError(f"backend returned missing outputs for {len(missing)} sample IDs")
    return [by_id[sample.id] for sample in samples]
