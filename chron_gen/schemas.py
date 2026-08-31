"""Typed ChronGen records shared by generation rules and data writers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SeriesFeatures:
    change_points: list[int]
    trends: list[str]
    period: int | None
    spikes: list[dict[str, int | str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedSeries:
    values: list[float]
    features: SeriesFeatures
