"""Signal primitives preserved from the original EvalTS generator.

This module intentionally uses Python's ``random`` module and NumPy's global
random state, matching the original generator's sampling implementation.
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np


Trend = Literal["stable", "up", "down"]


def trend(length: int, kind: Trend | str = "stable") -> tuple[np.ndarray, str]:
    if kind == "random":
        kind = random.choices(["stable", "up", "down"], [0.2, 0.4, 0.4])[0]
    start = random.uniform(0.1, 0.4) if kind == "up" else random.uniform(0.6, 0.9) if kind == "down" else random.uniform(0.3, 0.7)
    end = random.uniform(0.6, 0.9) if kind == "up" else random.uniform(0.1, 0.4) if kind == "down" else start
    if kind == "up" and start >= end:
        end = min(1.0, start + random.uniform(0.1, 0.5))
    if kind == "down" and start <= end:
        end = max(0.0, start - random.uniform(0.1, 0.5))
    shape = "none" if kind == "stable" else random.choice(["linear", "quadratic", "exponential"])
    x = np.arange(length)
    if shape == "linear":
        values = np.linspace(start, end, length)
    elif shape == "quadratic":
        values = ((end - start) / length**2) * x**2 + start
    elif shape == "exponential":
        safe_start, safe_end = (start + 0.1, end + 0.1) if start <= 0 or end <= 0 else (start, end)
        values = safe_start * ((safe_end / safe_start) ** (1 / length)) ** x
    else:
        values = np.full(length, start)
    return values + (np.zeros(length) if kind == "stable" else np.random.normal(0, 0.005, length)), str(kind)


def season(length: int, kind: str = "stable") -> tuple[np.ndarray, int | None]:
    if kind == "random":
        kind = random.choice(["stable", "square", "sine"])
    period = length // random.randint(3, 6)
    amplitude = random.uniform(0.1, 0.5)
    x = np.arange(length)
    if kind == "stable":
        return np.zeros(length), None
    if kind == "square":
        return amplitude * ((x % period) > (period / 2)).astype(float), period
    return amplitude * np.sin(2 * np.pi * x / period) / 2, period


def segment(length: int, kind: str = "none") -> np.ndarray:
    values = np.zeros(length)
    if kind == "random":
        kind = random.choice(["up", "down", "volatile", "platform"])
    if kind == "none":
        return values
    segment_length = int(np.ceil(length * np.random.uniform(0.2, 0.5)))
    start = random.randint(0, length - segment_length - 1)
    end = start + segment_length
    if kind == "up":
        values[start:end] = np.linspace(0, random.uniform(0.7, 1), segment_length)
        values[end:] = values[end - 1]
    elif kind == "down":
        values[start:end] = np.linspace(0, -random.uniform(0.7, 1), segment_length)
        values[end:] = values[end - 1]
    elif kind == "volatile":
        values[start:end] = np.random.uniform(-1, 1, segment_length)
    else:
        values[start:end] = random.uniform(-0.5, 0.5)
    return values


def spikes(length: int, kind: str = "none", count: int = 1) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    values = np.zeros(length)
    if kind == "none":
        return values, []
    indices = sorted(random.sample(range(length), count))
    directions = [random.choice(["up", "down"]) for _ in indices] if kind == "random" else [kind] * len(indices)
    events = []
    for index, direction in zip(indices, directions):
        values[index] = random.uniform(0.5, 1) * (1 if direction == "up" else -1)
        events.append({"point": index + 1, "direction": "upward" if direction == "up" else "downward"})
    return values, events


def composite(length: int, *, seasonal: str = "random", segmented: str = "random", spike_count: int = 0) -> tuple[np.ndarray, dict[str, object]]:
    base, overall_trend = trend(length, "random")
    periodic, period = season(length, seasonal)
    local = segment(length, segmented)
    impulse, events = spikes(length, "random", spike_count) if spike_count else (np.zeros(length), [])
    values = (np.random.uniform(-0.3, 0.3) + base + periodic + local + impulse + np.random.normal(0, 0.05, length)) * np.random.uniform(1, 300)
    return np.round(values, 3), {"trend": overall_trend, "period": period, "spikes": events}


def piecewise_trends(length: int, change_points: list[int]) -> tuple[np.ndarray, list[str]]:
    """Original EvalTS-style piecewise trends with reconstruction scaling."""
    bounds = [0, *change_points, length]
    labels: list[str] = []
    values = np.array([], dtype=float)
    previous = None
    for left, right in zip(bounds[:-1], bounds[1:]):
        label = random.choice(["up", "down", "stable"])
        while label == previous:
            label = random.choice(["up", "down", "stable"])
        previous = label
        labels.append(label)
        size = right - left
        start = float(values[-1]) if len(values) else 0.0
        if label == "stable":
            part = np.full(size, start)
        else:
            magnitude = random.uniform(0.1 * size, 0.5 + 0.1 * size)
            part = np.linspace(start, start + magnitude if label == "up" else start - magnitude, size)
        values = np.concatenate([values, part])
    low, high = float(values.min()), float(values.max())
    if high > low:
        values = (values - low) / (high - low)
    values = (random.uniform(-0.3, 0.3) + values + np.random.normal(0, 0.01, length)) * random.uniform(1, 300)
    return np.round(values, 3), labels


def change_points(length: int, count: int) -> list[int]:
    segment_length = length // (count + 1)
    points = [random.randint(index * segment_length + segment_length // 2, (index + 1) * segment_length + segment_length // 2 - 1) for index in range(count)]
    return sorted(min(length - 2, point) for point in points)
