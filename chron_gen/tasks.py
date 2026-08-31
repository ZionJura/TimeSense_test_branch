"""Deterministic ChronGen rules and camera-ready task prompts.

The prompt strings below are the public camera-ready prompts. Keep their wording
and ``<ts><ts/>`` placeholders unchanged when modifying the generator.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

import numpy as np

from .legacy_primitives import composite as legacy_composite
from .legacy_primitives import change_points as legacy_change_points
from .legacy_primitives import piecewise_trends as legacy_piecewise_trends
from .legacy_primitives import season as legacy_season
from .legacy_primitives import trend as legacy_trend
from .schemas import GeneratedSeries, SeriesFeatures

ATOMIC_TASKS = ("change_point", "extreme", "spike", "trend", "period")
MOLECULAR_TASKS = ("segment", "comparison", "relative")
COMPOSITIONAL_TASKS = ("anomaly_detection", "root_cause_analysis")
TASKS = ATOMIC_TASKS + MOLECULAR_TASKS + COMPOSITIONAL_TASKS
EVALTS_TASKS = (
    "uni_change_point", "uni_extreme", "uni_period", "uni_spike", "uni_trend",
    "multi_change_point", "multi_extreme", "multi_period", "multi_spike", "multi_trend",
    "segment", "comparison", "relative", "anomaly_detection", "root_cause_analysis",
)
TREND_SLOPES = {"downward": -0.08, "stable": 0.0, "upward": 0.08}


def _validate_shape(length: int, dimensions: int) -> None:
    if not 24 <= length <= 512:
        raise ValueError("length must be between 24 and 512")
    if not 1 <= dimensions <= 5:
        raise ValueError("dimensions must be between 1 and 5")


def _change_points(length: int, segments: int, rng: np.random.Generator) -> list[int]:
    if segments == 1:
        return []
    minimum = 8
    if length < segments * minimum:
        raise ValueError("length is too short for the requested number of segments")
    lengths = rng.multinomial(length - segments * minimum, np.full(segments, 1 / segments)) + minimum
    return np.cumsum(lengths)[:-1].astype(int).tolist()


def _series(length: int, trends: Sequence[str], rng: np.random.Generator, *, period: int | None = None, spike_count: int = 0) -> GeneratedSeries:
    points = _change_points(length, len(trends), rng)
    bounds = [0, *points, length]
    values = np.empty(length, dtype=np.float64)
    current = float(rng.uniform(-2.0, 2.0))
    for index, trend in enumerate(trends):
        left, right = bounds[index], bounds[index + 1]
        slope = TREND_SLOPES[trend] * rng.uniform(0.75, 1.25)
        segment = current + slope * np.arange(right - left)
        if period is not None:
            segment += rng.uniform(0.2, 0.5) * np.sin(2 * np.pi * np.arange(left, right) / period + rng.uniform(0, 2 * np.pi))
        segment += rng.normal(0.0, 0.04, right - left)
        values[left:right] = segment
        current = float(segment[-1] + slope)
    spikes: list[dict[str, int | str]] = []
    if spike_count:
        for point in sorted(rng.choice(np.arange(2, length - 2), size=min(spike_count, length - 4), replace=False).tolist()):
            direction = "upward" if rng.random() >= 0.5 else "downward"
            values[point] += (1 if direction == "upward" else -1) * rng.uniform(4.0, 6.0) * max(float(np.std(values)), 0.5)
            spikes.append({"point": int(point + 1), "direction": direction})
    return GeneratedSeries(np.round(values, 4).tolist(), SeriesFeatures(points, list(trends), period, spikes))


def _placeholder(index: int, length: int) -> str:
    return f"Time series {index} has length {length}: <ts><ts/>."


def _multi_prompt(task: str, length: int, dimensions: int) -> str:
    """Return the anonymous-code multi-series prompt without paraphrasing."""
    series = ", ".join(
        f"Time series {index + 1} is a time series of length {length}: <ts><ts/>;"
        for index in range(dimensions)
    )
    if task == "change_point":
        return f"Observe the following multiple time series, where {series} Please identify for each time series whether there are any change points. If so, at which points do the change points occur?"
    if task == "extreme":
        return f"Observe the following multiple time series, where {series} Please identify the minimum and maximum values for each time series in order."
    if task == "period":
        return f"Observe the following multiple time series, where {series} Please identify for each time series whether there is any seasonal variation. If so, what is the period of the seasonal variation?"
    if task == "trend":
        return f"Observe the following multiple time series, where {series} Please identify for each time series the overall trend: is it upward, downward, or stable?"
    if task == "spike":
        return f"Given multiple time series, where{series}. Please answer in order whether each time series contains a spike. If so, at which point does the spike occur? Also, please indicate the direction of the spike."
    raise ValueError(f"no multi-series prompt for {task}")


def _anomaly_detection_prompt(
    length: int, interval: tuple[int, int], rules: Sequence[str],
) -> str:
    """Return the legacy rule-based anomaly-detection instruction verbatim."""
    series = "".join(
        f"Time series {index + 1} is a time series of length {length}:<ts><ts/>\n"
        for index in range(len(rules))
    )
    start, end = interval
    instruction = (
        "Under normal circumstances, the time series remain stable with minor spikes. "
        "In anomalous states, there will be consistent anomalies throughout the entire interval. "
        "Different time series have different types of anomalies (upward and downward; upward "
        "indicates that the values in this interval are higher than normal, while downward indicates "
        "that the values in this interval are lower than normal. Note that even if there is a platform "
        "change, it is still considered normal if it does not meet the anomaly criteria). Please "
        f"determine which series exhibit anomalies in the interval {start}-{end}, and specify whether "
        "they are upward or downward anomalies. The anomaly determination rules for each series are as follows:\n"
    )
    return "Observe the following multiple time series, where " + series + "。\n" + instruction + "".join(
        f"- Series {index + 1}: {rule}\n" for index, rule in enumerate(rules)
    )


def _root_cause_prompt(length: int, rules: Sequence[str]) -> str:
    """Return the legacy root-cause-analysis instruction verbatim."""
    series = "".join(
        f"Time series {index + 1} is a time series of length {length}:<ts><ts/>\n"
        for index in range(len(rules))
    )
    instruction = (
        "Under normal circumstances, the time series remain stable with minor spikes. "
        "In anomalous states, there will be consistent anomalies throughout the entire interval. "
        "Different time series have different types of anomalies (upward and downward; upward "
        "indicates that the values in this interval are higher than normal, while downward indicates "
        "that the values in this interval are lower than normal. Note that even if there is a platform "
        "change, it is still considered normal if it does not meet the anomaly criteria). Please identify "
        "the anomalies present in the above time series and attempt to locate the root causes of the anomalies.\n"
        "The causes of anomalies are divided into two categories: overall environmental changes causing "
        "all time series to exhibit anomalies in the same interval, without any temporal order of anomaly "
        "intervals; propagation of internal anomalies within the time series, where the earliest occurring "
        "anomaly is the root cause, and the timing of anomaly occurrence reflects the propagation path. "
        "Please use the relevant information to determine whether each time series has anomalies, locate "
        "the root cause of the anomalies, and trace the propagation path (the path is represented as "
        "Time Series i -> Time Series j -> ...). The anomaly determination rules for each series are as follows:\n"
    )
    return "Observe the following multiple time series, where " + series + "。\n" + instruction + "".join(
        f"- Series {index + 1}: {rule}\n" for index, rule in enumerate(rules)
    )


def _base(task: str, level: str, series: list[GeneratedSeries], question: str, answer: Any, seed: int | None) -> dict[str, Any]:
    return {"task": task, "level": level, "question": question, "answer": answer, "timeseries": [item.values for item in series], "features": [item.features.to_dict() for item in series], "generator": {"name": "ChronGen", "version": "0.1.0", "seed": seed}}


def _trends(rng: np.random.Generator, count: int) -> list[str]:
    labels = np.array(["downward", "stable", "upward"])
    values = rng.choice(labels, size=count, replace=True).tolist()
    for index in range(1, len(values)):
        if values[index] == values[index - 1]:
            values[index] = str(rng.choice(labels[labels != values[index - 1]]))
    return [str(value) for value in values]


def generate_sample(task: str, *, length: int = 128, dimensions: int = 1, seed: int | None = None) -> dict[str, Any]:
    """Generate a rule-labelled ChronGen sample without I/O or model calls."""
    _validate_shape(length, dimensions)
    if task not in TASKS:
        raise ValueError(f"unknown ChronGen task: {task}")
    rng = np.random.default_rng(seed)
    if task in ATOMIC_TASKS:
        series = []
        for _ in range(dimensions):
            if task == "change_point":
                has_change_point = dimensions == 1 or rng.random() < 0.6
                points = legacy_change_points(length, 1) if has_change_point else []
                values, labels = legacy_piecewise_trends(length, points)
                series.append(GeneratedSeries(values.tolist(), SeriesFeatures(points, [label.replace("up", "upward").replace("down", "downward") for label in labels], None, [])))
            elif task == "spike":
                base, trend = legacy_trend(length, "random")
                scale = np.random.uniform(1, 100)
                values = (np.random.uniform(-0.3, 0.3) + base + np.random.normal(0, 0.005, length)) * scale
                events = []
                if rng.random() < 0.7:
                    point = int(rng.integers(1, length - 1))
                    direction = "upward" if rng.random() >= 0.5 else "downward"
                    values[point] += np.random.uniform(0.5, 1.0) * scale * (1 if direction == "upward" else -1)
                    events = [{"point": point + 1, "direction": direction}]
                series.append(GeneratedSeries(np.round(values, 3).tolist(), SeriesFeatures([], [trend.replace("up", "upward").replace("down", "downward")], None, events)))
            elif task == "period":
                seasonal_kind = str(rng.choice(["square", "sine"])) if rng.random() < 0.6 else "stable"
                base, trend = legacy_trend(length, "random")
                seasonal, period = legacy_season(length, seasonal_kind)
                values = (np.random.uniform(-0.3, 0.3) + base + seasonal + np.random.normal(0, 0.05, length)) * np.random.uniform(1, 300)
                series.append(GeneratedSeries(np.round(values, 3).tolist(), SeriesFeatures([], [trend.replace("up", "upward").replace("down", "downward")], period, [])))
            elif task == "extreme":
                values, meta = legacy_composite(length, spike_count=int(rng.integers(1, 7)))
                trend = str(meta["trend"]).replace("up", "upward").replace("down", "downward")
                series.append(GeneratedSeries(values.tolist(), SeriesFeatures([], [trend], meta["period"], list(meta["spikes"]))))
            else:
                base, trend = legacy_trend(length, str(rng.choice(["up", "down", "stable"])))
                values = (np.random.uniform(-0.3, 0.3) + base + np.random.normal(0, 0.005, length)) * np.random.uniform(1, 300)
                series.append(GeneratedSeries(np.round(values, 3).tolist(), SeriesFeatures([], [trend.replace("up", "upward").replace("down", "downward")], None, [])))
        if task == "change_point":
            question = f"Observe the following time series of length {length}: <ts><ts/>. Please identify whether there are any change points in this time series. If so, at which points do the change points occur?"
            return _base(task, "atomic", series, question if dimensions == 1 else _multi_prompt(task, length, dimensions), {"change_points": [item.features.change_points for item in series]}, seed)
        if task == "extreme":
            question = f"Observe the following time series of length {length}: <ts><ts/>. Please identify the minimum and maximum values of this time series."
            return _base(task, "atomic", series, question if dimensions == 1 else _multi_prompt(task, length, dimensions), {"extrema": [{"min": min(item.values), "max": max(item.values)} for item in series]}, seed)
        if task == "spike":
            question = f"Given a time series of length {length}:<ts><ts/>, please answer the following question. Does this time series contain a spike? If so, at which point does the spike occur?"
            if series[0].features.spikes:
                question += " Also, please indicate the direction of the spike."
            return _base(task, "atomic", series, question if dimensions == 1 else _multi_prompt(task, length, dimensions), {"spikes": [item.features.spikes for item in series]}, seed)
        if task == "trend":
            question = f"Observe the following time series of length {length}: <ts><ts/>. Please identify the overall trend of this time series: is it upward, downward, or stable?"
            return _base(task, "atomic", series, question if dimensions == 1 else _multi_prompt(task, length, dimensions), {"trends": [item.features.trends[0] for item in series]}, seed)
        question = f"Observe the following time series of length {length}: <ts><ts/>. Please identify whether there is any seasonal variation in this time series. If so, what is the period of the seasonal variation?"
        return _base(task, "atomic", series, question if dimensions == 1 else _multi_prompt(task, length, dimensions), {"periods": [item.features.period for item in series]}, seed)
    if task == "segment":
        maximum = min(4, max(1, length // 24))
        count = int(rng.integers(1, maximum + 1))
        points = legacy_change_points(length, count)
        values, labels = legacy_piecewise_trends(length, points)
        item = GeneratedSeries(values.tolist(), SeriesFeatures(points, [label.replace("up", "upward").replace("down", "downward") for label in labels], None, []))
        question = f"Observe the following time series of length {length}: <ts><ts/>. Please identify the points at which trend change points occur in this time series. Please list all change points in order (if there are multiple change points, please list them in ascending order)."
        return _base(task, "molecular", [item], question, {"change_points": item.features.change_points}, seed)
    if task == "comparison":
        first_values, _ = legacy_composite(length, spike_count=int(rng.integers(1, 7)))
        second_values, _ = legacy_composite(length, spike_count=int(rng.integers(1, 7)))
        first = GeneratedSeries(first_values.tolist(), SeriesFeatures([], ["stable"], None, []))
        has_difference = rng.random() < 0.7
        change_length = int(rng.integers(max(3, length // 5), max(length // 4, 5) + 1))
        start = int(rng.integers(0, length - change_length))
        end = start + change_length - 1
        if has_difference:
            delta = np.linspace(0, rng.uniform(0.7, 1.0) * (1 if rng.random() >= 0.5 else -1), change_length)
            second_values[start:end + 1] += delta * np.random.uniform(1, 300)
        second = GeneratedSeries(np.round(second_values, 3).tolist(), SeriesFeatures([], ["stable"], None, []))
        question = f"Observe the following two time series of length {length}: Time series A is <ts><ts/>, and time series B is <ts><ts/>. Please answer the following question: Does time series B have any significant differences compared to time series A? If so, please specify the interval where the differences occur (from which point to which point)."
        return _base(task, "molecular", [first, second], question, {"intervals": [[start + 1, end + 1]] if has_difference else []}, seed)
    if task == "relative":
        change_point = int(rng.integers(length // 4, 3 * length // 4))
        first_kind = random.choices(["up", "down", "stable"], [0.4, 0.4, 0.2])[0]
        if first_kind == "up":
            second_kind = random.choices(["up", "down", "stable"], [0.6, 0.1, 0.3])[0]
        elif first_kind == "down":
            second_kind = random.choices(["up", "down", "stable"], [0.1, 0.6, 0.3])[0]
        else:
            second_kind = random.choices(["up", "down"], [0.5, 0.5])[0]

        def make_trend(size: int, kind: str, start: float, previous_slope: float) -> tuple[np.ndarray, float]:
            if kind == "stable":
                return np.full(size, start), 0.0
            slope: float | None = None
            minimum_change = 0.5
            while slope is None or abs(slope - previous_slope) < 0.3:
                end = start + random.uniform(minimum_change, minimum_change + 0.5) * (1 if kind == "up" else -1)
                minimum_change += 0.5
                slope = (end - start) / size
            return np.linspace(start, end, size), slope

        first_values, first_slope = make_trend(change_point, first_kind, 0.0, 0.0)
        second_values, second_slope = make_trend(length - change_point, second_kind, float(first_values[-1]), first_slope)
        values = np.concatenate([first_values, second_values])
        low, high = float(values.min()), float(values.max())
        if high > low:
            values = (values - low) / (high - low)
        values = (random.uniform(-0.3, 0.3) + values + np.random.normal(0, 0.01, length)) * random.uniform(1, 100)
        values = np.round(values, 3)
        trend_label = {"up": "upward trend", "down": "downward trend", "stable": "stable state"}
        relative_change = trend_label[second_kind]
        if first_kind != "stable":
            if second_kind == first_kind:
                if abs(second_slope) > abs(first_slope) + 0.3:
                    relative_change = f"an accelerating {relative_change}"
                elif abs(second_slope) < abs(first_slope) - 0.3:
                    relative_change = f"a decelerating {relative_change}"
            else:
                relative_change = f"an abrupt {relative_change}"
                if second_kind == "stable":
                    relative_change += " a new stable state"
        item = GeneratedSeries(
            values.tolist(),
            SeriesFeatures([change_point], [first_kind.replace("up", "upward").replace("down", "downward"), second_kind.replace("up", "upward").replace("down", "downward")], None, []),
        )
        question = f"Observe the following time series of length {length}: <ts><ts/>. Please answer the following questions: The trend change point in this time series occurs at point {change_point + 1}. What is the trend of the first segment: upward, downward, or stable? Compared to the first segment, is the trend of the second segment accelerating, decelerating, undergoing a sudden change, or entering a new stable state? Please answer these two questions in order."
        return _base(task, "molecular", [item], question, {"first_trend": trend_label[first_kind], "relative_change": relative_change}, seed)
    dimensions = max(3, dimensions)
    series = []
    for _ in range(dimensions):
        base, _ = legacy_trend(length, "stable")
        scale = np.random.uniform(1, 300)
        values = (np.random.uniform(-0.3, 0.3) + base + np.random.normal(0, 0.05, length)) * scale
        series.append(GeneratedSeries(np.round(values, 3).tolist(), SeriesFeatures([], ["stable"], None, [])))
    root = int(rng.integers(0, dimensions))
    order = [root, *[index for index in range(dimensions) if index != root]]
    onset = int(rng.integers(length // 4, length // 2))
    anomalies = []
    for rank, index in enumerate(order[: int(rng.integers(2, dimensions + 1))]):
        point = min(onset + rank * max(2, length // 12), length - 2)
        direction = "upward" if rng.random() >= 0.5 else "downward"
        values = np.array(series[index].values)
        end = min(point + int(rng.integers(max(3, length // 10), max(4, length // 5) + 1)), length)
        scale = max(float(np.std(values)), 1.0)
        values[point:end] += (1 if direction == "upward" else -1) * np.random.uniform(0.7, 1.0) * scale
        series[index] = GeneratedSeries(np.round(values, 3).tolist(), series[index].features)
        anomalies.append({"series": index + 1, "onset": point + 1, "direction": direction})
    if task == "anomaly_detection":
        interval_length = int(rng.integers(max(6, length // 10), max(7, length // 5) + 1))
        interval_start = int(rng.integers(1, length - interval_length + 1))
        interval_end = interval_start + interval_length - 1
        rules: list[str] = []
        anomalies = []
        series = []
        for index in range(dimensions):
            rule_direction = "upward" if rng.random() >= 0.5 else "downward"
            rules.append(f"deviations {'above' if rule_direction == 'upward' else 'below'} normal are considered anomalies")
            base, _ = legacy_trend(length, "stable")
            scale = np.random.uniform(1, 300)
            values = (np.random.uniform(-0.3, 0.3) + base + np.random.normal(0, 0.05, length)) * scale
            actual_direction = "upward" if rng.random() >= 0.5 else "downward"
            has_deviation = rng.random() >= 0.35
            is_anomalous = has_deviation and actual_direction == rule_direction
            if has_deviation:
                values[interval_start - 1:interval_end] += (np.random.uniform(0.7, 1.0) if actual_direction == "upward" else -np.random.uniform(0.5, 1.0)) * scale
            if is_anomalous:
                anomalies.append({"series": index + 1, "onset": interval_start, "direction": actual_direction})
            series.append(GeneratedSeries(np.round(values, 3).tolist(), SeriesFeatures([], ["stable"], None, [])))
        status_by_series = {entry["series"]: entry for entry in anomalies}
        series_status = [
            {"series": index + 1, "is_anomalous": index + 1 in status_by_series,
             **({"direction": status_by_series[index + 1]["direction"]} if index + 1 in status_by_series else {})}
            for index in range(dimensions)
        ]
        question = _anomaly_detection_prompt(length, (interval_start, interval_end), rules)
        return _base(task, "compositional", series, question, {"series_status": series_status}, seed)
    anomaly_by_series = {entry["series"]: entry for entry in anomalies}
    rules = [
        f"Values {'higher' if anomaly_by_series.get(index + 1, {}).get('direction', 'upward') == 'upward' else 'lower'} than normal are considered anomalies"
        if index + 1 in anomaly_by_series
        else f"Values {'higher' if rng.random() >= 0.5 else 'lower'} than normal are considered anomalies"
        for index in range(dimensions)
    ]
    return _base(task, "compositional", series, _root_cause_prompt(length, rules), {"root_cause": root + 1, "propagation_order": [item["series"] for item in anomalies], "anomalies": anomalies}, seed)


def generate_eval_sample(task: str, *, seed: int, index: int = 0) -> dict[str, Any]:
    """Generate an EvalTS sample using the camera-ready ChronGen rules."""
    mapping = {
        "uni_change_point": ("change_point", 1), "uni_extreme": ("extreme", 1), "uni_period": ("period", 1), "uni_spike": ("spike", 1), "uni_trend": ("trend", 1),
        "multi_change_point": ("change_point", 0), "multi_extreme": ("extreme", 0), "multi_period": ("period", 0), "multi_spike": ("spike", 0), "multi_trend": ("trend", 0),
        "segment": ("segment", 1), "comparison": ("comparison", 2), "relative": ("relative", 1), "anomaly_detection": ("anomaly_detection", 0), "root_cause_analysis": ("root_cause_analysis", 0),
    }
    if task not in mapping:
        raise ValueError(f"unknown EvalTS task: {task}")
    random_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(seed + index)
    np.random.seed(seed + index)
    base_task, dimensions = mapping[task]
    if dimensions == 0:
        dimensions = int(np.random.default_rng(seed + index).integers(3, 6))
    if task == "segment":
        length = random.choice((64, 128, 256))
    elif task in {"anomaly_detection", "root_cause_analysis"}:
        length = 305 if random.random() > 0.9 else random.randint(128, 512)
    else:
        length = random.choice((64, 128, 256, 336, 512))
    try:
        sample = generate_sample(base_task, length=length, dimensions=dimensions, seed=seed + index)
        sample["task"] = task
        return sample
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
