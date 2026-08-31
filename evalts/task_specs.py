"""Task names and scoring constants for the EvalTS protocol."""

from __future__ import annotations


POSITION_TOLERANCE = 3
CHANGE_POINT_TOLERANCE = 5
SEGMENT_TOLERANCE = 5
PERIOD_TOLERANCE = 5

TASK_SCORING = {
    "change_point": "f1",
    "extreme": "accuracy",
    "spike": "f1",
    "period": "accuracy",
    "trend": "accuracy",
    "segment": "f1",
    "comparison": "f1",
    "relative": "legacy_relative",
    "anomaly_detection": "f1",
    "root_cause_analysis": "f1",
}


def base_task(task: str) -> str:
    """Map a uni/multi EvalTS task name to its common scoring family."""
    return task.removeprefix("uni_").removeprefix("multi_")
