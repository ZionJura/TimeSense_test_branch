"""Extract a structured EvalTS answer from an open-ended model response."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm import LLM, parse_json


PARSER_INSTRUCTIONS = {
    "change_point": "Return exactly {\"change_points\": [[point, ...], ...]}. One list per input series; use integer indices only.",
    "extreme": "Return exactly {\"extrema\": [{\"min\": number, \"max\": number}, ...]}. One object per input series; retain numeric values without adding fields.",
    "spike": "Return exactly {\"spikes\": [[{\"point\": integer, \"direction\": \"upward\" or \"downward\"}, ...], ...]}. One list per input series; include only detected spikes.",
    "period": "Return exactly {\"periods\": [integer or null, ...]}. One entry per input series; use null for an aperiodic series.",
    "trend": "Return exactly {\"trends\": [\"upward\", \"downward\", or \"stable\", ...]}. One label per input series and no other fields.",
    "segment": "Return exactly {\"change_points\": [point, ...]}. Use integer segment boundary indices only; do not describe trends.",
    "comparison": "Return exactly {\"intervals\": [[start, end], ...]}. Each interval has two integer endpoints; include only intervals selected by the response.",
    "relative": "Return exactly {\"first_trend\": string, \"relative_change\": string}. Canonicalize first_trend to one of \"upward trend\", \"downward trend\", or \"stable state\". Canonicalize relative_change to one of \"an accelerating upward trend\", \"a decelerating upward trend\", \"an accelerating downward trend\", \"a decelerating downward trend\", \"an abrupt upward trend\", \"an abrupt downward trend\", \"upward trend\", \"downward trend\", or \"an abrupt stable state a new stable state\". For example, \"upward\" and \"accelerating\" become \"upward trend\" and \"an accelerating upward trend\" when the second segment is upward.",
    "anomaly_detection": "Return exactly {\"series_status\": [{\"series\": integer, \"is_anomalous\": boolean, optional \"direction\": \"upward\" or \"downward\"}, ...]}. Include every input series; include direction only when is_anomalous is true.",
    "root_cause_analysis": "Return exactly {\"root_cause\": integer, \"propagation_order\": [integer, ...], \"anomalies\": [{\"series\": integer, \"onset\": integer, \"direction\": \"upward\" or \"downward\"}, ...]}. Do not add fields.",
}


def _answer_text(response: str) -> str:
    """Prefer the final answer over an optional hidden-reasoning prefix."""
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[-1]
    return response.strip()


def extract_answer(response: str, *, llm_config: str | Path, task: str, **overrides: Any) -> Any:
    """Parse JSON directly, or use the configured extractor for open-ended text."""
    try:
        return parse_json(response)
    except (TypeError, ValueError):
        pass
    try:
        instruction = PARSER_INSTRUCTIONS[task]
    except KeyError as exc:
        raise ValueError(f"unsupported EvalTS parser task: {task}") from exc
    prompt = (
        "Convert the model response into the requested JSON answer for this EvalTS task. "
        "Do not infer information that is absent from the response. "
        f"{instruction} Return JSON only."
    )
    llm = LLM.from_config(llm_config, "answer_extractor", **overrides)
    text = llm.generate([{"role": "system", "content": prompt}, {"role": "user", "content": f"Model response:\n{_answer_text(response)}"}], json_mode=True)
    return parse_json(text)
