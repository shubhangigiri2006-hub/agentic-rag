"""Shared JSON-extraction helper. Every node that asks the LLM for
structured JSON output was duplicating this parsing logic — this file is
the single place it lives now.
"""

from __future__ import annotations

import json
import re


def extract_json(text: str):
    """Best-effort extraction of a JSON object or array from raw LLM text.
    Handles the LLM wrapping its answer in ```json fences, or adding a
    stray sentence of preamble, despite being told not to."""
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    brace_start = text.find("{")
    bracket_start = text.find("[")
    if bracket_start != -1 and (brace_start == -1 or bracket_start < brace_start):
        candidates_in_order = [("[", "]"), ("{", "}")]
    else:
        candidates_in_order = [("{", "}"), ("[", "]")]

    for open_c, close_c in candidates_in_order:
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise ValueError("Could not extract JSON from LLM output: " + text[:200])