"""Query Planner node.

Input: state["question"] — the raw user question.
Output: state["search_queries"] — 2-4 concrete search queries.

Why not just search for the raw question directly? Because a broad question
like "how did computer architecture change in the 1940s-50s" searches badly
as one string, but breaks cleanly into several concrete queries a search
engine actually handles well.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = (
    "You are a research query planner. Given a user's question, produce 2-4 "
    "concrete, distinct web search queries that together would gather enough "
    "evidence to answer it. Respond with ONLY a JSON array of strings, "
    'e.g. ["query one", "query two"]. No other text, no markdown formatting.'
)


def _extract_json_array(text: str) -> list:
    """LLMs sometimes wrap JSON in ```json fences despite instructions not
    to. This strips that before parsing, and falls back gracefully if the
    output still isn't valid JSON."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def build_query_planner_node(llm):
    """Returns a node function with `llm` baked in via closure — same
    pattern as get_search_fn in tools/search_factory.py."""

    def query_planner_node(state: dict) -> dict:
        question = state["question"]
        try:
            response = llm.invoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
            )
            queries = _extract_json_array(response.content)
            if not isinstance(queries, list) or not queries:
                raise ValueError("planner did not return a non-empty list")
            queries = [str(q) for q in queries][:4]
        except Exception:
            # If the LLM output can't be parsed, don't fail the whole run —
            # just fall back to searching the raw question.
            queries = [question]

        return {
            "search_queries": queries,
            "steps": [{"step": "query_planner", "queries": queries}],
        }

    return query_planner_node