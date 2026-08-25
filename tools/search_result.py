"""Shared result type every search backend returns. Nothing downstream
(retrieval node, evidence manager) needs to know or care whether a result
came from Tavily, Serper, or anything else — it just sees this shape."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0   # relevance score, when the provider gives one