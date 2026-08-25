"""Tavily search — 1,000 free searches/month, no credit card.
Get a key: https://app.tavily.com

Implemented as a plain HTTP POST rather than pulling in the official
langchain-tavily package — keeps this one file self-contained and easy to
read line by line, no extra abstraction layer in between.
"""

from __future__ import annotations

import requests

from tools.search_result import SearchResult

TAVILY_URL = "https://api.tavily.com/search"


def tavily_search(query: str, api_key: str, top_k: int = 5) -> list[SearchResult]:
    if not api_key:
        raise EnvironmentError(
            "TAVILY_API_KEY not set in .env. Get a free key at https://app.tavily.com"
        )

    response = requests.post(
        TAVILY_URL,
        json={
            "api_key": api_key,
            "query": query,
            "max_results": top_k,
            "search_depth": "basic",
        },
        timeout=30,
    )
    response.raise_for_status()  # raises an error if Tavily returns a 4xx/5xx status
    data = response.json()

    results = []
    for r in data.get("results", []):
        results.append(
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                score=float(r.get("score", 0.0)),
            )
        )
    return results