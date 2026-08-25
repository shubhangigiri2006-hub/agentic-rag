"""Search provider factory.

get_search_fn(config) returns a plain function: call it with (query, top_k)
and get back a list[SearchResult]. The retrieval node calls this function
without knowing or caring which provider actually answered — that's the
whole point of this file.
"""

from __future__ import annotations

from config import Config
from tools.tavily_tool import tavily_search


def get_search_fn(config: Config):
    if config.search_provider == "tavily":
        api_key = config.tavily_api_key
        # Returns a function that already "remembers" the api_key, so
        # callers elsewhere just do search_fn(query, top_k) — they never
        # need to know or pass around the API key themselves.
        return lambda query, top_k=config.search_top_k: tavily_search(query, api_key, top_k)

    raise ValueError(
        f"Unknown search_provider: {config.search_provider!r} (only 'tavily' is wired up so far)"
    )