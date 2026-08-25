"""Retrieval node — deterministic, no LLM call.

Input: state["search_queries"] (set by the planner, or later by the
controller when it asks for more evidence).
Output: new Evidence entries appended to state["evidence"], and
retrieval_round incremented by one.
"""

from __future__ import annotations

from schemas import Evidence, new_id


def build_retrieval_node(search_fn, top_k: int = 5):
    """search_fn is whatever get_search_fn(config) returned — a callable
    that takes (query, top_k) and returns list[SearchResult]."""

    def retrieval_node(state):
        queries = state.get("search_queries") or [state["question"]]
        current_round = state.get("retrieval_round", 0)

        new_evidence = []
        # Seed seen_urls with every URL already in state, not just this
        # round's — otherwise the same source gets re-added every round.
        existing_evidence = state.get("evidence", [])
        seen_urls = set(e["source"] for e in existing_evidence)

        new_evidence: list[dict] = []
        seen_urls = set()   # avoid storing the same URL twice across queries

        for query in queries:
            try:
                results = search_fn(query, top_k)
            except Exception:
                # A failed search call shouldn't crash the whole run — treat
                # it as "this query found nothing" and move on. The
                # verifier/controller downstream will notice the evidence
                # gap and can decide to retry.
                continue

            for r in results:
                if r.url in seen_urls:
                    continue
                seen_urls.add(r.url)

                ev = Evidence(
                    evidence_id=new_id("ev"),
                    source=r.url,
                    title=r.title,
                    text=r.snippet,
                    relevance=r.score,
                    retrieval_round=current_round,
                    metadata={"query": query},
                )
                new_evidence.append(ev.to_dict())

        return {
            "evidence": new_evidence,
            "retrieval_round": current_round + 1,
            "steps": [
                {
                    "step": "retrieval",
                    "round": current_round,
                    "queries": queries,
                    "num_new_evidence": len(new_evidence),
                }
            ],
        }

    return retrieval_node