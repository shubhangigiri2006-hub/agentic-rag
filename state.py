"""LangGraph state schema — the shared memory object passed between every
node in the graph. Each node reads what it needs from this and returns a
dict of updates, which LangGraph merges back into the state automatically.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # --- input ---
    question: str
    mode: str  # "A" | "B" | "C" — which pipeline mode we're running

    # --- query planning ---
    search_queries: list[str]

    # --- retrieval / evidence ---
    # Annotated[..., operator.add] means: don't overwrite this list when a
    # node returns it, APPEND to it instead. This is what lets evidence from
    # retrieval round 1 and round 2 both stick around, instead of round 2
    # wiping out round 1.
    evidence: Annotated[list[dict], operator.add]
    retrieval_round: int

    # --- writer ---
    draft_answer: str

    # --- claims + verification ---
    claims: list[dict]
    verification: dict
    verification_round: int

    # --- adaptive controller ---
    decision: str            # one of ACCEPT / REVISE / RETRIEVE_AGAIN / ABSTAIN
    decision_reason: str

    # --- revision ---
    revision_notes: str

    # --- final auditor ---
    audit: dict

    # --- output ---
    final_answer: Optional[str]
    final_status: Optional[str]   # "PASS" | "FAIL" | "ABSTAINED"

    # --- execution trace ---
    # Also accumulates — every node appends one entry here, so at the end
    # you have a full step-by-step log of everything that happened.
    steps: Annotated[list[dict], operator.add]