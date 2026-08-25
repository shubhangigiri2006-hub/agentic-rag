"""Core data types. Plain dataclasses — simple, no framework magic, easy
to read. These get converted to plain dicts when they enter LangGraph's
state (see state.py) since LangGraph works best with JSON-serializable
values, not custom objects.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


def new_id(prefix: str) -> str:
    """Generates a short unique id like 'ev_a1b2c3d4e5'."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class SupportStatus(str, Enum):
    """Every claim gets exactly one of these after verification."""
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNVERIFIED = "UNVERIFIED"   # default, before the verifier runs


class ControllerDecision(str, Enum):
    """What the adaptive controller decides to do next."""
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    RETRIEVE_AGAIN = "RETRIEVE_AGAIN"
    ABSTAIN = "ABSTAIN"


@dataclass
class Evidence:
    """One retrieved piece of information, with everything needed to trace
    a claim back to its source."""
    evidence_id: str
    source: str          # URL it came from
    title: str
    text: str             # the actual snippet/content
    relevance: float       # search engine's relevance score
    retrieval_round: int   # which retrieval pass found this (0 = first)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Claim:
    """One atomic factual statement extracted from the writer's answer."""
    claim_id: str
    text: str
    evidence_ids: list[str] = field(default_factory=list)
    support_status: str = SupportStatus.UNVERIFIED.value

    def to_dict(self) -> dict:
        return asdict(self)