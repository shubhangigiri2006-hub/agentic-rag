"""Tests for the deterministic controller logic — no API calls, no network,
runs instantly. This is the part of the system where a bug would be most
dangerous (an infinite loop), so it gets the most thorough testing.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.controller import build_controller_node, route_from_controller


def test_retrieves_again_when_evidence_missing_and_budget_remains():
    controller = build_controller_node(max_retrieval_rounds=3, max_verification_rounds=3)
    state = {
        "verification": {
            "needs_retrieval": True,
            "needs_revision": False,
            "missing_queries": ["a specific follow-up query"],
        },
        "retrieval_round": 1,
        "verification_round": 1,
        "claims": [{"support_status": "INSUFFICIENT_EVIDENCE"}],
    }
    result = controller(state)
    assert result["decision"] == "RETRIEVE_AGAIN"
    assert result["search_queries"] == ["a specific follow-up query"]


def test_does_not_retrieve_past_budget():
    controller = build_controller_node(max_retrieval_rounds=2, max_verification_rounds=3)
    state = {
        "verification": {"needs_retrieval": True, "needs_revision": False, "missing_queries": ["q"]},
        "retrieval_round": 2,  # already at the budget limit
        "verification_round": 0,
        "claims": [{"support_status": "UNSUPPORTED"}],
    }
    result = controller(state)
    assert result["decision"] != "RETRIEVE_AGAIN"


def test_accepts_when_claims_are_well_supported():
    controller = build_controller_node(max_retrieval_rounds=3, max_verification_rounds=3)
    state = {
        "verification": {"needs_retrieval": False, "needs_revision": False, "missing_queries": []},
        "retrieval_round": 1,
        "verification_round": 1,
        "claims": [{"support_status": "SUPPORTED"}, {"support_status": "SUPPORTED"}],
    }
    result = controller(state)
    assert result["decision"] == "ACCEPT"


def test_abstains_when_budget_exhausted_and_poorly_supported():
    controller = build_controller_node(max_retrieval_rounds=1, max_verification_rounds=1)
    state = {
        "verification": {"needs_retrieval": True, "needs_revision": True, "missing_queries": ["q"]},
        "retrieval_round": 1,     # at budget
        "verification_round": 1,  # at budget
        "claims": [
            {"support_status": "UNSUPPORTED"},
            {"support_status": "CONTRADICTED"},
            {"support_status": "SUPPORTED"},
        ],
    }
    result = controller(state)
    assert result["decision"] == "ABSTAIN"


def test_never_loops_past_budget_no_matter_how_bad_the_verification_is():
    """The critical property: even if the verifier is stubbornly convinced
    more retrieval AND more revision are needed forever, once both budgets
    are hit, the controller must stop — never RETRIEVE_AGAIN or REVISE."""
    controller = build_controller_node(max_retrieval_rounds=2, max_verification_rounds=2)
    state = {
        "verification": {"needs_retrieval": True, "needs_revision": True, "missing_queries": ["still missing"]},
        "retrieval_round": 2,      # over budget
        "verification_round": 2,   # over budget
        "claims": [{"support_status": "UNSUPPORTED"}],
    }
    result = controller(state)
    assert result["decision"] in ("ACCEPT", "ABSTAIN")  # never a loop action


def test_route_from_controller_maps_every_decision_to_a_node():
    assert route_from_controller({"decision": "ACCEPT"}) == "audit"
    assert route_from_controller({"decision": "REVISE"}) == "revise"
    assert route_from_controller({"decision": "RETRIEVE_AGAIN"}) == "retrieve"
    assert route_from_controller({"decision": "ABSTAIN"}) == "abstain"