"""Adaptive Retrieval Controller — deterministic, no LLM call.

Reads the verifier's structured output plus round counters against
config-defined budgets, and returns exactly one of:

  ACCEPT          — claims are well-supported enough to finalize
  REVISE          — fixable with existing evidence, no new retrieval needed
  RETRIEVE_AGAIN  — missing evidence identified, budget allows another round
  ABSTAIN         — budget exhausted and support is too weak to finalize

This is what makes the loop provably bounded: every branch below compares
numbers against explicit limits. There's no path where the loop can run
forever, no matter what the verifier says.
"""

from __future__ import annotations

SUPPORTED = "SUPPORTED"


def build_controller_node(max_retrieval_rounds: int, max_verification_rounds: int):
    def controller_node(state: dict) -> dict:
        verification = state.get("verification", {})
        retrieval_round = state.get("retrieval_round", 0)
        verification_round = state.get("verification_round", 0)
        claims = state.get("claims", [])

        needs_retrieval = verification.get("needs_retrieval", False)
        needs_revision = verification.get("needs_revision", False)
        missing_queries = verification.get("missing_queries", [])

        update: dict = {}

        if needs_retrieval and retrieval_round < max_retrieval_rounds and missing_queries:
            decision = "RETRIEVE_AGAIN"
            reason = (
                f"{len(missing_queries)} missing-evidence quer(y/ies) identified; "
                f"retrieval round {retrieval_round}/{max_retrieval_rounds} budget remains."
            )
            # This is what feeds the NEXT retrieval round — the controller
            # sets search_queries to the verifier's suggested fixes, not the
            # original question.
            update["search_queries"] = missing_queries

        elif needs_revision and verification_round < max_verification_rounds:
            decision = "REVISE"
            reason = (
                f"Unresolved or partially supported claims exist; "
                f"verification round {verification_round}/{max_verification_rounds} budget remains."
            )

        else:
            total = len(claims)
            supported = sum(1 for c in claims if c.get("support_status") == SUPPORTED)
            support_ratio = (supported / total) if total else 0.0

            if total == 0 or support_ratio < 0.5:
                decision = "ABSTAIN"
                reason = (
                    f"Budget exhausted (retrieval {retrieval_round}/{max_retrieval_rounds}, "
                    f"verification {verification_round}/{max_verification_rounds}) with only "
                    f"{supported}/{total} claims supported — abstaining rather than finalizing "
                    "a weakly-grounded answer."
                )
            else:
                decision = "ACCEPT"
                reason = (
                    f"Budget exhausted or no further issues flagged; "
                    f"{supported}/{total} claims supported — accepting for final audit."
                )

        update.update(
            {
                "decision": decision,
                "decision_reason": reason,
                "steps": [
                    {
                        "step": "controller",
                        "decision": decision,
                        "reason": reason,
                        "retrieval_round": retrieval_round,
                        "verification_round": verification_round,
                    }
                ],
            }
        )
        return update

    return controller_node


def route_from_controller(state: dict) -> str:
    """This is what LangGraph calls to decide which node runs next, based
    on the decision the controller just made. Kept as a separate plain
    function (not a method) so it's trivially testable on its own, and
    because LangGraph's conditional-edge API expects exactly this shape:
    a function that takes state and returns a node-name string."""
    decision = state.get("decision", "ABSTAIN")
    return {
        "ACCEPT": "audit",
        "REVISE": "revise",
        "RETRIEVE_AGAIN": "retrieve",
        "ABSTAIN": "abstain",
    }[decision]