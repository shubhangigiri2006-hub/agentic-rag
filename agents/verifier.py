"""Verification Agent node.

Input: state["claims"], state["evidence"]
Output: state["verification"] — per-claim support status, plus flags for
whether more retrieval or revision is needed.

Also updates state["claims"] in place, attaching each claim's status.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents.json_utils import extract_json

SYSTEM_PROMPT = (
    "You are a strict fact-checking verifier. For each claim below, check it "
    "against the cited evidence (and note if any OTHER evidence contradicts "
    "it). Classify each claim's support_status as exactly one of: "
    "SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, CONTRADICTED, "
    "INSUFFICIENT_EVIDENCE. A reviewer cannot recover information that was "
    "never retrieved — if the evidence simply doesn't cover the claim, that "
    "is INSUFFICIENT_EVIDENCE, not the writer's fault. For any claim that is "
    "not SUPPORTED, suggest a short, targeted search query that could find "
    "the missing evidence. Respond with ONLY a JSON array: "
    '[{"claim_id": "...", "support_status": "...", '
    '"missing_query": "..." (omit or null if not applicable)}]'
)


def _format_claims_with_evidence(claims, evidence_by_id):
    blocks = []
    for c in claims:
        cited_lines = []
        for eid in c.get("evidence_ids", []):
            if eid in evidence_by_id:
                cited_lines.append("  [" + eid + "] " + evidence_by_id[eid]["text"])
        cited = "\n".join(cited_lines) if cited_lines else "  (no evidence cited)"
        block = "claim_id: " + c["claim_id"] + "\ntext: " + c["text"] + "\ncited evidence:\n" + cited
        blocks.append(block)
    return "\n\n".join(blocks)


def build_verifier_node(llm):
    def verifier_node(state):
        claims = state.get("claims", [])
        evidence = state.get("evidence", [])
        evidence_by_id = {e["evidence_id"]: e for e in evidence}
        round_num = state.get("verification_round", 0)

        if not claims:
            verification = {
                "claim_statuses": [],
                "needs_retrieval": True,
                "needs_revision": False,
                "missing_queries": [state["question"]],
            }
            return {
                "verification": verification,
                "verification_round": round_num + 1,
                "steps": [{"step": "verifier", "skipped": True, "reason": "no claims"}],
            }

        prompt = (
            "Question: " + state["question"] + "\n\n"
            "Claims:\n" + _format_claims_with_evidence(claims, evidence_by_id)
        )

        error_info = None
        try:
            response = llm.invoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            statuses = extract_json(response.content)
            if not isinstance(statuses, list):
                raise ValueError("verifier did not return a list")
        except Exception as e:
            error_info = str(e)
            statuses = [
                {"claim_id": c["claim_id"], "support_status": "INSUFFICIENT_EVIDENCE", "missing_query": None}
                for c in claims
            ]

        status_by_id = {}
        for s in statuses:
            if "claim_id" in s:
                status_by_id[s["claim_id"]] = s

        updated_claims = []
        for c in claims:
            s = status_by_id.get(c["claim_id"])
            c = dict(c)
            c["support_status"] = s["support_status"] if s else "INSUFFICIENT_EVIDENCE"
            updated_claims.append(c)

        bad_statuses = {"UNSUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}
        missing_queries = []
        for s in statuses:
            if s.get("support_status") in bad_statuses and s.get("missing_query"):
                missing_queries.append(s["missing_query"])

        needs_retrieval = len(missing_queries) > 0
        revise_statuses = bad_statuses | {"PARTIALLY_SUPPORTED"}
        needs_revision = any(s.get("support_status") in revise_statuses for s in statuses)

        verification = {
            "claim_statuses": statuses,
            "needs_retrieval": needs_retrieval,
            "needs_revision": needs_revision,
            "missing_queries": missing_queries,
        }
        if error_info:
            verification["error"] = error_info

        return {
            "claims": updated_claims,
            "verification": verification,
            "verification_round": round_num + 1,
            "steps": [
                {
                    "step": "verifier",
                    "round": round_num,
                    "needs_retrieval": needs_retrieval,
                    "needs_revision": needs_revision,
                    "error": error_info,
                }
            ],
        }

    return verifier_node