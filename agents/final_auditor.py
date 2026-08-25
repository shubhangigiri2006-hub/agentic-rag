"""Final Auditor node + the ABSTAIN terminal node.

The auditor is an independent last check: even if the controller already
decided ACCEPT, this runs a fresh PASS/FAIL judgment with its own framing,
rather than just trusting the controller's earlier decision.

abstain_node is what runs when the controller decides ABSTAIN — it isn't
an LLM call, deliberately: the whole point of abstaining is to STOP, not
give the model one more chance to sound confident about weak evidence.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents.json_utils import extract_json

SYSTEM_PROMPT = (
    "You are the final auditor for a research answer. Review the answer, "
    "its claims and their support statuses, and check: factual support, "
    "citation correctness, contradictions, and unresolved claims. Respond "
    "with ONLY JSON: "
    '{"status": "PASS" or "FAIL", "confidence": 0.0-1.0, "notes": "..."}'
)


def build_final_auditor_node(llm):
    def final_auditor_node(state):
        draft = state.get("draft_answer", "")
        claims = state.get("claims", [])

        claims_block = "\n".join(
            "- [" + c["support_status"] + "] " + c["text"] for c in claims
        )
        prompt = (
            "Question: " + state["question"] + "\n\n"
            "Answer:\n" + draft + "\n\n"
            "Claims:\n" + claims_block
        )

        try:
            response = llm.invoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            audit = extract_json(response.content)
            status = audit.get("status", "FAIL")
            confidence = float(audit.get("confidence", 0.0))
        except Exception as e:
            audit = {
                "status": "FAIL",
                "confidence": 0.0,
                "notes": "auditor output unparsable",
                "error": str(e),
            }
            status = "FAIL"
            confidence = 0.0

        return {
            "audit": audit,
            "final_answer": draft,
            "final_status": status,
            "steps": [{"step": "final_auditor", "status": status, "confidence": confidence}],
        }

    return final_auditor_node


def abstain_node(state):
    claims = state.get("claims", [])
    supported = sum(1 for c in claims if c.get("support_status") == "SUPPORTED")
    total = len(claims)
    message = (
        "I don't have sufficient verified evidence to answer this confidently "
        "(" + str(supported) + "/" + str(total) + " claims were fully supported after "
        + str(state.get("retrieval_round", 0)) + " retrieval round(s)). "
        "Rather than guess, I'm abstaining."
    )
    return {
        "final_answer": message,
        "final_status": "ABSTAINED",
        "steps": [{"step": "abstain", "supported": supported, "total": total}],
    }