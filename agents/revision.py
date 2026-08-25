"""Revision Agent node.

Uses the verifier's per-claim feedback to fix only the problematic claims.
Explicitly instructed to preserve SUPPORTED claims verbatim — otherwise a
full rewrite could accidentally break something that was already correct.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = (
    "You are a revision editor. You will get a draft answer, per-claim "
    "verification feedback, and the available evidence. Rewrite the answer "
    "so that: (1) claims marked SUPPORTED are preserved as-is, (2) claims "
    "marked UNSUPPORTED, CONTRADICTED, or PARTIALLY_SUPPORTED are fixed or "
    "removed based on the evidence, (3) every remaining factual sentence "
    "still cites its evidence_id(s) in square brackets. If evidence to fix "
    "a claim isn't available, state that plainly rather than guessing. "
    "Output ONLY the revised answer text."
)

MAX_SNIPPET_CHARS = 500


def _format_feedback(claims):
    lines = []
    for c in claims:
        lines.append("- [" + c["support_status"] + "] " + c["text"])
    return "\n".join(lines)


def _format_evidence_block(evidence):
    blocks = []
    for e in evidence:
        text = e["text"]
        if len(text) > MAX_SNIPPET_CHARS:
            text = text[:MAX_SNIPPET_CHARS] + "..."
        block = "[" + e["evidence_id"] + "] (" + e["source"] + ")\n" + text
        blocks.append(block)
    return "\n\n".join(blocks)


def build_revision_node(llm):
    def revision_node(state):
        draft = state.get("draft_answer", "")
        claims = state.get("claims", [])
        evidence = state.get("evidence", [])

        feedback_block = _format_feedback(claims)
        evidence_block = _format_evidence_block(evidence)

        prompt_parts = []
        prompt_parts.append("Question: " + state["question"])
        prompt_parts.append("Draft answer:")
        prompt_parts.append(draft)
        prompt_parts.append("Per-claim verification:")
        prompt_parts.append(feedback_block)
        prompt_parts.append("Available evidence:")
        prompt_parts.append(evidence_block)
        prompt = "\n\n".join(prompt_parts)

        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        return {
            "draft_answer": response.content,
            "revision_notes": "revised based on verifier feedback",
            "steps": [{"step": "revision", "num_claims_reviewed": len(claims)}],
        }

    return revision_node