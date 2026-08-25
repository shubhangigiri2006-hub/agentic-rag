"""Writer Agent node.

Input: state["question"], state["evidence"] (everything accumulated so far,
across every retrieval round — not just the latest one).
Output: state["draft_answer"] — grounded in evidence, with inline citations.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = (
    "You are a careful research writer. Answer the user's question using "
    "ONLY the evidence provided below. Every factual sentence must end with "
    "the evidence_id(s) it draws from, in square brackets, e.g. [ev_1a2b3c]. "
    "If the evidence doesn't fully support an answer, say so explicitly "
    "rather than guessing or using outside knowledge."
)


MAX_SNIPPET_CHARS = 500  # keeps prompts within free-tier token budgets
MAX_EVIDENCE_ITEMS = 10  # keeps the prompt (and citation count) manageable


def _format_evidence_block(evidence):
    # Use only the most relevant N pieces — with duplicates now fixed at
    # the retrieval layer, this mainly protects against very broad queries
    # that legitimately pull in a lot of evidence across 2-3 rounds.
    top_evidence = sorted(evidence, key=lambda e: e["relevance"], reverse=True)[:MAX_EVIDENCE_ITEMS]
    blocks = []
    for e in top_evidence:
        text = e["text"]
        if len(text) > MAX_SNIPPET_CHARS:
            text = text[:MAX_SNIPPET_CHARS] + "..."
        blocks.append("[" + e["evidence_id"] + "] (" + e["source"] + ")\n" + text)
    return "\n\n".join(blocks)


def build_writer_node(llm):
    def writer_node(state: dict) -> dict:
        evidence = state.get("evidence", [])

        if not evidence:
            # No evidence at all — don't even call the LLM, since there's
            # nothing grounded it could possibly write.
            return {
                "draft_answer": "INSUFFICIENT_EVIDENCE: no evidence available to draft an answer.",
                "steps": [{"step": "writer", "skipped": True, "reason": "no evidence"}],
            }

        prompt = (
            f"Question: {state['question']}\n\n"
            f"Evidence:\n{_format_evidence_block(evidence)}\n\n"
            "Write a grounded answer. Cite evidence_ids inline."
        )
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )

        return {
            "draft_answer": response.content,
            "steps": [{"step": "writer", "evidence_count": len(evidence)}],
        }

    return writer_node