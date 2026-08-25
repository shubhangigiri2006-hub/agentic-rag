"""Claim Extraction node.

Input: state["draft_answer"] — the writer's prose, with [ev_xxx] citations.
Output: state["claims"] — a list of atomic claims, each with its own
evidence_ids, ready for the verifier to check one at a time.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from schemas import Claim, new_id

SYSTEM_PROMPT = (
    "You extract atomic factual claims from a draft answer that cites "
    "evidence_ids in square brackets like [ev_1a2b3c]. For each claim, "
    "output its text and the list of evidence_ids cited for it. Respond "
    "with ONLY a JSON array of objects: "
    '[{"text": "...", "evidence_ids": ["ev_..."]}]. No other text.'
)


def _extract_json_array(text: str) -> list:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def _fallback_split(draft_answer: str) -> list[Claim]:
    """If the LLM's JSON extraction fails to parse, fall back to a cruder
    method: split on sentence boundaries, regex out the evidence_ids from
    each sentence. Worse quality, but keeps the pipeline moving instead of
    crashing on a single bad LLM response."""
    sentences = re.split(r"(?<=[.!?])\s+", draft_answer.strip())
    claims = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        ids = re.findall(r"\[(ev_[a-f0-9]+)", s)
        claims.append(Claim(claim_id=new_id("claim"), text=s, evidence_ids=ids))
    return claims


def build_claim_extractor_node(llm):
    def claim_extractor_node(state: dict) -> dict:
        draft = state.get("draft_answer", "")
        if not draft or draft.startswith("INSUFFICIENT_EVIDENCE"):
            return {"claims": [], "steps": [{"step": "claim_extractor", "skipped": True}]}

        prompt = f"Draft answer:\n{draft}"
        try:
            response = llm.invoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            parsed = _extract_json_array(response.content)
            claims = [
                Claim(
                    claim_id=new_id("claim"),
                    text=str(c["text"]),
                    evidence_ids=list(c.get("evidence_ids", [])),
                )
                for c in parsed
            ]
            if not claims:
                raise ValueError("empty claim list")
        except Exception:
            claims = _fallback_split(draft)

        return {
            "claims": [c.to_dict() for c in claims],
            "steps": [{"step": "claim_extractor", "num_claims": len(claims)}],
        }

    return claim_extractor_node