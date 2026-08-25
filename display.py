"""Formats the final answer for human reading — converts raw [ev_xxxxx]
citations into numbered footnotes with a Sources list, so the reader never
sees internal evidence_ids, only clean [1], [2] style references.

This is intentionally a separate step from the writer/verifier pipeline:
those need the raw evidence_ids to check citations against evidence. This
function only runs once, right before printing/returning the final answer.
"""

from __future__ import annotations

import re


def format_for_display(answer: str, evidence: list[dict]) -> str:
    evidence_by_id = {e["evidence_id"]: e for e in evidence}

    # Map each evidence_id to a footnote number, in the order it first
    # appears in the text — not the order it was retrieved in, so footnote
    # [1] is always the first thing the reader actually encounters.
    footnote_number = {}

    def assign_numbers(match: re.Match) -> str:
        ids = [i.strip() for i in match.group(1).split(",")]
        numbers = []
        for eid in ids:
            if eid not in footnote_number:
                footnote_number[eid] = len(footnote_number) + 1
            numbers.append(str(footnote_number[eid]))
        return "[" + ",".join(numbers) + "]"

    # Matches [ev_xxxxx] or [ev_xxxxx, ev_yyyyy, ...] and replaces with
    # footnote numbers via assign_numbers above.
    clean_text = re.sub(r"\[((?:ev_[a-f0-9]+,?\s*)+)\]", assign_numbers, answer)

    if not footnote_number:
        return clean_text  # no citations found — nothing to append

    sources_lines = ["", "Sources:"]
    for eid, num in sorted(footnote_number.items(), key=lambda kv: kv[1]):
        e = evidence_by_id.get(eid)
        if e:
            sources_lines.append(f"  [{num}] {e['title']} — {e['source']}")
        else:
            sources_lines.append(f"  [{num}] (evidence_id {eid} not found in evidence list)")

    return clean_text + "\n" + "\n".join(sources_lines)