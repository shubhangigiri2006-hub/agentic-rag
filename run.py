"""CLI entrypoint.

    python run.py "Who invented the transistor and when?"
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid

from config import DEFAULT_CONFIG
from display import format_for_display


def save_run(final_state: dict, question: str, logs_dir: str = "experiments/logs") -> str:
    os.makedirs(logs_dir, exist_ok=True)
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    record = {"run_id": run_id, "question": question, "timestamp": time.time(), **final_state}
    path = os.path.join(logs_dir, f"{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the adaptive agentic RAG graph.")
    parser.add_argument("question", type=str)
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    problems = config.validate_for_run()
    if problems:
        print("Missing configuration:")
        for p in problems:
            print(f"  - {p}")
        return

    from llm_provider import get_llm
    from tools.search_factory import get_search_fn
    from graph import build_graph

    llm = get_llm(config)
    search_fn = get_search_fn(config)
    app = build_graph(llm, search_fn, config)

    print(f"Running: {args.question}\n")
    final_state = app.invoke({"question": args.question, "mode": "C"})

    print(f"=== Answer (status: {final_state.get('final_status')}) ===")
    print(format_for_display(final_state.get("final_answer", ""), final_state.get("evidence", [])))

    print(f"\n=== Evidence used ({len(final_state.get('evidence', []))}) ===")
    for e in final_state.get("evidence", []):
        print(f"  [{e['evidence_id']}] {e['source']} (round {e['retrieval_round']})")

    path = save_run(final_state, args.question, config.logs_dir if hasattr(config, "logs_dir") else "experiments/logs")
    print(f"\nFull trace saved to: {path}")


if __name__ == "__main__":
    main()