"""Builds the adaptive agentic RAG graph.

    START -> plan -> retrieve -> write -> extract_claims -> verify -> controller
                        ^                                                |
                        |-------------------- RETRIEVE_AGAIN ------------|
                                                                          |
              revise -> extract_claims -> verify <---- REVISE -----------|
                                                                          |
                                                    ACCEPT -> audit -> END
                                                    ABSTAIN -> abstain -> END

The loop is bounded by config.max_retrieval_rounds / max_verification_rounds,
enforced inside controller.py — so this graph can never run forever no
matter what the verifier says.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.claim_extractor import build_claim_extractor_node
from agents.controller import build_controller_node, route_from_controller
from agents.final_auditor import abstain_node, build_final_auditor_node
from agents.query_planner import build_query_planner_node
from agents.retrieval import build_retrieval_node
from agents.revision import build_revision_node
from agents.verifier import build_verifier_node
from agents.writer import build_writer_node
from config import Config
from state import AgentState


def build_graph(llm, search_fn, config: Config):
    graph = StateGraph(AgentState)

    # Step 1: register every node with a name. build_X_node(llm) returns the
    # actual function; add_node associates it with a string name we'll use
    # to wire edges below.
    graph.add_node("plan", build_query_planner_node(llm))
    graph.add_node("retrieve", build_retrieval_node(search_fn, top_k=config.search_top_k))
    graph.add_node("write", build_writer_node(llm))
    graph.add_node("extract_claims", build_claim_extractor_node(llm))
    graph.add_node("verify", build_verifier_node(llm))
    graph.add_node(
        "controller",
        build_controller_node(config.max_retrieval_rounds, config.max_verification_rounds),
    )
    graph.add_node("revise", build_revision_node(llm))
    graph.add_node("audit", build_final_auditor_node(llm))
    graph.add_node("abstain", abstain_node)

    # Step 2: the straight-line path every run starts with.
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "write")
    graph.add_edge("write", "extract_claims")
    graph.add_edge("extract_claims", "verify")
    graph.add_edge("verify", "controller")

    # Step 3: the branch point. route_from_controller (from controller.py)
    # looks at state["decision"] and returns one of these four key names;
    # LangGraph sends execution to the matching node.
    graph.add_conditional_edges(
        "controller",
        route_from_controller,
        {
            "retrieve": "retrieve",   # RETRIEVE_AGAIN loops back
            "revise": "revise",       # REVISE goes to the revision node
            "audit": "audit",         # ACCEPT goes to the final auditor
            "abstain": "abstain",     # ABSTAIN goes to the terminal node
        },
    )

    # Step 4: after revision, re-extract claims from the NEW draft and
    # verify again — this is the REVISE loop closing.
    graph.add_edge("revise", "extract_claims")

    # Step 5: both terminal paths end the graph.
    graph.add_edge("audit", END)
    graph.add_edge("abstain", END)

    # compile() turns the graph definition into something callable —
    # app.invoke(initial_state) runs it start to finish.
    return graph.compile()