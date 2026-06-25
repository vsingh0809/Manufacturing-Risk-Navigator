"""
LangGraph agent graph definition.

Compiles the StateGraph with parallel fan-out for
risk and dependency nodes.
"""

import logging
from functools import partial

from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.core.config import AppSettings
from app.services.agent.nodes import (
    dependency_node,
    retrieve_node,
    risk_node,
    summarise_node,
    timeline_node,
)
from app.services.agent.state import AgentState
from app.services.retrieval.vector_store import HybridRetriever

logger = logging.getLogger(__name__)


def build_graph(
    retriever: HybridRetriever,
    llm: AzureChatOpenAI,
    settings: AppSettings,
):
    """
    Build and compile the risk analysis LangGraph.

    Graph structure:
        START → retrieve → [risk ‖ dependency] → timeline → summarise → END

    Args:
        retriever: HybridRetriever instance (injected).
        llm:       AzureChatOpenAI instance (injected).
        settings:  AppSettings instance (injected).

    Returns:
        Compiled LangGraph runnable.
    """
    graph = StateGraph(AgentState)

    # [WHY] functools.partial binds injected dependencies to node
    # functions. LangGraph nodes must accept only (state) as argument.
    # partial() pre-fills the extra args without changing the signature.
    graph.add_node(
        "retrieve",
        partial(retrieve_node, retriever=retriever, settings=settings),
    )
    graph.add_node(
        "risk",
        partial(risk_node, llm=llm),
    )
    graph.add_node(
        "dependency",
        partial(dependency_node, llm=llm),
    )
    graph.add_node(
        "timeline",
        partial(timeline_node, llm=llm),
    )
    graph.add_node(
        "summarise",
        partial(summarise_node, llm=llm),
    )

    # ── Edges ──────────────────────────────────────────────────────────────
    graph.add_edge(START, "retrieve")

    # [WHY] Fan-out — retrieve feeds BOTH risk and dependency in parallel.
    # LangGraph executes nodes with no interdependency concurrently.
    graph.add_edge("retrieve", "risk")
    graph.add_edge("retrieve", "dependency")

    # [WHY] Fan-in — timeline waits for BOTH risk and dependency to complete.
    # LangGraph merges parallel state updates before proceeding.
    graph.add_edge("risk", "timeline")
    graph.add_edge("dependency", "timeline")

    graph.add_edge("timeline", "summarise")
    graph.add_edge("summarise", END)

    compiled = graph.compile()

    logger.info("Agent graph compiled successfully")
    return compiled