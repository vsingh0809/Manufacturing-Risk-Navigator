"""
LangGraph agent state.

TypedDict is required by LangGraph — not Pydantic.
Every node reads from and writes to this state.
"""

from typing import TypedDict

from app.models.analysis import DependencyEdge, RiskItem
from app.models.observability import TokenUsage
from app.models.search import SearchResult
from app.models.timeline import Milestone


class AgentState(TypedDict):
    """
    Shared state across all LangGraph nodes.

    [WHY] TypedDict not Pydantic — LangGraph serialises state
    internally and requires plain dict-compatible types.
    Pydantic models are stored as values inside the dict.
    """

    # ── Input ─────────────────────────────────────────────────────────────
    query: str
    project_name: str

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieved_chunks: list[SearchResult]
    context_truncated: bool          # True if token guard trimmed chunks

    # ── Parallel node outputs ─────────────────────────────────────────────
    risks: list[RiskItem]
    dependencies: list[DependencyEdge]

    # ── Sequential node outputs ───────────────────────────────────────────
    milestones: list[Milestone]
    summary: str

    # ── Observability ─────────────────────────────────────────────────────
    token_usage: TokenUsage

    # ── Error handling ────────────────────────────────────────────────────
    error: str | None