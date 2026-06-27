# ADD
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from app.models.analysis import DependencyEdge, RiskItem
from app.models.observability import TokenUsage
from app.models.search import SearchResult
from app.models.timeline import Milestone


def _merge_token_usage(existing: TokenUsage, new: TokenUsage) -> TokenUsage:
    """
    Reducer for parallel node token_usage updates.

    [WHY] risk_node and dependency_node both update token_usage
    simultaneously. Without a reducer LangGraph does not know
    how to merge two updates to the same key — it crashes.
    Reducer accumulates both nodes' token counts correctly.
    """
    return TokenUsage(
        prompt_tokens=existing.prompt_tokens + new.prompt_tokens,
        completion_tokens=existing.completion_tokens + new.completion_tokens,
        total_tokens=existing.total_tokens + new.total_tokens,
    )


def _merge_risks(existing: list, new: list) -> list:
    """Reducer for risk list — extend not overwrite."""
    return existing + new


def _merge_dependencies(existing: list, new: list) -> list:
    """Reducer for dependency list — extend not overwrite."""
    return existing + new


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────
    query: str
    project_name: str

    # ── Retrieval ──────────────────────────────────────────────
    retrieved_chunks: list[SearchResult]
    context_truncated: bool

    # ── Parallel node outputs — need reducers ──────────────────
    # [WHY] Annotated[type, reducer] tells LangGraph how to merge
    # updates from parallel nodes into one state value.
    risks: Annotated[list[RiskItem], _merge_risks]
    dependencies: Annotated[list[DependencyEdge], _merge_dependencies]
    token_usage: Annotated[TokenUsage, _merge_token_usage]

    # ── Sequential node outputs — no reducer needed ────────────
    milestones: list[Milestone]
    summary: str

    # ── Error ──────────────────────────────────────────────────
    error: str | None