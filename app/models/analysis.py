"""
Analysis output models — risk identification and dependency mapping.

Every model carries source_chunk_ids as the citation chain.
This is what separates grounded AI output from hallucination.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.observability import TokenUsage


def _utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


class RiskItem(BaseModel):
    """
    A single identified risk extracted by the analysis agent.

    source_chunk_ids links every risk claim back to the exact
    chunks that evidenced it — mandatory for evaluator traceability.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    risk_id: str = Field(..., description="UUID for this risk item")
    category: str = Field(
        ...,
        description=(
            "Risk type: DELIVERY_DELAY | SUPPLIER_RISK | "
            "DEPENDENCY_BLOCKER | QUALITY_ISSUE | APPROVAL_PENDING | UNKNOWN"
        ),
    )
    severity: str = Field(
        ...,
        description="Risk severity: LOW | MEDIUM | HIGH | CRITICAL",
    )
    description: str = Field(
        ...,
        description="Agent-generated natural language explanation of the risk",
        min_length=10,
    )

    # ── Affected Entities ─────────────────────────────────────────────────────
    affected_project: str = Field(
        ...,
        description="Project name this risk belongs to",
    )
    affected_milestone: str | None = Field(
        default=None,
        description="Milestone affected by this risk",
    )
    supplier: str | None = Field(
        default=None,
        description="Supplier involved in this risk if applicable",
    )

    # ── Citation Chain ────────────────────────────────────────────────────────
    source_chunk_ids: list[str] = Field(
        ...,
        description="Chunk UUIDs that evidenced this risk — the grounding proof",
        min_length=1,
    )

    detected_at: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp when agent detected this risk",
    )


class DependencyEdge(BaseModel):
    """
    A directed dependency relationship between two tasks or milestones.

    Extracted by the agent from unstructured project communications.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    from_task: str = Field(
        ...,
        description="Task or milestone that is the dependency source",
    )
    to_task: str = Field(
        ...,
        description="Task or milestone that depends on from_task",
    )
    dependency_type: str = Field(
        ...,
        description="Relationship type: blocks | depends_on | triggers",
    )
    source_chunk_ids: list[str] = Field(
        ...,
        description="Chunk UUIDs that evidenced this dependency",
        min_length=1,
    )


class RiskReport(BaseModel):
    """
    Full agent response to a single risk analysis query.

    Contains all identified risks, dependency edges, a narrative
    summary, and full token usage for cost observability.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    report_id: str = Field(..., description="UUID for this report")
    project_name: str = Field(..., description="Project analysed")
    query: str = Field(..., description="Original user query that triggered this report")

    # ── Agent Outputs ─────────────────────────────────────────────────────────
    risks: list[RiskItem] = Field(
        default_factory=list,
        description="All risk items identified by the agent",
    )
    dependencies: list[DependencyEdge] = Field(
        default_factory=list,
        description="All dependency relationships extracted by the agent",
    )
    summary: str = Field(
        ...,
        description="Agent-generated narrative summary of the risk landscape",
        min_length=10,
    )

    # ── Observability ─────────────────────────────────────────────────────────
    token_usage: TokenUsage = Field(
        ...,
        description="Token consumption for this agent run",
    )
    generated_at: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp when this report was generated",
    )