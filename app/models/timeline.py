"""
Timeline and dependency graph models.

MilestoneStatus is INFERRED by the agent from unstructured text —
no one labels documents as DELAYED; the agent reasons it from context.
"""

from datetime import UTC, date, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.analysis import DependencyEdge


def _utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


class Milestone(BaseModel):
    """
    A single project milestone with agent-inferred delivery status.

    blocking_risks links to RiskItem.risk_id — enables the UI to
    show exactly which risks are causing a milestone to be BLOCKED.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    milestone_id: str = Field(..., description="UUID for this milestone")
    project_name: str = Field(..., description="Owning project")
    name: str = Field(..., description="Milestone name e.g. 'Turbine Delivery'")

    planned_date: date | None = Field(
        default=None,
        description="Planned completion date from structured data (CSV/XLSX). None if not found.",
    )
    inferred_status: str = Field(
        ...,
        description="Agent-inferred status: ON_TRACK | AT_RISK | DELAYED | BLOCKED",
    )

    # ── Cross-Model Links ─────────────────────────────────────────────────────
    blocking_risks: list[str] = Field(
        default_factory=list,
        description="RiskItem.risk_id values causing this milestone to be blocked",
    )
    source_chunk_ids: list[str] = Field(
        ...,
        description="Chunk UUIDs that evidenced this milestone's status",
        min_length=1,
    )


class DependencyGraph(BaseModel):
    """
    Full project dependency graph combining milestones and edges.

    Edges are reused from models/analysis.py — same DependencyEdge
    schema serves both risk analysis and timeline views.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    project_name: str = Field(..., description="Project this graph covers")
    milestones: list[Milestone] = Field(
        default_factory=list,
        description="All milestones with inferred statuses",
    )
    edges: list[DependencyEdge] = Field(
        default_factory=list,
        description="Directed dependency edges between milestones and tasks",
    )
    generated_at: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp when this graph was generated",
    )