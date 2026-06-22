"""
Observability models — telemetry contracts for every AI operation.

TokenUsage maps directly to OpenAI billing.
TraceEvent is the unified record written by the observability middleware.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


class TokenUsage(BaseModel):
    """
    Token consumption for a single LLM call.

    Stored on RiskReport and TraceEvent.
    Maps directly to OpenAI usage response fields.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    prompt_tokens: int = Field(
        ...,
        description="Tokens consumed by the prompt / context",
        ge=0,
    )
    completion_tokens: int = Field(
        ...,
        description="Tokens generated in the completion",
        ge=0,
    )
    total_tokens: int = Field(
        ...,
        description="Sum of prompt and completion tokens",
        ge=0,
    )


class LatencyRecord(BaseModel):
    """
    Latency measurement for a single named operation.

    operation values: embed | vector_search | bm25_search | rerank | agent | ingest
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    operation: str = Field(
        ...,
        description="Name of the operation being measured",
    )
    latency_ms: float = Field(
        ...,
        description="Duration in milliseconds",
        ge=0.0,
    )
    recorded_at: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp when measurement was taken",
    )


class TraceEvent(BaseModel):
    """
    Unified telemetry record for one complete request lifecycle.

    Written by observability middleware after every API request.
    One TraceEvent per request — latency records per sub-operation
    are nested inside if needed.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    trace_id: str = Field(
        ...,
        description="OpenTelemetry distributed trace ID",
    )
    request_id: str = Field(
        ...,
        description="UUID for this specific HTTP request",
    )
    operation: str = Field(
        ...,
        description="High-level operation: ingest | search | analysis | timeline",
    )

    # ── Telemetry ─────────────────────────────────────────────────────────────
    token_usage: TokenUsage | None = Field(
        default=None,
        description="Token usage — present only for LLM operations",
    )
    latency: LatencyRecord = Field(
        ...,
        description="End-to-end latency for this request",
    )

    # ── Outcome ───────────────────────────────────────────────────────────────
    status: str = Field(
        ...,
        description="Request outcome: success | error",
    )
    error_detail: str | None = Field(
        default=None,
        description="Exception message if status is error",
    )