"""
Document lifecycle models.

Covers three stages:
  1. DocumentMetadata — domain context attached to every document and chunk
  2. RawDocument      — extracted content before chunking (never stored in Qdrant)
  3. ChunkedDocument  — chunked content that gets embedded and upserted to Qdrant
"""

from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.exceptions import ConfigurationError


def _utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


class DocumentMetadata(BaseModel):
    """
    Domain context for a document or chunk.

    These fields are stored as Qdrant payload and used for
    filtered retrieval (project_name, department, supplier, risk_category).
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    # ── Identity ──────────────────────────────────────────────────────────────
    document_id: str = Field(
        ...,
        description="UUID identifying the parent document",
    )
    source_id: str = Field(
        ...,
        description="UUID identifying the upload session or batch",
    )
    source_name: str = Field(
        ...,
        description="Human-readable source name (e.g. 'Project Atlas Q1 Reports')",
    )
    source_type: str = Field(
        ...,
        description="File type: pdf | spreadsheet | email | transcript | status_report | rfq | text|txt",
    )

    # ── File ──────────────────────────────────────────────────────────────────
    file_name: str = Field(
        ...,
        description="Original filename with extension",
    )
    file_path: Path = Field(
        ...,
        description="Absolute path to the source file",
    )
    page_number: int | None = Field(
        default=None,
        description="Page number within the source document (1-indexed)",
    )
    total_pages: int | None = Field(
        default=None,
        description="Total page count of the source document",
    )
    content_hash: str = Field(
        ...,
        description="SHA-256 hash of raw content — used by deduplicator (next sprint)",
    )

    # ── Manufacturing Domain ───────────────────────────────────────────────────
    project_name: str = Field(
        ...,
        description="Project this document belongs to (e.g. 'Project Atlas')",
    )
    department: str = Field(
        ...,
        description="Originating department (e.g. 'procurement', 'engineering', 'QA')",
    )
    supplier: str | None = Field(
        default=None,
        description="Supplier name if document is supplier-related",
    )
    milestone: str | None = Field(
        default=None,
        description="Project milestone this document references",
    )
    risk_category: str = Field(
        default="UNKNOWN",
        description=(
            "Risk classification: DELIVERY_DELAY | SUPPLIER_RISK | "
            "DEPENDENCY_BLOCKER | QUALITY_ISSUE | APPROVAL_PENDING | UNKNOWN"
        ),
    )

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="When the source document was authored (from file metadata)",
    )
    ingested_at: datetime = Field(
        default_factory=_utc_now,
        description="When this document was processed by our ingestion pipeline",
    )

    # ── Flexible Overflow ─────────────────────────────────────────────────────
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Arbitrary extra fields that do not fit structured schema",
    )

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, value: str) -> str:
        """Enforce known source types to keep Qdrant payload filters reliable."""
        valid = {
            "pdf",
            "spreadsheet",
            "email",
            "transcript",
            "status_report",
            "rfq",
            "text",
            "txt",
        }
        lower = value.lower()
        if lower not in valid:
            raise ConfigurationError(
                message=f"Invalid source_type: {value}",
                detail=f"Must be one of: {', '.join(sorted(valid))}",
            )
        return lower

    @field_validator("risk_category")
    @classmethod
    def validate_risk_category(cls, value: str) -> str:
        """Enforce known risk categories."""
        valid = {
            "DELIVERY_DELAY",
            "SUPPLIER_RISK",
            "DEPENDENCY_BLOCKER",
            "QUALITY_ISSUE",
            "APPROVAL_PENDING",
            "UNKNOWN",
        }
        upper = value.upper()
        if upper not in valid:
            raise ConfigurationError(
                message=f"Invalid risk_category: {value}",
                detail=f"Must be one of: {', '.join(sorted(valid))}",
            )
        return upper

    @field_validator("page_number")
    @classmethod
    def validate_page_number(cls, value: int | None) -> int | None:
        """Page numbers must be positive if provided."""
        if value is not None and value < 1:
            raise ValueError(f"page_number must be >= 1, got {value}")
        return value


class RawDocument(BaseModel):
    """
    Extracted document content before any chunking.

    This model is NEVER stored in Qdrant.
    It is the intermediate state between extraction and chunking.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    metadata: DocumentMetadata
    content: str = Field(
        ...,
        description="Full raw text extracted from the source file",
        min_length=1,
    )


class ChunkedDocument(BaseModel):
    """
    A single chunk ready for embedding and Qdrant upsert.

    Each chunk carries full metadata so retrieval results are
    self-contained — no secondary lookup needed to identify source.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    # ── Chunk Identity ────────────────────────────────────────────────────────
    chunk_id: str = Field(
        ...,
        description="UUID uniquely identifying this chunk",
    )
    document_id: str = Field(
        ...,
        description="UUID of the parent RawDocument",
    )
    source_chunk_id: str | None = Field(
        default=None,
        description=(
            "UUID of parent chunk — supports hierarchical/parent-child chunking. "
            "None for MVP flat chunking. Plug in next sprint without schema changes."
        ),
    )

    # ── Content ───────────────────────────────────────────────────────────────
    metadata: DocumentMetadata = Field(
        ...,
        description="Full domain metadata inherited from parent document",
    )
    content: str = Field(
        ...,
        description="Chunk text that will be embedded",
        min_length=1,
    )

    # ── Chunk Position ────────────────────────────────────────────────────────
    chunk_index: int = Field(
        ...,
        description="Zero-based position of this chunk within its parent document",
        ge=0,
    )
    token_count: int = Field(
        ...,
        description="Estimated token count — used for agent context window management",
        gt=0,
    )