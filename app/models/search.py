"""
Search query and result models.

Defines what enters and exits the hybrid retrieval pipeline.
All three retrieval scores are preserved for observability and RAGAS evaluation.
"""

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import DocumentMetadata


class SearchQuery(BaseModel):
    """
    Incoming search request from the API layer.

    Optional filters are pushed to Qdrant payload filters —
    they narrow the vector search space before scoring.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    query: str = Field(
        ...,
        description="Natural language search query",
        min_length=3,
    )
    project_name: str | None = Field(
        default=None,
        description="Filter results to a specific project",
    )
    department: str | None = Field(
        default=None,
        description="Filter results to a specific department",
    )
    supplier: str | None = Field(
        default=None,
        description="Filter results related to a specific supplier",
    )
    top_k: int = Field(
        default=10,
        description="Number of final results to return after reranking",
        ge=1,
        le=50,
    )
    rerank: bool = Field(
        default=True,
        description="Whether to apply cross-encoder reranking (disable for speed)",
    )


class SearchResult(BaseModel):
    """
    A single retrieval result with all scoring signals preserved.

    All three scores are retained for:
    - RAGAS evaluation (measure each path independently)
    - Observability (which path contributed most)
    - Future weight tuning (without schema changes)
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    chunk_id: str = Field(..., description="UUID of the matched chunk")
    document_id: str = Field(..., description="UUID of the parent document")
    content: str = Field(..., description="Chunk text content")
    metadata: DocumentMetadata = Field(
        ...,
        description="Full document metadata — enables source display in UI",
    )

    # ── Retrieval Scores ──────────────────────────────────────────────────────
    vector_score: float = Field(
        ...,
        description="Qdrant cosine similarity score from dense vector search",
        ge=0.0,
        le=1.0,
    )
    bm25_score: float = Field(
        ...,
        description="BM25 keyword relevance score",
        ge=0.0,
    )
    rerank_score: float | None = Field(
        default=None,
        description="ms-marco cross-encoder score. None when rerank=False",
    )
    final_score: float = Field(
        ...,
        description="RRF fused score used for final ranking",
        ge=0.0,
    )


class HybridResult(BaseModel):
    """
    Complete response from the hybrid retrieval pipeline.

    Wraps results with pipeline-level metadata for observability.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    query: str = Field(..., description="Echo of the original query")
    results: list[SearchResult] = Field(
        default_factory=list,
        description="Ranked list of retrieval results",
    )
    total_retrieved: int = Field(
        ...,
        description="Total candidates pulled before reranking",
        ge=0,
    )
    latency_ms: float = Field(
        ...,
        description="End-to-end retrieval pipeline latency in milliseconds",
        ge=0.0,
    )