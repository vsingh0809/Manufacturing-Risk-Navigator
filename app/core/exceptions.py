"""
Domain exception hierarchy for Manufacturing Risk Navigator.

All application exceptions inherit from MRNBaseError, enabling
callers to catch at the right level of specificity.
"""


class MRNBaseError(Exception):
    """Base for all application-level exceptions."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(message)


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestionError(MRNBaseError):
    """Raised when the ingestion pipeline fails at any stage."""


class ExtractionError(IngestionError):
    """Raised when a file extractor cannot parse or read a document."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when no extractor is registered for the given MIME type."""

    def __init__(self, mime_type: str) -> None:
        super().__init__(
            message=f"No extractor registered for MIME type: {mime_type}",
            detail=mime_type,
        )


class ChunkingError(IngestionError):
    """Raised when chunking fails for a document."""


class EmbeddingError(IngestionError):
    """Raised when the embedding API call fails."""


# ── Retrieval ─────────────────────────────────────────────────────────────────

class RetrievalError(MRNBaseError):
    """Raised when any retrieval path (vector, BM25, hybrid) fails."""


class VectorStoreError(RetrievalError):
    """Raised on Qdrant client errors (connection, upsert, query)."""


class RerankerError(RetrievalError):
    """Raised when the cross-encoder reranker fails."""


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentError(MRNBaseError):
    """Raised when the analysis agent workflow fails."""


class ContextWindowError(AgentError):
    """Raised when retrieved context exceeds the model's token limit."""


# ── Configuration ─────────────────────────────────────────────────────────────

class ConfigurationError(MRNBaseError):
    """Raised when a required configuration value is missing or invalid."""


# ── Storage ───────────────────────────────────────────────────────────────────

class StorageError(MRNBaseError):
    """Raised on database or vector store connectivity failures."""