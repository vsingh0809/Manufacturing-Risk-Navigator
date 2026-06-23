"""
Token-aware document chunker.

Splits RawDocument content into overlapping chunks using tiktoken
for accurate token counting. Each chunk becomes one Qdrant point.
"""

import logging
import uuid

import tiktoken

from app.core.config import AppSettings
from app.core.exceptions import ChunkingError
from app.models.document import ChunkedDocument, RawDocument

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Splits documents into token-bounded overlapping chunks.

    Uses tiktoken to count tokens accurately — character-based
    splitting gives inconsistent chunk sizes across languages.
    """

    def __init__(self, settings: AppSettings) -> None:
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap
        # [WHY] cl100k_base is the tokeniser for text-embedding-3-small.
        # Wrong tokeniser = inaccurate token counts = chunks that exceed
        # model context window silently.
        try:
            self._tokeniser = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            raise ChunkingError(
                message="Failed to load tiktoken tokeniser",
                detail=str(exc),
            ) from exc

    async def chunk(self, document: RawDocument) -> list[ChunkedDocument]:
        """
        Split one RawDocument into overlapping ChunkedDocuments.

        Args:
            document: RawDocument from cleaner.

        Returns:
            List of ChunkedDocument — at least one guaranteed.

        Raises:
            ChunkingError: If tokenisation or chunking fails.
        """
        if not document.content or not document.content.strip():
            raise ChunkingError(
                message="Document content cannot be empty or whitespace only",
                detail=f"document_id={document.metadata.document_id}",
            )
        try:
            tokens = self._tokeniser.encode(document.content)
        except Exception as exc:
            raise ChunkingError(
                message="Tokenisation failed",
                detail=str(exc),
            ) from exc

        if not tokens:
            raise ChunkingError(
                message="Document produced zero tokens after encoding",
                detail=f"document_id={document.metadata.document_id}",
            )

        chunks = self._sliding_window(tokens)
        chunked_documents: list[ChunkedDocument] = []

        for index, chunk_tokens in enumerate(chunks):
            try:
                content = self._tokeniser.decode(chunk_tokens)
            except Exception as exc:
                raise ChunkingError(
                    message=f"Failed to decode chunk {index}",
                    detail=str(exc),
                ) from exc

            chunked_documents.append(
                ChunkedDocument(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document.metadata.document_id,
                    source_chunk_id=None,  # Sprint 2 — hierarchical chunking
                    metadata=document.metadata,
                    content=content,
                    chunk_index=index,
                    token_count=len(chunk_tokens),
                )
            )

        logger.info(
            "Chunking complete",
            extra={
                "document_id": document.metadata.document_id,
                "total_tokens": len(tokens),
                "chunks_produced": len(chunked_documents),
            },
        )

        return chunked_documents

    def _sliding_window(self, tokens: list[int]) -> list[list[int]]:
        """
        Split token list into overlapping windows.

        Args:
            tokens: Full token list for a document.

        Returns:
            List of token sublists, each of length <= chunk_size.
        """
        # [WHY] Sliding window with overlap preserves context at chunk
        # boundaries. Without overlap, a sentence split across two chunks
        # loses meaning in both halves.
        windows: list[list[int]] = []
        start = 0
        total = len(tokens)

        while start < total:
            end = min(start + self._chunk_size, total)
            windows.append(tokens[start:end])

            if end == total:
                break

            # [WHY] Step back by overlap so next chunk shares context
            # with the tail of the current chunk.
            start += self._chunk_size - self._chunk_overlap

        return windows