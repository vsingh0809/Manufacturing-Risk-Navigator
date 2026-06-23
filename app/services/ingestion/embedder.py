"""
Document embedder using LangChain OpenAIEmbeddings.

LangChain used here specifically for its async OpenAI wrapper.
Qdrant upsert is handled by the direct client — not LangChain.
"""

import logging

from langchain_openai import AzureOpenAIEmbeddings

from app.core.config import AppSettings
from app.core.exceptions import EmbeddingError
from app.models.document import ChunkedDocument

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """
    Generates dense vector embeddings for chunked documents.

    Uses LangChain OpenAIEmbeddings for its async batch support.
    Returns (ChunkedDocument, embedding) pairs — pipeline handles upsert.
    """

    def __init__(self, settings: AppSettings) -> None:
        # [WHY] LangChain embeddings initialised once — reuses HTTP
        # connection pool across all embed calls in a request.
        try:
            self._embeddings = AzureOpenAIEmbeddings(
    azure_deployment=settings.azure_openai_embedding_deployment,
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key,
    api_version=settings.azure_openai_api_version,
)
        except Exception as exc:
            raise EmbeddingError(
                message="Failed to initialise OpenAI embeddings client",
                detail=str(exc),
            ) from exc

    async def embed(
        self,
        chunks: list[ChunkedDocument],
    ) -> list[tuple[ChunkedDocument, list[float]]]:
        """
        Embed a list of chunks in a single async batch call.

        Args:
            chunks: ChunkedDocuments to embed.

        Returns:
            List of (ChunkedDocument, embedding_vector) pairs.

        Raises:
            EmbeddingError: If the OpenAI API call fails.
        """
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]

        try:
            # [WHY] aembed_documents sends one batched API request
            # not one request per chunk. Dramatically reduces latency
            # and API call overhead for large documents.
            vectors = await self._embeddings.aembed_documents(texts)
        except Exception as exc:
            raise EmbeddingError(
                message="OpenAI embedding API call failed",
                detail=str(exc),
            ) from exc

        if len(vectors) != len(chunks):
            raise EmbeddingError(
                message="Embedding count mismatch",
                detail=f"Expected {len(chunks)} vectors, got {len(vectors)}",
            )

        logger.info(
            "Embedding complete",
            extra={
                "chunks_embedded": len(chunks),
                "vector_dimension": len(vectors[0]) if vectors else 0,
            },
        )

        return list(zip(chunks, vectors))