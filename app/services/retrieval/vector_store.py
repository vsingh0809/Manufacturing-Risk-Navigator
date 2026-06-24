"""
Hybrid retriever using Qdrant native hybrid search.

Single query combines:
  - Dense vectors  (OpenAI text-embedding-3-small via Azure)
  - Sparse vectors (FastEmbed SPLADE model)
  - RRF fusion     (built into Qdrant)

Replaces: vector_store.py + bm25.py + custom RRF implementation.
"""

import logging

from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from app.core.config import AppSettings
from app.core.exceptions import VectorStoreError
from app.models.document import DocumentMetadata
from app.models.search import SearchQuery, SearchResult

logger = logging.getLogger(__name__)

# [WHY] SPLADE is the standard sparse model for hybrid search.
# It learns term importance rather than raw TF-IDF/BM25 weights.
# Qdrant + FastEmbed ship this as the recommended sparse model.
_SPARSE_MODEL = "prithivida/Splade_PP_en_v1"


class HybridRetriever:
    """
    Executes dense + sparse hybrid search natively in Qdrant.

    One class. One query. No custom fusion code.
    Reranker applied after Qdrant returns fused candidates.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        embedder,          # DocumentEmbedder — for dense query vector
        reranker,          # MsMarcoReranker  — for cross-encoder reranking
        settings: AppSettings,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embedder = embedder
        self._reranker = reranker
        self._settings = settings

        # [WHY] SparseTextEmbedding loaded once at init.
        # Model download happens on first instantiation (~200MB).
        # Loading per-request would be catastrophically slow.
        try:
            self._sparse_model = SparseTextEmbedding(
                model_name=_SPARSE_MODEL
            )
            logger.info(
                "Sparse embedding model loaded",
                extra={"model": _SPARSE_MODEL},
            )
        except Exception as exc:
            raise VectorStoreError(
                message="Failed to load sparse embedding model",
                detail=str(exc),
            ) from exc

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """
        Execute hybrid search and return reranked results.

        Args:
            query: SearchQuery with text, filters, top_k, rerank flag.

        Returns:
            List of SearchResult sorted by final_score descending.

        Raises:
            VectorStoreError: If embedding or Qdrant query fails.
        """
        dense_vector = await self._embed_dense(query.query)
        sparse_vector = self._embed_sparse(query.query)
        qdrant_filter = self._build_filter(query)

        try:
            hits = await self._client.query_points(
                collection_name=self._collection_name,
                prefetch=[
                    # [WHY] Prefetch pulls candidates from each vector
                    # space independently before fusion. Fetch top_k * 3
                    # so RRF has enough candidates to work with.
                    Prefetch(
                        query=dense_vector,
                        using="dense",
                        filter=qdrant_filter,
                        limit=query.top_k * 3,
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_vector.indices.tolist(),
                            values=sparse_vector.values.tolist(),
                        ),
                        using="sparse",
                        filter=qdrant_filter,
                        limit=query.top_k * 3,
                    ),
                ],
                # [WHY] FusionQuery(RRF) tells Qdrant to merge the
                # two prefetch result sets using Reciprocal Rank Fusion.
                # Built-in — no custom implementation needed.
                query=FusionQuery(fusion=Fusion.RRF),
                limit=query.top_k * 3,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(
                message="Qdrant hybrid search failed",
                detail=str(exc),
            ) from exc

        results = self._parse_hits(hits.points)

        if query.rerank and results:
            try:
                results = await self._reranker.rerank(query.query, results)
            except Exception as exc:
                # [WHY] Reranker failure non-fatal — return RRF order.
                logger.error(
                    "Reranker failed — using RRF order",
                    extra={"error": str(exc)},
                )

        final = results[: query.top_k]

        logger.info(
            "Hybrid search complete",
            extra={
                "query": query.query,
                "candidates": len(results),
                "returned": len(final),
            },
        )

        return final

    async def _embed_dense(self, text: str) -> list[float]:
        """
        Generate dense embedding for query text.

        Raises:
            VectorStoreError: If embedding fails.
        """
        try:
            vectors = await self._embedder._embeddings.aembed_documents([text])
            return vectors[0]
        except Exception as exc:
            raise VectorStoreError(
                message="Dense query embedding failed",
                detail=str(exc),
            ) from exc

    def _embed_sparse(self, text: str):
        """
        Generate sparse SPLADE embedding for query text.

        Raises:
            VectorStoreError: If sparse embedding fails.
        """
        try:
            # [WHY] embed() returns a generator — next() pulls first result.
            return next(self._sparse_model.embed([text]))
        except Exception as exc:
            raise VectorStoreError(
                message="Sparse query embedding failed",
                detail=str(exc),
            ) from exc

    def _build_filter(self, query: SearchQuery) -> Filter | None:
        """Build Qdrant payload filter from optional query fields."""
        conditions = []

        if query.project_name:
            conditions.append(
                FieldCondition(
                    key="project_name",
                    match=MatchValue(value=query.project_name),
                )
            )
        if query.department:
            conditions.append(
                FieldCondition(
                    key="department",
                    match=MatchValue(value=query.department),
                )
            )
        if query.supplier:
            conditions.append(
                FieldCondition(
                    key="supplier",
                    match=MatchValue(value=query.supplier),
                )
            )

        return Filter(must=conditions) if conditions else None

    def _parse_hits(self, points) -> list[SearchResult]:
        """Parse Qdrant ScoredPoint list into SearchResult list."""
        results: list[SearchResult] = []

        for point in points:
            payload = point.payload or {}
            try:
                metadata = DocumentMetadata(**{
                    k: v for k, v in payload.items()
                    if k in DocumentMetadata.model_fields
                })
                results.append(
                    SearchResult(
                        chunk_id=payload.get("chunk_id", str(point.id)),
                        document_id=payload.get("document_id", ""),
                        content=payload.get("content", ""),
                        metadata=metadata,
                        # [WHY] Qdrant returns one fused RRF score.
                        # We store it in all three score fields for
                        # schema consistency. Reranker overwrites final_score.
                        vector_score=float(point.score),
                        bm25_score=float(point.score),
                        rerank_score=None,
                        final_score=float(point.score),
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Skipping malformed hit",
                    extra={"id": point.id, "error": str(exc)},
                )
                continue

        return results