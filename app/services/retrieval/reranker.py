"""
Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Scores (query, chunk) pairs directly — more accurate than
vector similarity but slower. Applied after RRF fusion
to final candidate pool only.
"""

import logging
from typing import Protocol, runtime_checkable

from sentence_transformers import CrossEncoder

from app.core.exceptions import RerankerError
from app.models.search import SearchResult

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@runtime_checkable
class BaseReranker(Protocol):
    """
    Structural contract for all reranker implementations.
    Swap ms-marco for any cross-encoder: implement + inject.
    """

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Rerank results by (query, content) cross-encoder score.

        Returns results sorted by rerank_score descending.
        """
        ...


class MsMarcoReranker:
    """
    Reranker using ms-marco-MiniLM-L-6-v2 cross-encoder.

    Loaded once at startup — model loading is expensive (~400ms).
    Inference per batch is fast (~50ms for 30 candidates).

    Satisfies BaseReranker Protocol without explicit inheritance.
    """

    def __init__(self) -> None:
        try:
            # [WHY] CrossEncoder loaded at init not at request time.
            # Loading on first request causes a cold-start latency spike
            # that would fail tight SLA requirements.
            self._model = CrossEncoder(
                _MODEL_NAME,
                max_length=512,
            )
            logger.info(
                "CrossEncoder loaded",
                extra={"model": _MODEL_NAME},
            )
        except Exception as exc:
            raise RerankerError(
                message="Failed to load CrossEncoder model",
                detail=str(exc),
            ) from exc

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Score each (query, content) pair and sort by score.

        Args:
            query:   Original search query string.
            results: Candidate SearchResults from RRF fusion.

        Returns:
            Results sorted by rerank_score descending with
            rerank_score and final_score updated.

        Raises:
            RerankerError: If cross-encoder inference fails.
        """
        if not results:
            return []

        pairs = [[query, result.content] for result in results]

        try:
            # [WHY] predict() is synchronous CPU-bound inference.
            # For MVP this is acceptable — model runs in ~50ms.
            # Sprint 2: wrap in asyncio.to_thread for true async.
            scores = self._model.predict(pairs)
        except Exception as exc:
            raise RerankerError(
                message="CrossEncoder inference failed",
                detail=str(exc),
            ) from exc

        reranked: list[SearchResult] = []

        for result, score in zip(results, scores):
            reranked.append(
                result.model_copy(
                    update={
                        "rerank_score": float(score),
                        "final_score": float(score),
                    }
                )
            )

        reranked.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)

        logger.info(
            "Reranking complete",
            extra={"candidates": len(results), "top_score": reranked[0].rerank_score},
        )

        return reranked