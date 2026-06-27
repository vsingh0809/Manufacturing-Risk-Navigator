import time
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.exceptions import RetrievalError, VectorStoreError
from app.dependencies import get_hybrid_retriever
from app.models.search import HybridResult, SearchQuery
from app.services.retrieval.vector_store import HybridRetriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=HybridResult)
async def hybrid_search(
    query: SearchQuery,
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
) -> HybridResult:
    """
    Execute hybrid search over ingested project documents.
    """
    logger.info(
        "Search request received",
        extra={
            "query": query.query,
            "project_name": query.project_name,
            "top_k": query.top_k,
            "rerank": query.rerank,
        },
    )

    # 1. Start the timer for latency tracking
    start_time = time.perf_counter()

    try:
        # 2. Get the raw list of SearchResult objects
        raw_results = await retriever.search(query)
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.message,
        )
    except RetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.message,
        )
    except Exception as exc:
        logger.error("Unexpected search error", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed unexpectedly",
        )

    # 3. Stop the timer and calculate milliseconds
    latency = round((time.perf_counter() - start_time) * 1000, 2)

    # 4. Construct and return the exact Pydantic model required
    return HybridResult(
        query=query.query,
        results=raw_results,
        total_retrieved=len(raw_results),
        latency_ms=latency
    )