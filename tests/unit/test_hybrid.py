"""Unit tests for Qdrant native HybridRetriever."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from app.models.search import SearchQuery, SearchResult
from app.models.document import DocumentMetadata
from app.core.exceptions import VectorStoreError


def make_result(chunk_id: str, score: float = 0.9) -> SearchResult:
    metadata = DocumentMetadata(
        document_id="doc-001",
        source_id="src-001",
        source_name="Test",
        source_type="pdf",
        file_name="test.pdf",
        file_path=Path("/tmp/test.pdf"),
        content_hash="hash123",
        project_name="Project Atlas",
        department="procurement",
    )
    return SearchResult(
        chunk_id=chunk_id,
        document_id="doc-001",
        content=f"Content for {chunk_id}",
        metadata=metadata,
        vector_score=score,
        bm25_score=score,
        rerank_score=None,
        final_score=score,
    )


@pytest.fixture
def mock_retriever():
    with patch("app.services.retrieval.vector_store.SparseTextEmbedding"):
        from app.services.retrieval.vector_store import HybridRetriever

        mock_client = MagicMock()
        mock_embedder = MagicMock()
        mock_reranker = MagicMock()
        mock_settings = MagicMock()

        retriever = HybridRetriever(
            client=mock_client,
            collection_name="test_collection",
            embedder=mock_embedder,
            reranker=mock_reranker,
            settings=mock_settings,
        )
        return retriever, mock_client, mock_embedder, mock_reranker


@pytest.fixture
def base_query():
    return SearchQuery(query="bearing delays Project Atlas", top_k=5, rerank=False)


@pytest.mark.asyncio
async def test_search_returns_top_k(mock_retriever, base_query):
    retriever, mock_client, mock_embedder, _ = mock_retriever

    mock_embedder._embeddings.aembed_documents = AsyncMock(
        return_value=[[0.1] * 1536]
    )
    retriever._embed_sparse = MagicMock(
        return_value=MagicMock(
            indices=MagicMock(tolist=lambda: [1, 2]),
            values=MagicMock(tolist=lambda: [0.5, 0.3]),
        )
    )

    mock_point = MagicMock()
    mock_point.id = "chunk-001"
    mock_point.score = 0.9
    mock_point.payload = {
        "chunk_id": "chunk-001",
        "document_id": "doc-001",
        "content": "bearing delay content",
        "document_id": "doc-001",
        "source_id": "src-001",
        "source_name": "Test",
        "source_type": "pdf",
        "file_name": "test.pdf",
        "file_path": "/tmp/test.pdf",
        "content_hash": "hash123",
        "project_name": "Project Atlas",
        "department": "procurement",
        "risk_category": "UNKNOWN",
        "ingested_at": "2024-01-01T00:00:00+00:00",
        "created_at": "2024-01-01T00:00:00+00:00",
        "metadata": {},
    }

    mock_response = MagicMock()
    mock_response.points = [mock_point]
    mock_client.query_points = AsyncMock(return_value=mock_response)

    results = await retriever.search(base_query)
    assert len(results) <= base_query.top_k


@pytest.mark.asyncio
async def test_search_calls_reranker_when_enabled(mock_retriever):
    retriever, mock_client, mock_embedder, mock_reranker = mock_retriever
    query = SearchQuery(query="test", top_k=5, rerank=True)

    mock_embedder._embeddings.aembed_documents = AsyncMock(
        return_value=[[0.1] * 1536]
    )
    retriever._embed_sparse = MagicMock(
        return_value=MagicMock(
            indices=MagicMock(tolist=lambda: [1]),
            values=MagicMock(tolist=lambda: [0.5]),
        )
    )

    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points = AsyncMock(return_value=mock_response)
    mock_reranker.rerank = AsyncMock(return_value=[])

    await retriever.search(query)
    mock_reranker.rerank.assert_not_called()  # no results = no rerank


@pytest.mark.asyncio
async def test_search_raises_on_qdrant_failure(mock_retriever, base_query):
    retriever, mock_client, mock_embedder, _ = mock_retriever

    mock_embedder._embeddings.aembed_documents = AsyncMock(
        return_value=[[0.1] * 1536]
    )
    retriever._embed_sparse = MagicMock(
        return_value=MagicMock(
            indices=MagicMock(tolist=lambda: [1]),
            values=MagicMock(tolist=lambda: [0.5]),
        )
    )
    mock_client.query_points = AsyncMock(side_effect=Exception("Qdrant down"))

    with pytest.raises(VectorStoreError):
        await retriever.search(base_query)


@pytest.mark.asyncio
async def test_dense_embedding_failure_raises(mock_retriever, base_query):
    retriever, _, mock_embedder, _ = mock_retriever
    mock_embedder._embeddings.aembed_documents = AsyncMock(
        side_effect=Exception("Azure down")
    )

    with pytest.raises(VectorStoreError):
        await retriever.search(base_query)