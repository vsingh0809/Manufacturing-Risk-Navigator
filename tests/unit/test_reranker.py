"""Unit tests for MsMarcoReranker."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from app.models.search import SearchResult
from app.models.document import DocumentMetadata


def make_result(chunk_id: str, content: str) -> SearchResult:
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
        content=content,
        metadata=metadata,
        vector_score=0.8,
        bm25_score=0.5,
        rerank_score=None,
        final_score=0.8,
    )


@pytest.fixture
def mock_reranker():
    with patch("app.services.retrieval.reranker.CrossEncoder") as mock_cls:
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        from app.services.retrieval.reranker import MsMarcoReranker
        reranker = MsMarcoReranker()
        return reranker, mock_model


@pytest.mark.asyncio
async def test_reranker_sets_rerank_score(mock_reranker):
    reranker, mock_model = mock_reranker
    results = [make_result("chunk-001", "bearing delay"), make_result("chunk-002", "quality issue")]
    mock_model.predict.return_value = [0.9, 0.3]

    reranked = await reranker.rerank("bearing delays", results)

    assert reranked[0].rerank_score == 0.9
    assert reranked[1].rerank_score == 0.3


@pytest.mark.asyncio
async def test_reranker_sorts_by_score_descending(mock_reranker):
    reranker, mock_model = mock_reranker
    results = [make_result("chunk-low", "low relevance"), make_result("chunk-high", "high relevance")]
    mock_model.predict.return_value = [0.2, 0.95]

    reranked = await reranker.rerank("test query", results)
    assert reranked[0].chunk_id == "chunk-high"


@pytest.mark.asyncio
async def test_reranker_empty_input_returns_empty(mock_reranker):
    reranker, _ = mock_reranker
    result = await reranker.rerank("query", [])
    assert result == []


@pytest.mark.asyncio
async def test_reranker_raises_on_model_failure(mock_reranker):
    from app.core.exceptions import RerankerError
    reranker, mock_model = mock_reranker
    mock_model.predict.side_effect = Exception("model crashed")

    with pytest.raises(RerankerError):
        await reranker.rerank("query", [make_result("chunk-001", "content")])