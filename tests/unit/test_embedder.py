"""Unit tests for DocumentEmbedder."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from app.services.ingestion.embedder import DocumentEmbedder
from app.models.document import ChunkedDocument, DocumentMetadata
from app.core.exceptions import EmbeddingError


@pytest.fixture
def mock_chunks() -> list[ChunkedDocument]:
    metadata = DocumentMetadata(
        document_id="doc-001",
        source_id="src-001",
        source_name="Test",
        source_type="pdf",
        file_name="test.pdf",
        file_path=Path("/tmp/test.pdf"),
        content_hash="hash123",
        project_name="Project Atlas",
        department="engineering",
    )
    return [
        ChunkedDocument(
            chunk_id=f"chunk-00{i}",
            document_id="doc-001",
            source_chunk_id=None,
            metadata=metadata,
            content=f"Content chunk {i}",
            chunk_index=i,
            token_count=10,
        )
        for i in range(3)
    ]


@pytest.fixture
def embedder(mock_settings):
    with patch("app.services.ingestion.embedder.AzureOpenAIEmbeddings") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield DocumentEmbedder(settings=mock_settings), mock_instance


@pytest.mark.asyncio
async def test_embed_returns_correct_pairs(embedder, mock_chunks):
    embedder_instance, mock_openai = embedder
    mock_openai.aembed_documents = AsyncMock(
        return_value=[[0.1] * 1536 for _ in mock_chunks]
    )
    result = await embedder_instance.embed(mock_chunks)
    assert len(result) == len(mock_chunks)
    for chunk, vector in result:
        assert len(vector) == 1536


@pytest.mark.asyncio
async def test_embed_empty_list_returns_empty(embedder):
    embedder_instance, _ = embedder
    result = await embedder_instance.embed([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_raises_on_api_failure(embedder, mock_chunks):
    embedder_instance, mock_openai = embedder
    mock_openai.aembed_documents = AsyncMock(side_effect=Exception("API error"))
    with pytest.raises(EmbeddingError):
        await embedder_instance.embed(mock_chunks)


@pytest.mark.asyncio
async def test_embed_raises_on_vector_count_mismatch(embedder, mock_chunks):
    embedder_instance, mock_openai = embedder
    # Return fewer vectors than chunks
    mock_openai.aembed_documents = AsyncMock(return_value=[[0.1] * 1536])
    with pytest.raises(EmbeddingError):
        await embedder_instance.embed(mock_chunks)