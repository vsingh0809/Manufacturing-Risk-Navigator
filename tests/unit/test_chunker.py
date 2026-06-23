"""Unit tests for DocumentChunker."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, UTC
from pathlib import Path

from app.services.ingestion.chunker import DocumentChunker
from app.models.document import RawDocument, DocumentMetadata
from app.core.config import AppSettings


@pytest.fixture
def settings(mock_settings) -> AppSettings:
    return mock_settings


@pytest.fixture
def chunker(settings) -> DocumentChunker:
    return DocumentChunker(settings=settings)


@pytest.fixture
def raw_document() -> RawDocument:
    metadata = DocumentMetadata(
        document_id="doc-001",
        source_id="src-001",
        source_name="Test Source",
        source_type="pdf",
        file_name="test.pdf",
        file_path=Path("/tmp/test.pdf"),
        content_hash="abc123",
        project_name="Project Atlas",
        department="procurement",
    )
    return RawDocument(
        metadata=metadata,
        content="This is a test document. " * 100,
    )


@pytest.mark.asyncio
async def test_chunk_returns_at_least_one_chunk(chunker, raw_document):
    chunks = await chunker.chunk(raw_document)
    assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_chunk_ids_are_unique(chunker, raw_document):
    chunks = await chunker.chunk(raw_document)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_chunk_index_is_sequential(chunker, raw_document):
    chunks = await chunker.chunk(raw_document)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


@pytest.mark.asyncio
async def test_chunk_token_count_within_limit(chunker, raw_document):
    chunks = await chunker.chunk(raw_document)
    for chunk in chunks:
        assert chunk.token_count <= chunker._chunk_size


@pytest.mark.asyncio
async def test_chunk_inherits_document_id(chunker, raw_document):
    chunks = await chunker.chunk(raw_document)
    for chunk in chunks:
        assert chunk.document_id == raw_document.metadata.document_id


@pytest.mark.asyncio
async def test_chunk_source_chunk_id_is_none_mvp(chunker, raw_document):
    chunks = await chunker.chunk(raw_document)
    for chunk in chunks:
        assert chunk.source_chunk_id is None


@pytest.mark.asyncio
async def test_chunk_raises_on_empty_content(chunker):
    metadata = DocumentMetadata(
        document_id="doc-002",
        source_id="src-001",
        source_name="Test",
        source_type="pdf",
        file_name="empty.pdf",
        file_path=Path("/tmp/empty.pdf"),
        content_hash="xyz",
        project_name="Project Atlas",
        department="engineering",
    )
    from app.core.exceptions import ChunkingError
    with pytest.raises(ChunkingError):
        await chunker.chunk(RawDocument(metadata=metadata, content=" "))