"""Unit tests for DocumentDeduplicator stub."""

import pytest
from pathlib import Path
from app.services.ingestion.deduplicator import DocumentDeduplicator
from app.models.document import RawDocument, DocumentMetadata


@pytest.fixture
def deduplicator() -> DocumentDeduplicator:
    return DocumentDeduplicator()


@pytest.fixture
def raw_document() -> RawDocument:
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
    return RawDocument(metadata=metadata, content="Test content.")


@pytest.mark.asyncio
async def test_deduplicator_always_returns_false(deduplicator, raw_document):
    """MVP: deduplicator must never block ingestion."""
    result = await deduplicator.is_duplicate(raw_document)
    assert result is False