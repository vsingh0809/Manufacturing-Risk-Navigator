"""Unit tests for DocumentCleaner stub."""

import pytest
from pathlib import Path
from app.services.ingestion.cleaner import DocumentCleaner
from app.models.document import RawDocument, DocumentMetadata


@pytest.fixture
def cleaner() -> DocumentCleaner:
    return DocumentCleaner()


@pytest.fixture
def raw_document() -> RawDocument:
    metadata = DocumentMetadata(
        document_id="doc-001",
        source_id="src-001",
        source_name="Test",
        source_type="text",
        file_name="test.txt",
        file_path=Path("/tmp/test.txt"),
        content_hash="hash123",
        project_name="Project Atlas",
        department="QA",
    )
    return RawDocument(metadata=metadata, content="Test content.")


@pytest.mark.asyncio
async def test_cleaner_returns_same_document(cleaner, raw_document):
    """MVP: cleaner must return document untouched."""
    result = await cleaner.clean(raw_document)
    assert result == raw_document


@pytest.mark.asyncio
async def test_cleaner_preserves_content(cleaner, raw_document):
    result = await cleaner.clean(raw_document)
    assert result.content == raw_document.content