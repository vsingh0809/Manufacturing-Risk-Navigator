"""Integration tests for IngestionPipeline."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.ingestion.pipeline import IngestionPipeline
from app.services.ingestion.chunker import DocumentChunker
from app.services.ingestion.cleaner import DocumentCleaner
from app.services.ingestion.deduplicator import DocumentDeduplicator
from app.services.ingestion.embedder import DocumentEmbedder
from app.core.exceptions import IngestionError
from extractors.registry import ExtractorRegistry
from app.models.document import DocumentMetadata, RawDocument, ChunkedDocument


@pytest.fixture
def mock_pipeline(mock_settings):
    """Build pipeline with all stages mocked."""
    mock_registry = MagicMock(spec=ExtractorRegistry)
    mock_cleaner = MagicMock(spec=DocumentCleaner)
    mock_deduplicator = MagicMock(spec=DocumentDeduplicator)
    mock_chunker = MagicMock(spec=DocumentChunker)
    mock_embedder = MagicMock(spec=DocumentEmbedder)
    mock_qdrant = MagicMock()

    pipeline = IngestionPipeline(
        registry=mock_registry,
        cleaner=mock_cleaner,
        deduplicator=mock_deduplicator,
        chunker=mock_chunker,
        embedder=mock_embedder,
        qdrant=mock_qdrant,
        collection_name="test_collection",
    )
    return pipeline, {
        "registry": mock_registry,
        "cleaner": mock_cleaner,
        "deduplicator": mock_deduplicator,
        "chunker": mock_chunker,
        "embedder": mock_embedder,
        "qdrant": mock_qdrant,
    }


@pytest.fixture
def sample_raw_document() -> RawDocument:
    metadata = DocumentMetadata(
        document_id="doc-001",
        source_id="src-001",
        source_name="Test Source",
        source_type="pdf",
        file_name="test.pdf",
        file_path=Path("/tmp/test.pdf"),
        content_hash="hash123",
        project_name="Project Atlas",
        department="procurement",
    )
    return RawDocument(metadata=metadata, content="Sample content for testing.")


@pytest.fixture
def sample_chunk(sample_raw_document) -> ChunkedDocument:
    return ChunkedDocument(
        chunk_id="chunk-001",
        document_id="doc-001",
        source_chunk_id=None,
        metadata=sample_raw_document.metadata,
        content="Sample content for testing.",
        chunk_index=0,
        token_count=5,
    )


@pytest.mark.asyncio
async def test_pipeline_full_flow_returns_stats(
    mock_pipeline, sample_raw_document, sample_chunk
):
    pipeline, mocks = mock_pipeline
    mocks["registry"].extract = AsyncMock(return_value=[sample_raw_document])
    mocks["cleaner"].clean = AsyncMock(return_value=sample_raw_document)
    mocks["deduplicator"].is_duplicate = AsyncMock(return_value=False)
    mocks["chunker"].chunk = AsyncMock(return_value=[sample_chunk])
    mocks["embedder"].embed = AsyncMock(return_value=[(sample_chunk, [0.1] * 1536)])
    mocks["qdrant"].upsert = AsyncMock()

    stats = await pipeline.ingest(
        file_path=Path("/tmp/test.pdf"),
        project_name="Project Atlas",
        department="procurement",
        source_name="Test Source",
        source_type="pdf",
    )

    assert stats["pages_extracted"] == 1
    assert stats["chunks_produced"] == 1
    assert stats["chunks_upserted"] == 1
    assert stats["duplicates_skipped"] == 0


@pytest.mark.asyncio
async def test_pipeline_skips_duplicates(
    mock_pipeline, sample_raw_document
):
    pipeline, mocks = mock_pipeline
    mocks["registry"].extract = AsyncMock(return_value=[sample_raw_document])
    mocks["cleaner"].clean = AsyncMock(return_value=sample_raw_document)
    mocks["deduplicator"].is_duplicate = AsyncMock(return_value=True)

    stats = await pipeline.ingest(
        file_path=Path("/tmp/test.pdf"),
        project_name="Project Atlas",
        department="procurement",
        source_name="Test Source",
        source_type="pdf",
    )

    assert stats["duplicates_skipped"] == 1
    assert stats["chunks_upserted"] == 0
    mocks["chunker"].chunk.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_raises_on_extraction_failure(mock_pipeline):
    pipeline, mocks = mock_pipeline
    mocks["registry"].extract = AsyncMock(side_effect=Exception("Extract failed"))

    with pytest.raises(IngestionError):
        await pipeline.ingest(
            file_path=Path("/tmp/test.pdf"),
            project_name="Project Atlas",
            department="procurement",
            source_name="Test Source",
            source_type="pdf",
        )


@pytest.mark.asyncio
async def test_pipeline_raises_on_upsert_failure(
    mock_pipeline, sample_raw_document, sample_chunk
):
    pipeline, mocks = mock_pipeline
    mocks["registry"].extract = AsyncMock(return_value=[sample_raw_document])
    mocks["cleaner"].clean = AsyncMock(return_value=sample_raw_document)
    mocks["deduplicator"].is_duplicate = AsyncMock(return_value=False)
    mocks["chunker"].chunk = AsyncMock(return_value=[sample_chunk])
    mocks["embedder"].embed = AsyncMock(return_value=[(sample_chunk, [0.1] * 1536)])
    mocks["qdrant"].upsert = AsyncMock(side_effect=Exception("Qdrant down"))

    with pytest.raises(IngestionError):
        await pipeline.ingest(
            file_path=Path("/tmp/test.pdf"),
            project_name="Project Atlas",
            department="procurement",
            source_name="Test Source",
            source_type="pdf",
        )