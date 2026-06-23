"""
Ingestion pipeline orchestrator.

Sequence: extract → clean → deduplicate → chunk → embed → upsert
Each stage is injected — pipeline has zero concrete dependencies.
"""

import logging
import uuid
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from app.core.exceptions import IngestionError
from app.models.document import DocumentMetadata
from app.services.ingestion.chunker import DocumentChunker
from app.services.ingestion.cleaner import DocumentCleaner
from app.services.ingestion.deduplicator import DocumentDeduplicator
from app.services.ingestion.embedder import DocumentEmbedder
from extractors.registry import ExtractorRegistry

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates the full document ingestion sequence.

    All dependencies injected via constructor — nothing instantiated
    inside methods. Fully testable with mocks.
    """

    def __init__(
        self,
        registry: ExtractorRegistry,
        cleaner: DocumentCleaner,
        deduplicator: DocumentDeduplicator,
        chunker: DocumentChunker,
        embedder: DocumentEmbedder,
        qdrant: AsyncQdrantClient,
        collection_name: str,
    ) -> None:
        self._registry = registry
        self._cleaner = cleaner
        self._deduplicator = deduplicator
        self._chunker = chunker
        self._embedder = embedder
        self._qdrant = qdrant
        self._collection_name = collection_name

    async def ingest(
        self,
        file_path: Path,
        project_name: str,
        department: str,
        source_name: str,
        source_type: str,
        supplier: str | None = None,
        milestone: str | None = None,
    ) -> dict[str, int]:
        """
        Run the full ingestion pipeline for one file.

        Args:
            file_path:    Absolute path to the file.
            project_name: Manufacturing project name.
            department:   Originating department.
            source_name:  Human-readable source label.
            source_type:  File category (pdf, spreadsheet, etc).
            supplier:     Optional supplier name.
            milestone:    Optional milestone name.

        Returns:
            Dict with ingestion stats:
            {
                "pages_extracted": int,
                "chunks_produced": int,
                "chunks_upserted": int,
                "duplicates_skipped": int,
            }

        Raises:
            IngestionError: If any pipeline stage fails.
        """
        source_id = str(uuid.uuid4())
        document_id = str(uuid.uuid4())

        base_metadata = DocumentMetadata(
            document_id=document_id,
            source_id=source_id,
            source_name=source_name,
            source_type=source_type,
            file_name=file_path.name,
            file_path=file_path,
            content_hash="",  # set per page by extractor
            project_name=project_name,
            department=department,
            supplier=supplier,
            milestone=milestone,
        )

        # ── Stage 1: Extract ───────────────────────────────────────────────
        try:
            raw_documents = await self._registry.extract(file_path, base_metadata)
        except Exception as exc:
            raise IngestionError(
                message=f"Extraction failed for {file_path.name}",
                detail=str(exc),
            ) from exc

        logger.info(
            "Extraction complete",
            extra={"file": file_path.name, "pages": len(raw_documents)},
        )

        # ── Stage 2: Clean + Deduplicate ───────────────────────────────────
        all_chunks = []
        duplicates_skipped = 0

        for raw_doc in raw_documents:
            cleaned = await self._cleaner.clean(raw_doc)

            if await self._deduplicator.is_duplicate(cleaned):
                duplicates_skipped += 1
                logger.debug(
                    "Duplicate skipped",
                    extra={"content_hash": cleaned.metadata.content_hash},
                )
                continue

            # ── Stage 3: Chunk ─────────────────────────────────────────────
            try:
                chunks = await self._chunker.chunk(cleaned)
                all_chunks.extend(chunks)
            except Exception as exc:
                raise IngestionError(
                    message="Chunking failed",
                    detail=str(exc),
                ) from exc

        if not all_chunks:
            logger.warning(
                "No chunks produced — all pages were duplicates or empty",
                extra={"file": file_path.name},
            )
            return {
                "pages_extracted": len(raw_documents),
                "chunks_produced": 0,
                "chunks_upserted": 0,
                "duplicates_skipped": duplicates_skipped,
            }

        # ── Stage 4: Embed ─────────────────────────────────────────────────
        try:
            embedded = await self._embedder.embed(all_chunks)
        except Exception as exc:
            raise IngestionError(
                message="Embedding failed",
                detail=str(exc),
            ) from exc

        # ── Stage 5: Upsert to Qdrant ──────────────────────────────────────
        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                # [WHY] Full metadata stored as payload so retrieval
                # results are self-contained — no secondary DB lookup.
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    **chunk.metadata.model_dump(mode="json"),
                },
            )
            for chunk, vector in embedded
        ]

        try:
            await self._qdrant.upsert(
                collection_name=self._collection_name,
                points=points,
                # [WHY] wait=True ensures upsert is confirmed before
                # returning. Without it, immediate search after ingest
                # may miss freshly added points.
                wait=True,
            )
        except Exception as exc:
            raise IngestionError(
                message="Qdrant upsert failed",
                detail=str(exc),
            ) from exc

        stats = {
            "pages_extracted": len(raw_documents),
            "chunks_produced": len(all_chunks),
            "chunks_upserted": len(points),
            "duplicates_skipped": duplicates_skipped,
        }

        logger.info("Ingestion complete", extra={"file": file_path.name, **stats})
        return stats