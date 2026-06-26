"""
Dependency injection container.

Every service is instantiated once here and injected
into routes via FastAPI's Depends() system.
Nothing is instantiated inside route handlers.
"""

import logging
from functools import lru_cache
from app.core.exceptions import AgentError
from app.core.exceptions import EmbeddingError
from app.core.exceptions import StorageError
from app.core.exceptions import IngestionError

from app.core.config import AppSettings, get_settings
from app.db.qdrant_client import QdrantClientFactory
from app.services.agent.analysis_agent import AnalysisAgent
from app.services.ingestion.chunker import DocumentChunker
from app.services.ingestion.cleaner import DocumentCleaner
from app.services.ingestion.deduplicator import DocumentDeduplicator
from app.services.ingestion.embedder import DocumentEmbedder
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.retrieval.reranker import MsMarcoReranker
from app.services.retrieval.vector_store import HybridRetriever
from extractors.pdf import PdfExtractor
from extractors.registry import ExtractorRegistry
from extractors.spreadsheet import SpreadsheetExtractor
from extractors.text import TextExtractor
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# ── Singletons (created once at startup) ──────────────────────────────────────

# [WHY] Module-level singletons for services that are expensive
# to initialise (model loading, connection pools).
# These are set during app lifespan startup — not at import time.
_qdrant_factory: QdrantClientFactory | None = None
_hybrid_retriever: HybridRetriever | None = None
_analysis_agent: AnalysisAgent | None = None
_ingestion_pipeline: IngestionPipeline | None = None
_llm: ChatGroq | None = None
_embeddings: HuggingFaceEmbeddings | None = None


async def initialise_services(settings: AppSettings) -> None:
    """
    Initialise all services at app startup.

    Called from app lifespan. Order matters:
    1. Qdrant client (others depend on it)
    2. Embedder (retriever depends on it)
    3. Reranker (retriever depends on it)
    4. HybridRetriever (agent depends on it)
    5. AnalysisAgent
    6. IngestionPipeline
    """
    global _qdrant_factory, _hybrid_retriever, _analysis_agent, _ingestion_pipeline,_llm,_embeddings

    logger.info("Initialising services...")

    # ── Qdrant ────────────────────────────────────────────────────────────
    _qdrant_factory = QdrantClientFactory(settings=settings)
    await _qdrant_factory.connect()

    qdrant_client = _qdrant_factory.get_client()

    # ── LLM — created ONCE ────────────────────────────────────
    _llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,
        max_tokens=2048,
    )

    # ── Embeddings — created ONCE ─────────────────────────────
    _embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )



    # ── Embedder ──────────────────────────────────────────────────────────
    embedder = DocumentEmbedder(embeddings=_embeddings)

    # ── Reranker ──────────────────────────────────────────────────────────
    reranker = MsMarcoReranker()

    # ── Hybrid Retriever ──────────────────────────────────────────────────
    _hybrid_retriever = HybridRetriever(
        client=qdrant_client,
        collection_name=settings.qdrant_collection_name,
        embedder=embedder,
        reranker=reranker,
        settings=settings,
    )

    # ── Analysis Agent ────────────────────────────────────────────────────
    _analysis_agent = AnalysisAgent(
        retriever=_hybrid_retriever,
        llm=_llm,                                        # injected
        settings=settings,
    )

    # ── Extractor Registry ────────────────────────────────────────────────
    registry = ExtractorRegistry()
    registry.register(PdfExtractor())
    registry.register(SpreadsheetExtractor())
    registry.register(TextExtractor())

    # ── Ingestion Pipeline ────────────────────────────────────────────────
    _ingestion_pipeline = IngestionPipeline(
        registry=registry,
        cleaner=DocumentCleaner(),
        deduplicator=DocumentDeduplicator(),
        chunker=DocumentChunker(settings=settings),
        embedder=embedder,
        qdrant=qdrant_client,
        collection_name=settings.qdrant_collection_name,
    )

    logger.info("All services initialised successfully")


async def shutdown_services() -> None:
    """
    Gracefully shut down services at app shutdown.
    """
    global _qdrant_factory

    if _qdrant_factory is not None:
        await _qdrant_factory.disconnect()

    logger.info("All services shut down")


# ── FastAPI Dependency Functions ───────────────────────────────────────────────
# [WHY] These functions are passed to FastAPI's Depends().
# FastAPI calls them per request and injects the return value.
# They never instantiate anything — they return existing singletons.

def get_hybrid_retriever() -> HybridRetriever:
    """Inject HybridRetriever into routes."""
    if _hybrid_retriever is None:
        raise StorageError(
            message="HybridRetriever not initialised",
            detail="App startup may have failed",
        )
    return _hybrid_retriever

def get_llm() -> ChatGroq:
    if _llm is None:
        raise AgentError(message="LLM not initialised")
    return _llm

def get_embeddings() -> HuggingFaceEmbeddings:
    if _embeddings is None:
        raise EmbeddingError(message="Embeddings not initialised")
    return _embeddings

def get_analysis_agent() -> AnalysisAgent:
    """Inject AnalysisAgent into routes."""
    if _analysis_agent is None:
        
        raise AgentError(
            message="AnalysisAgent not initialised",
            detail="App startup may have failed",
        )
    return _analysis_agent


def get_ingestion_pipeline() -> IngestionPipeline:
    """Inject IngestionPipeline into routes."""
    if _ingestion_pipeline is None:
        raise IngestionError(
            message="IngestionPipeline not initialised",
            detail="App startup may have failed",
        )
    return _ingestion_pipeline


def get_app_settings() -> AppSettings:
    """Inject AppSettings into routes."""
    return get_settings()