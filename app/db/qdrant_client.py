"""
Qdrant async client factory.

Single responsibility: create, configure, and return an async Qdrant client.
Never imported directly into services — always injected via dependencies.py.
"""

import logging
from functools import lru_cache

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)

from app.core.config import AppSettings
from app.core.exceptions import StorageError

logger = logging.getLogger(__name__)


class QdrantClientFactory:
    """
    Builds and owns the async Qdrant client connection.

    One instance created at app startup via dependencies.py.
    Injected into every service that needs vector store access.
    """

    def __init__(self, settings: AppSettings) -> None:
        # [WHY] Settings injected — never read from env directly here.
        # Makes this class fully testable with a mock settings object.
        self._settings = settings
        self._client: AsyncQdrantClient | None = None

    async def connect(self) -> None:
        """
        Initialise the async Qdrant client and ensure collection exists.

        Called once during app lifespan startup.

        Raises:
            StorageError: If connection or collection setup fails.
        """
        try:
            self._client = AsyncQdrantClient(
                url=self._settings.qdrant_url,
                api_key=self._settings.qdrant_api_key,
                # [WHY] timeout=30 — Qdrant Cloud cold starts can be slow.
                # Default is 5s which causes false negatives on first request.
                timeout=30,
            )
            logger.info(
                "Qdrant client connected",
                extra={"url": self._settings.qdrant_url},
            )
        except Exception as exc:
            raise StorageError(
                message="Failed to connect to Qdrant",
                detail=str(exc),
            ) from exc

        await self._ensure_collection()

    async def disconnect(self) -> None:
        """
        Close the async Qdrant client connection.

        Called during app lifespan shutdown.
        """
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("Qdrant client disconnected")



    

    def get_client(self) -> AsyncQdrantClient:
        """
        Return the active async Qdrant client.

        Public method used by services via DI.

        Returns:
            AsyncQdrantClient instance.

        Raises:
            StorageError: If called before connect().
        """
        return self._get_client()

    def _get_client(self) -> AsyncQdrantClient:
        """
        Internal guard — ensures connect() was called before use.

        Raises:
            StorageError: If client is not initialised.
        """
        if self._client is None:
            raise StorageError(
                message="Qdrant client not initialised",
                detail="Call QdrantClientFactory.connect() during app startup",
            )
        return self._client
    
    async def _ensure_collection(self) -> None:
       
       client = self._get_client()
       collection_name = self._settings.qdrant_collection_name

       try:
        exists = await client.collection_exists(collection_name)
       except UnexpectedResponse as exc:
        raise StorageError(
            message=f"Failed to check collection existence: {collection_name}",
            detail=str(exc),
        ) from exc
       
       if exists:
        logger.info(
            "Collection already exists — skipping creation",
            extra={"collection": collection_name},
        )
        return
       
       try:

        await client.create_collection(
            collection_name=collection_name,
            # [WHY] Named vectors — "dense" for semantic search,
            # "sparse" for keyword/BM25-style search.
            # Both live in one collection, one query hits both.
            vectors_config={
                "dense": VectorParams(
                    size=self._settings.embedding_dimension,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        # [WHY] on_disk=True keeps sparse index
                        # off RAM — sparse vectors are large.
                        on_disk=True,
                    )
                ),
            },
        )
        logger.info(
            "Collection created with dense + sparse vectors",
            extra={
                "collection": collection_name,
                "dimension": self._settings.embedding_dimension,
            },
        )
       except UnexpectedResponse as exc:
        raise StorageError(
            message=f"Failed to create collection: {collection_name}",
            detail=str(exc),
        ) from exc

    

    

    