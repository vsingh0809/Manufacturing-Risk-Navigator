"""
Shared pytest fixtures across all test modules.
"""

import pytest
from app.core.config import AppSettings


@pytest.fixture
def mock_settings() -> AppSettings:
    """
    Return a test AppSettings instance with safe dummy values.
    Never hits real Qdrant or OpenAI in unit tests.
    """
    return AppSettings(
        openai_api_key="test-key",
        qdrant_url="http://localhost:6333",
        qdrant_api_key="test-qdrant-key",
        qdrant_collection_name="test_collection",
        embedding_dimension=1536,
        chunk_size=512,
        chunk_overlap=64,
    )