"""Unit tests for LangGraph node functions."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from app.services.agent.nodes import (
    retrieve_node,
    risk_node,
    dependency_node,
    summarise_node,
    _trim_chunks_to_budget,
    _build_context,
)
from app.models.search import SearchResult, SearchQuery
from app.models.document import DocumentMetadata
from app.models.observability import TokenUsage


def make_chunk(chunk_id: str, content: str = "test content") -> SearchResult:
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
        vector_score=0.9,
        bm25_score=0.8,
        rerank_score=0.95,
        final_score=0.95,
    )


@pytest.fixture
def base_state():
    return {
        "query": "What risks affect turbine delivery?",
        "project_name": "Project Atlas",
        "retrieved_chunks": [],
        "context_truncated": False,
        "risks": [],
        "dependencies": [],
        "milestones": [],
        "summary": "",
        "token_usage": TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        ),
        "error": None,
    }


# ── retrieve_node ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieve_node_returns_chunks(base_state, mock_settings):
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(
        return_value=[make_chunk("chunk-001"), make_chunk("chunk-002")]
    )

    result = await retrieve_node(
        base_state,
        retriever=mock_retriever,
        settings=mock_settings,
    )

    assert len(result["retrieved_chunks"]) == 2
    assert result["context_truncated"] is False


@pytest.mark.asyncio
async def test_retrieve_node_handles_empty_results(base_state, mock_settings):
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])

    result = await retrieve_node(
        base_state,
        retriever=mock_retriever,
        settings=mock_settings,
    )

    assert result["retrieved_chunks"] == []
    assert result["context_truncated"] is False


@pytest.mark.asyncio
async def test_retrieve_node_handles_retriever_failure(base_state, mock_settings):
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(side_effect=Exception("retriever down"))

    result = await retrieve_node(
        base_state,
        retriever=mock_retriever,
        settings=mock_settings,
    )

    # [WHY] Retriever failure must not crash agent — returns empty
    assert result["retrieved_chunks"] == []


# ── token budget ───────────────────────────────────────────────────────────────

def test_trim_chunks_within_budget():
    chunks = [make_chunk(f"chunk-{i}", "short content") for i in range(5)]
    trimmed, truncated = _trim_chunks_to_budget(chunks, budget=10_000)
    assert len(trimmed) == 5
    assert truncated is False


def test_trim_chunks_exceeds_budget():
    # Each chunk content is ~500 tokens worth of text
    long_content = "word " * 600
    chunks = [make_chunk(f"chunk-{i}", long_content) for i in range(10)]
    trimmed, truncated = _trim_chunks_to_budget(chunks, budget=2_000)
    assert len(trimmed) < 10
    assert truncated is True


# ── risk_node ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_risk_node_empty_chunks_returns_empty(base_state):
    mock_llm = MagicMock()
    result = await risk_node(base_state, llm=mock_llm)
    assert result["risks"] == []
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_risk_node_parses_llm_output(base_state):
    base_state["retrieved_chunks"] = [make_chunk("chunk-001")]

    mock_response = MagicMock()
    
    mock_response.content = '''[
        {
            "risk_id": "risk-001",
            "category": "DELIVERY_DELAY",
            "severity": "HIGH",
            "description": "Bearing shipment delayed by 3 weeks",
            "affected_project": "Project Atlas",
            "affected_milestone": "Turbine Assembly",
            "supplier": "Supplier X",
            "source_chunk_ids": ["chunk-001"]
        }
    ]'''
    mock_response.response_metadata = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}}

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    result = await risk_node(base_state, llm=mock_llm)

    assert len(result["risks"]) == 1
    assert result["risks"][0].category == "DELIVERY_DELAY"
    assert result["risks"][0].severity == "HIGH"


@pytest.mark.asyncio
async def test_risk_node_skips_malformed_items(base_state):
    base_state["retrieved_chunks"] = [make_chunk("chunk-001")]

    mock_response = MagicMock()
    # Missing required fields — should be skipped not crash
    mock_response.content = '[{"bad_field": "bad_value"}]'
    mock_response.response_metadata = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    result = await risk_node(base_state, llm=mock_llm)
    assert result["risks"] == []


# ── dependency_node ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dependency_node_empty_chunks_returns_empty(base_state):
    mock_llm = MagicMock()
    result = await dependency_node(base_state, llm=mock_llm)
    assert result["dependencies"] == []


@pytest.mark.asyncio
async def test_dependency_node_parses_llm_output(base_state):
    base_state["retrieved_chunks"] = [make_chunk("chunk-001")]

    mock_response = MagicMock()
    mock_response.content = '''[
        {
            "from_task": "Bearing Delivery",
            "to_task": "Turbine Assembly",
            "dependency_type": "blocks",
            "source_chunk_ids": ["chunk-001"]
        }
    ]'''
    mock_response.response_metadata = {"token_usage": {"prompt_tokens": 80, "completion_tokens": 40}}

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    result = await dependency_node(base_state, llm=mock_llm)

    assert len(result["dependencies"]) == 1
    assert result["dependencies"][0].from_task == "Bearing Delivery"
    assert result["dependencies"][0].dependency_type == "blocks"


# ── summarise_node ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_summarise_node_empty_chunks_returns_default(base_state):
    mock_llm = MagicMock()
    result = await summarise_node(base_state, llm=mock_llm)
    assert "No relevant project data" in result["summary"]
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_summarise_node_returns_llm_text(base_state):
    base_state["retrieved_chunks"] = [make_chunk("chunk-001")]

    mock_response = MagicMock()
    mock_response.content = "Project Atlas faces critical delays due to bearing shipment issues."
    mock_response.response_metadata = {"token_usage": {"prompt_tokens": 200, "completion_tokens": 60}}

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    result = await summarise_node(base_state, llm=mock_llm)
    assert "Project Atlas" in result["summary"]
    assert result["error"] is None