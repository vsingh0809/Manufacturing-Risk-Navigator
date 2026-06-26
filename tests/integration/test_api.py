"""Integration tests for API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

from app.models.search import HybridResult
from app.models.analysis import RiskReport
from app.models.observability import TokenUsage
from app.dependencies import get_hybrid_retriever, get_analysis_agent


@pytest.fixture
def mock_hybrid_result():
    return HybridResult(
        query="bearing delays in project atlas",
        results=[],
        total_retrieved=0,
        latency_ms=12.5,
    )


@pytest.fixture
def mock_risk_report():
    return RiskReport(
        report_id="report-001",
        project_name="Project Atlas",
        query="What risks affect turbine delivery?",
        risks=[],
        dependencies=[],
        summary="No major risks identified.",
        token_usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
    )


@pytest.fixture
def test_app(mock_settings):
    """Create test app with mocked services."""
    with patch("app.dependencies.initialise_services", new_callable=AsyncMock), \
         patch("app.dependencies.shutdown_services", new_callable=AsyncMock), \
         patch("app.core.config.get_settings", return_value=mock_settings):
        from app.main import create_app
        return create_app()


@pytest.mark.asyncio
async def test_health_check(test_app):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_readiness_check(test_app):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_search_endpoint(test_app, mock_hybrid_result):
    mock_retriever = MagicMock()
    
    # Use Pydantic's copy method with update to safely bypass the frozen restriction
    updated_result = mock_hybrid_result.model_copy(
        update={"query": "bearing delays in Project Atlas"}
    )
    mock_retriever.search = AsyncMock(return_value=updated_result)
    
    test_app.dependency_overrides[get_hybrid_retriever] = lambda: mock_retriever

    try:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/search",
                json={
                    "query": "bearing delays in Project Atlas",
                    "project_name": "Project Atlas",
                    "top_k": 5,
                    "rerank": False,
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "bearing delays in Project Atlas"
    finally:
        test_app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_analysis_endpoint(test_app, mock_risk_report):
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_risk_report)
    
    test_app.dependency_overrides[get_analysis_agent] = lambda: mock_agent

    try:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/analysis/run",
                json={
                    "query": "What risks affect turbine delivery?",
                    "project_name": "Project Atlas",
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["project_name"] == "Project Atlas"
    finally:
        test_app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_search_returns_503_on_vector_store_error(test_app):
    from app.core.exceptions import VectorStoreError

    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(
        side_effect=VectorStoreError(message="Qdrant unavailable")
    )
    
    test_app.dependency_overrides[get_hybrid_retriever] = lambda: mock_retriever

    try:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/search",
                json={"query": "test query", "top_k": 5, "rerank": False},
            )
        assert response.status_code == 503
    finally:
        test_app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_analysis_returns_503_on_agent_error(test_app):
    from app.core.exceptions import AgentError

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(
        side_effect=AgentError(message="LLM unavailable")
    )
    
    test_app.dependency_overrides[get_analysis_agent] = lambda: mock_agent

    try:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/analysis/run",
                json={
                    "query": "test query",
                    "project_name": "Project Atlas",
                },
            )
        assert response.status_code == 503
    finally:
        test_app.dependency_overrides.clear()
