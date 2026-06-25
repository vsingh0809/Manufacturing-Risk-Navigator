"""Unit tests for AnalysisAgent entry point."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.analysis import RiskReport
from app.models.observability import TokenUsage
from app.core.exceptions import AgentError


@pytest.fixture
def mock_agent(mock_settings):
    with patch("app.services.agent.analysis_agent.AzureChatOpenAI"), \
         patch("app.services.agent.analysis_agent.build_graph") as mock_build:

        mock_graph = MagicMock()
        mock_build.return_value = mock_graph

        from app.services.agent.analysis_agent import AnalysisAgent
        mock_retriever = MagicMock()

        agent = AnalysisAgent(
            retriever=mock_retriever,
            settings=mock_settings,
        )
        return agent, mock_graph


@pytest.mark.asyncio
async def test_agent_returns_risk_report(mock_agent):
    agent, mock_graph = mock_agent

    mock_graph.ainvoke = AsyncMock(return_value={
        "risks": [],
        "dependencies": [],
        "milestones": [],
        "summary": "No major risks identified.",
        "token_usage": TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
        "error": None,
    })

    report = await agent.run(
        query="What risks affect Project Atlas?",
        project_name="Project Atlas",
    )

    assert isinstance(report, RiskReport)
    assert report.project_name == "Project Atlas"
    assert report.token_usage.total_tokens == 150


@pytest.mark.asyncio
async def test_agent_raises_on_graph_failure(mock_agent):
    agent, mock_graph = mock_agent
    mock_graph.ainvoke = AsyncMock(side_effect=Exception("graph crashed"))

    with pytest.raises(AgentError):
        await agent.run(
            query="What risks affect Project Atlas?",
            project_name="Project Atlas",
        )


@pytest.mark.asyncio
async def test_agent_handles_empty_state_gracefully(mock_agent):
    agent, mock_graph = mock_agent

    mock_graph.ainvoke = AsyncMock(return_value={
        "risks": [],
        "dependencies": [],
        "milestones": [],
        "summary": "Analysis failed gracefully due to an empty execution state",
        "token_usage": TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        "error": "partial failure",
    })

    report = await agent.run(
        query="test query",
        project_name="Project Atlas",
    )

    assert isinstance(report, RiskReport)
    assert report.risks == []