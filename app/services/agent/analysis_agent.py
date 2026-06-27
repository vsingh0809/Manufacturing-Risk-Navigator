"""
Analysis agent entry point.

Builds the LangGraph, runs it with input state,
and converts final state into a typed RiskReport.
"""

import logging
import uuid
from datetime import UTC, datetime
from langchain_core.language_models import BaseChatModel

from app.core.config import AppSettings
from app.core.exceptions import AgentError
from app.models.analysis import RiskReport
from app.models.observability import TokenUsage
from app.services.agent.graph import build_graph
from app.services.retrieval.vector_store import HybridRetriever

logger = logging.getLogger(__name__)


class AnalysisAgent:
    """
    Orchestrates the LangGraph risk analysis workflow.

    One instance created at app startup and reused per request.
    Graph is compiled once — execution is per invocation.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: BaseChatModel,
        settings: AppSettings,
    ) -> None:
        # [WHY] LLM initialised once here — not per request.
        # AzureChatOpenAI connection pool is reused across calls.
        try:
            self._llm = llm
        except Exception as exc:
            raise AgentError(
                message="Failed to initialise AzureChatOpenAI",
                detail=str(exc),
            ) from exc

        self._graph = build_graph(
            retriever=retriever,
            llm=self._llm,
            settings=settings,
        )
        logger.info("AnalysisAgent initialised")

    async def run(
        self,
        query: str,
        project_name: str,
    ) -> RiskReport:
        """
        Execute the analysis graph and return a typed RiskReport.

        Args:
            query:        User's natural language query.
            project_name: Project to scope the analysis to.

        Returns:
            RiskReport with risks, dependencies, milestones, summary.

        Raises:
            AgentError: If graph execution fails unrecoverably.
        """
        initial_state = {
    "query": query,
    "project_name": project_name,
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

        try:
            final_state = await self._graph.ainvoke(initial_state)
        except Exception as exc:
    # ADD this before raise
         logger.error(
        "Graph execution failed",
        extra={
            "error_type": type(exc).__name__,
            "error": str(exc),
            "query": query,
            "project": project_name,
        },
    )
         raise AgentError(
        message="Agent graph execution failed",
        detail=str(exc),   # ← this will now show real cause
           ) from exc
            

        return RiskReport(
            report_id=str(uuid.uuid4()),
            project_name=project_name,
            query=query,
            risks=final_state.get("risks", []),
            dependencies=final_state.get("dependencies", []),
            summary=final_state.get("summary", "Analysis could not be completed."),
            token_usage=final_state.get(
                "token_usage",
                TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            ),
            generated_at=datetime.now(UTC),
        )