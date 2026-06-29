"""
Analysis agent with LangSmith run metadata.
"""

import logging
import uuid
from datetime import UTC, datetime

from langchain_core.language_models import BaseChatModel
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from app.core.config import AppSettings
from app.core.exceptions import AgentError
from app.models.analysis import RiskReport
from app.models.observability import TokenUsage
from app.services.agent.graph import build_graph
from app.services.retrieval.vector_store import HybridRetriever

logger = logging.getLogger(__name__)


class AnalysisAgent:

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: BaseChatModel,
        settings: AppSettings,
    ) -> None:
        self._graph = build_graph(
            retriever=retriever,
            llm=llm,
            settings=settings,
        )
        self._settings = settings
        logger.info("AnalysisAgent initialised")

    # [WHY] @traceable decorator tells LangSmith to create a
    # named run for this method. Shows up as "risk_analysis"
    # in LangSmith UI as parent of all child runs.
    @traceable(
        name="risk_analysis",
        run_type="chain",
        tags=["agent", "manufacturing"],
    )
    async def run(
        self,
        query: str,
        project_name: str,
    ) -> RiskReport:
        """
        Execute analysis graph and return RiskReport.

        LangSmith automatically traces:
          - This method as parent run
          - Every LangGraph node as child run
          - Every LLM call as grandchild run
          - Token usage per LLM call
          - Latency per node
        """
        # [WHY] Attach metadata to current LangSmith run.
        # Visible in UI — helps filter traces by project.
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.add_metadata({
                "project_name": project_name,
                "query": query,
            })

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
            logger.error(
                "Graph execution failed",
                extra={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise AgentError(
                message="Agent graph execution failed",
                detail=str(exc),
            ) from exc

        return RiskReport(
            report_id=str(uuid.uuid4()),
            project_name=project_name,
            query=query,
            risks=final_state.get("risks", []),
            dependencies=final_state.get("dependencies", []),
            summary=final_state.get(
                "summary", "Analysis could not be completed."
            ),
            token_usage=final_state.get(
                "token_usage",
                TokenUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                ),
            ),
            generated_at=datetime.now(UTC),
        )