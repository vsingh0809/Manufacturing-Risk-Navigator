"""
Timeline API endpoints.

GET /timeline/{project_name} → dependency graph + milestone status
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.exceptions import AgentError
from app.dependencies import get_analysis_agent
from app.models.analysis import RiskReport
from app.models.timeline import DependencyGraph
from app.services.agent.analysis_agent import AnalysisAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/timeline", tags=["timeline"])


@router.get("/{project_name}", response_model=DependencyGraph)
async def get_project_timeline(
    project_name: str,
    agent: AnalysisAgent = Depends(get_analysis_agent),
) -> DependencyGraph:
    """
    Infer project timeline and dependency graph.

    Runs the analysis agent with a timeline-focused query
    and returns the dependency graph with milestone statuses.

    Args:
        project_name: Project to analyse.

    Returns:
        DependencyGraph with milestones and dependency edges.
    """
    logger.info(
        "Timeline request received",
        extra={"project_name": project_name},
    )

    try:
        report: RiskReport = await agent.run(
            query=f"What are the current milestone statuses and dependency blockers for {project_name}?",
            project_name=project_name,
        )
    except AgentError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.message,
        )
    except Exception as exc:
        logger.error("Timeline analysis failed", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Timeline analysis failed unexpectedly",
        )

    from datetime import UTC, datetime
    return DependencyGraph(
        project_name=project_name,
        milestones=report.milestones if hasattr(report, "milestones") else [],
        edges=report.dependencies,
        generated_at=datetime.now(UTC),
    )