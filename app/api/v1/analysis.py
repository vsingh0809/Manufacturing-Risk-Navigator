"""
Analysis API endpoints.

POST /analysis/run → trigger LangGraph risk analysis agent
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.exceptions import AgentError
from app.dependencies import get_analysis_agent
from app.models.analysis import RiskReport
from app.services.agent.analysis_agent import AnalysisAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])


class AnalysisRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language query")
    project_name: str = Field(..., description="Project to scope analysis to")


@router.post("/run", response_model=RiskReport)
async def run_analysis(
    request: AnalysisRequest,
    agent: AnalysisAgent = Depends(get_analysis_agent),
) -> RiskReport:
    """
    Run the LangGraph risk analysis agent.

    Retrieves relevant chunks, identifies risks and dependencies,
    infers timeline status, and returns a grounded RiskReport.

    Args:
        request: AnalysisRequest with query and project_name.

    Returns:
        RiskReport with risks, dependencies, milestones, summary,
        token usage, and source citations.
    """
    logger.info(
        "Analysis request received",
        extra={
            "query": request.query,
            "project_name": request.project_name,
        },
    )

    try:
        report = await agent.run(
            query=request.query,
            project_name=request.project_name,
        )
    except AgentError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.message,
        )
    except Exception as exc:
        logger.error("Unexpected analysis error", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed unexpectedly",
        )

    return report