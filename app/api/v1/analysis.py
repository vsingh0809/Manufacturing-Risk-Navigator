"""
Analysis API endpoints.

POST /analysis/run → trigger LangGraph risk analysis agent
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from langsmith import Client as LangSmithClient
from pydantic import BaseModel

from app.core.config import AppSettings
from app.dependencies import get_app_settings


from app.core.exceptions import AgentError
from app.dependencies import get_analysis_agent
from app.models.analysis import RiskReport
from app.services.agent.analysis_agent import AnalysisAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])


class AnalysisRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language query")
    project_name: str = Field(..., description="Project to scope analysis to")

class FeedbackRequest(BaseModel):
    run_id: str
    score: float        # 0.0 to 1.0
    comment: str | None = None

@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    settings: AppSettings = Depends(get_app_settings),
) -> dict:
    """
    Submit human feedback on an analysis run.

    Feedback is attached to the LangSmith run and
    visible in the evaluation dashboard.

    [WHY] Human feedback is gold for RAG evaluation.
    It trains you to improve retrieval + prompts over time.
    """
    if not settings.langchain_api_key:
        return {"status": "langsmith_disabled"}

    try:
        client = LangSmithClient(api_key=settings.langchain_api_key)
        client.create_feedback(
            run_id=request.run_id,
            key="user_feedback",
            score=request.score,
            comment=request.comment,
        )
        return {"status": "feedback_recorded"}
    except Exception as exc:
        logger.error(
            "Feedback submission failed",
            extra={"error": str(exc)},
        )
        raise HTTPException(
            status_code=500,
            detail="Feedback submission failed",
        )    
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