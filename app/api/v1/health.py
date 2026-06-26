"""
Health check endpoints.

GET /health        → liveness probe
GET /health/ready  → readiness probe (checks Qdrant)
"""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.config import AppSettings
from app.dependencies import get_app_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


@router.get("", response_model=HealthResponse)
async def health_check(
    settings: AppSettings = Depends(get_app_settings),
) -> HealthResponse:
    """
    Liveness probe — confirms app is running.
    Does not check external dependencies.
    """
    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=settings.app_env,
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness_check(
    settings: AppSettings = Depends(get_app_settings),
) -> HealthResponse:
    """
    Readiness probe — confirms app is ready to serve traffic.
    Sprint 2: add Qdrant ping check here.
    """
    return HealthResponse(
        status="ready",
        version="0.1.0",
        environment=settings.app_env,
    )