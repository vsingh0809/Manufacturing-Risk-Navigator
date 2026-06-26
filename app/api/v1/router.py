"""
V1 API router aggregator.

Imports and includes all v1 route modules.
Add new route modules here — zero changes to main.py.
"""

from fastapi import APIRouter

from app.api.v1 import analysis, health, ingestion, search, timeline

router = APIRouter(prefix="/api/v1")

router.include_router(health.router)
router.include_router(ingestion.router)
router.include_router(search.router)
router.include_router(analysis.router)
router.include_router(timeline.router)