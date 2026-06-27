"""
FastAPI application factory.

Handles:
- App lifespan (startup + shutdown)
- Middleware registration
- Router inclusion
- Global exception handlers
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import router as v1_router
from app.core.config import get_settings
from app.core.exceptions import MRNBaseError
from app.core.logging import setup_logging
from app.dependencies import initialise_services, shutdown_services
from app.core.tracing import setup_tracing
from app.core.observability import RequestTracingMiddleware, get_prometheus_metrics
from fastapi.responses import PlainTextResponse
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    App lifespan manager.

    Startup:  initialise all services in correct dependency order.
    Shutdown: gracefully close all connections.

    [WHY] Lifespan over @app.on_event — on_event is deprecated
    in FastAPI 0.93+. Lifespan is the current standard.
    """
    settings = get_settings()
    setup_logging(settings.log_level)
    setup_tracing(service_name="manufacturing-risk-navigator")

    logger.info(
        "Starting Manufacturing Risk Navigator",
        extra={"env": settings.app_env},
    )

    await initialise_services(settings)

    logger.info("Application ready to serve requests")

    yield  # app runs here

    logger.info("Shutting down...")
    await shutdown_services()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="Manufacturing Risk Navigator",
        description="AI-Powered Manufacturing Project Intelligence Workspace",
        version="0.1.0",
        lifespan=lifespan,
        # [WHY] Disable docs in production — avoid exposing API surface.
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url="/redoc" if settings.app_env != "production" else None,
    )

    # ── Middleware ─────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], 
        allow_methods=["*"],
        allow_headers=["*"],
    )

    origins = [
    "http://localhost:3000",  # Your frontend development server
]

# 2. Add the CORS middleware to the application pipeline
    app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Allows requests from localhost:3000
    allow_credentials=True,           # Allows cookies and authorization headers
    allow_methods=["*"],              # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],              # Allows all headers (Content-Type, Authorization, etc.)
)


    app.add_middleware(RequestTracingMiddleware)

    # ── Routers ────────────────────────────────────────────────────────────
    app.include_router(v1_router)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> PlainTextResponse:
      
      """Prometheus metrics scrape endpoint."""
      return PlainTextResponse(
        content=get_prometheus_metrics().decode("utf-8"),
        media_type="text/plain",
    )

    # ── Global Exception Handlers ──────────────────────────────────────────
    @app.exception_handler(MRNBaseError)
    async def domain_exception_handler(
        request: Request,
        exc: MRNBaseError,
    ) -> JSONResponse:
        """
        Convert domain exceptions to structured JSON error responses.

        [WHY] Prevents raw Python exceptions leaking to API consumers.
        All MRNBaseError subclasses get consistent error shape.
        """
        logger.error(
            "Domain exception",
            extra={
                "path": request.url.path,
                "message": exc.message,
                "detail": exc.detail,
            },
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "detail": exc.detail,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Catch-all for unhandled exceptions."""
        logger.error(
            "Unhandled exception",
            extra={"path": request.url.path, "error": str(exc)},
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
            },
        )

    return app


# [WHY] Module-level app instance — uvicorn imports this directly.
# create_app() is a factory for testability — tests call it directly.
app = create_app()