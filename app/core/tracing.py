"""
Tracing setup — OpenTelemetry + LangSmith.
"""

import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

from app.core.config import AppSettings

logger = logging.getLogger(__name__)

_tracer: trace.Tracer | None = None


def setup_tracing(settings: AppSettings) -> None:
    """
    Initialise OpenTelemetry + LangSmith tracing.

    LangSmith is enabled by setting env vars before
    any langchain imports resolve their tracing config.
    """
    global _tracer

    # ── LangSmith ──────────────────────────────────────────────
    # [WHY] LangChain reads these env vars at import time.
    # Must be set before any langchain module is used.
    # Setting via os.environ here guarantees correct order.
    if settings.langchain_tracing and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint

        logger.info(
            "LangSmith tracing enabled",
            extra={"project": settings.langchain_project},
        )
    else:
        logger.info("LangSmith tracing disabled")

    # ── OpenTelemetry ──────────────────────────────────────────
    resource = Resource.create(
        {"service.name": "manufacturing-risk-navigator"}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(ConsoleSpanExporter())
    )
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("manufacturing-risk-navigator")

    logger.info("OpenTelemetry tracing initialised")


def get_tracer() -> trace.Tracer:
    if _tracer is None:
        raise RuntimeError("Call setup_tracing() during app startup")
    return _tracer