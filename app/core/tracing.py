"""
OpenTelemetry tracing setup.

Provides a tracer that wraps key operations in spans.
Spans are exported to stdout (console exporter) for MVP.
Sprint 2: swap exporter to Jaeger or Azure Monitor.
"""

import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

logger = logging.getLogger(__name__)

# [WHY] Module-level tracer — initialised once, used everywhere.
# Never re-initialise per request.
_tracer: trace.Tracer | None = None


def setup_tracing(service_name: str = "manufacturing-risk-navigator") -> None:
    """
    Initialise OpenTelemetry tracer with console exporter.

    Called once during app lifespan startup.

    Args:
        service_name: Identifies this service in trace output.
    """
    global _tracer

    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)

    # [WHY] ConsoleSpanExporter for MVP — writes spans to stdout.
    # Docker logs capture stdout — spans visible without extra infra.
    # Sprint 2: replace with OTLPSpanExporter pointing to Jaeger.
    provider.add_span_processor(
        BatchSpanProcessor(ConsoleSpanExporter())
    )

    trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer(service_name)

    logger.info(
        "OpenTelemetry tracing initialised",
        extra={"service": service_name},
    )


def get_tracer() -> trace.Tracer:
    """
    Return the initialised tracer.

    Returns:
        OpenTelemetry Tracer instance.

    Raises:
        RuntimeError: If setup_tracing() was not called first.
    """
    if _tracer is None:
        raise RuntimeError(
            "Tracer not initialised. Call setup_tracing() during app startup."
        )
    return _tracer