"""
Observability layer.

Covers:
  1. Prometheus metrics (request count, latency, token usage)
  2. RequestTracingMiddleware (per-request trace + latency)
  3. OperationTimer (context manager for timing any operation)
  4. TokenTracker (accumulates token usage across LLM calls)
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable

from opentelemetry import trace
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.models.observability import LatencyRecord, TokenUsage

logger = logging.getLogger(__name__)

# ── Prometheus Metrics ─────────────────────────────────────────────────────────
# [WHY] Module-level metric objects — Prometheus requires single
# registration. Creating per-request raises DuplicateMetric error.

REQUEST_COUNT = Counter(
    "mrn_request_total",
    "Total HTTP requests received",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "mrn_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

TOKEN_USAGE_COUNTER = Counter(
    "mrn_token_usage_total",
    "Total LLM tokens consumed",
    ["operation", "token_type"],
)

RETRIEVAL_LATENCY = Histogram(
    "mrn_retrieval_latency_seconds",
    "Retrieval pipeline latency in seconds",
    ["operation"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0],
)

ACTIVE_REQUESTS = Gauge(
    "mrn_active_requests",
    "Number of requests currently being processed",
)

ERROR_COUNT = Counter(
    "mrn_error_total",
    "Total errors by type",
    ["error_type", "endpoint"],
)


# ── Request Tracing Middleware ─────────────────────────────────────────────────

class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that traces every HTTP request.

    Per request:
    - Generates request_id
    - Creates OpenTelemetry span
    - Records Prometheus latency + request count
    - Logs structured request summary
    - Tracks active request count
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Wrap each request in a trace span and record metrics.

        Args:
            request:   Incoming Starlette request.
            call_next: Next middleware or route handler.

        Returns:
            Response from downstream handler.
        """
        request_id = str(uuid.uuid4())
        # [WHY] Store request_id in state — route handlers can
        # read it for logging correlation without re-generating.
        request.state.request_id = request_id

        endpoint = request.url.path
        method = request.method

        ACTIVE_REQUESTS.inc()
        start = time.monotonic()

        try:
            tracer = trace.get_tracer("manufacturing-risk-navigator")
        except Exception:
            tracer = None

        span_context = (
            tracer.start_as_current_span(
                f"{method} {endpoint}",
                attributes={
                    "http.method": method,
                    "http.url": str(request.url),
                    "request.id": request_id,
                },
            )
            if tracer
            else _noop_span_context()
        )

        with span_context:
            try:
                response = await call_next(request)
                status_code = response.status_code
            except Exception as exc:
                status_code = 500
                ERROR_COUNT.labels(
                    error_type=type(exc).__name__,
                    endpoint=endpoint,
                ).inc()
                raise
            finally:
                latency = time.monotonic() - start

                REQUEST_LATENCY.labels(
                    method=method,
                    endpoint=endpoint,
                ).observe(latency)

                REQUEST_COUNT.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=str(status_code),
                ).inc()

                ACTIVE_REQUESTS.dec()

                logger.info(
                    "Request completed",
                    extra={
                        "request_id": request_id,
                        "method": method,
                        "endpoint": endpoint,
                        "status_code": status_code,
                        "latency_ms": round(latency * 1000, 2),
                    },
                )

        return response


# ── Operation Timer ────────────────────────────────────────────────────────────

class OperationTimer:
    """
    Context manager for timing named operations.

    Usage:
        async with OperationTimer("rerank") as timer:
            results = await reranker.rerank(...)
        record = timer.record  # LatencyRecord
    """

    def __init__(self, operation: str) -> None:
        self._operation = operation
        self._start: float = 0.0
        self.record: LatencyRecord | None = None

    async def __aenter__(self) -> "OperationTimer":
        self._start = time.monotonic()
        return self

    async def __aexit__(self, *_) -> None:
        latency_ms = (time.monotonic() - self._start) * 1000

        self.record = LatencyRecord(
            operation=self._operation,
            latency_ms=round(latency_ms, 2),
        )

        # [WHY] Record to Prometheus for dashboarding.
        RETRIEVAL_LATENCY.labels(operation=self._operation).observe(
            latency_ms / 1000
        )

        logger.debug(
            "Operation timed",
            extra={
                "operation": self._operation,
                "latency_ms": round(latency_ms, 2),
            },
        )


# ── Token Tracker ──────────────────────────────────────────────────────────────

class TokenTracker:
    """
    Accumulates token usage across multiple LLM calls.

    Usage:
        tracker = TokenTracker(operation="risk_analysis")
        tracker.record(prompt_tokens=100, completion_tokens=50)
        tracker.record(prompt_tokens=80, completion_tokens=40)
        usage = tracker.total  # TokenUsage(180, 90, 270)
    """

    def __init__(self, operation: str) -> None:
        self._operation = operation
        self._prompt = 0
        self._completion = 0

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        """
        Add token counts from one LLM call.

        Args:
            prompt_tokens:     Tokens in the prompt.
            completion_tokens: Tokens in the completion.
        """
        self._prompt += prompt_tokens
        self._completion += completion_tokens

        # [WHY] Record to Prometheus per call so dashboards
        # show token burn rate in real time.
        TOKEN_USAGE_COUNTER.labels(
            operation=self._operation,
            token_type="prompt",
        ).inc(prompt_tokens)

        TOKEN_USAGE_COUNTER.labels(
            operation=self._operation,
            token_type="completion",
        ).inc(completion_tokens)

        logger.debug(
            "Tokens recorded",
            extra={
                "operation": self._operation,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

    @property
    def total(self) -> TokenUsage:
        """Return accumulated token usage as TokenUsage model."""
        return TokenUsage(
            prompt_tokens=self._prompt,
            completion_tokens=self._completion,
            total_tokens=self._prompt + self._completion,
        )


# ── Prometheus Metrics Endpoint ────────────────────────────────────────────────

def get_prometheus_metrics() -> bytes:
    """
    Return Prometheus metrics in text exposition format.

    Called by GET /metrics endpoint.
    """
    return generate_latest()


# ── Internal Helpers ───────────────────────────────────────────────────────────

from contextlib import contextmanager


@contextmanager
def _noop_span_context():
    """No-op span context when tracer is unavailable."""
    yield