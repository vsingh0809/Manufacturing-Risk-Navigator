"""Unit tests for observability components."""

import pytest
import asyncio

from app.core.observability import OperationTimer, TokenTracker
from app.models.observability import TokenUsage, LatencyRecord


# ── OperationTimer ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_operation_timer_records_latency():
    async with OperationTimer("test_op") as timer:
        await asyncio.sleep(0.01)

    assert timer.record is not None
    assert timer.record.operation == "test_op"
    assert timer.record.latency_ms >= 10.0


@pytest.mark.asyncio
async def test_operation_timer_returns_latency_record():
    async with OperationTimer("embed") as timer:
        pass

    assert isinstance(timer.record, LatencyRecord)


# ── TokenTracker ───────────────────────────────────────────────────────────────

def test_token_tracker_accumulates_correctly():
    tracker = TokenTracker(operation="risk_analysis")
    tracker.record(prompt_tokens=100, completion_tokens=50)
    tracker.record(prompt_tokens=80, completion_tokens=40)

    total = tracker.total
    assert total.prompt_tokens == 180
    assert total.completion_tokens == 90
    assert total.total_tokens == 270


def test_token_tracker_starts_at_zero():
    tracker = TokenTracker(operation="test")
    total = tracker.total
    assert total.prompt_tokens == 0
    assert total.completion_tokens == 0
    assert total.total_tokens == 0


def test_token_tracker_returns_token_usage_model():
    tracker = TokenTracker(operation="test")
    tracker.record(10, 5)
    assert isinstance(tracker.total, TokenUsage)


def test_token_tracker_single_record():
    tracker = TokenTracker(operation="summarise")
    tracker.record(prompt_tokens=200, completion_tokens=75)

    assert tracker.total.total_tokens == 275