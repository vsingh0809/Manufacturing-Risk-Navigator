"""Unit tests for tracing setup."""

import pytest
from unittest.mock import patch


def test_setup_tracing_initialises_tracer():
    from app.core.tracing import setup_tracing, get_tracer
    setup_tracing(service_name="test-service")
    tracer = get_tracer()
    assert tracer is not None


def test_get_tracer_raises_before_setup():
    import app.core.tracing as tracing_module
    original = tracing_module._tracer
    tracing_module._tracer = None

    with pytest.raises(RuntimeError, match="not initialised"):
        tracing_module.get_tracer()

    # restore
    tracing_module._tracer = original