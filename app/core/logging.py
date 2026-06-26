"""
Structured JSON logging setup.

Called once at app startup. All modules use
logging.getLogger(__name__) after this runs.
"""

import logging
import sys


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure root logger with structured JSON-style output.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    # [WHY] Force=True — overrides any logging config that
    # libraries (langchain, qdrant) set before our app starts.
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # [WHY] Silence noisy third-party loggers that pollute output.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("fastembed").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialised", extra={"level": log_level}
    )