"""Central logging setup for the RAG application.

Usage in any module:
    from src.utils.logging import get_logger
    logger = get_logger(__name__)

Environment variables (all optional):
    LOG_LEVEL   DEBUG | INFO | WARNING | ERROR   (default: INFO)
    LOG_FORMAT  text | json                      (default: text)
    LOG_DIR     folder for rotating log files    (default: logs; empty = console only)
"""
import contextvars
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler

# Holds the id of the request currently being handled, so every log line
# written while serving that request carries the same id.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)

TEXT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(request_id)s | %(name)s | %(message)s"
)

_configured = False


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
    """Configure the root logger once. Safe to call multiple times."""
    global _configured
    if _configured:
        return

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "text").lower()
    log_dir = os.getenv("LOG_DIR", "logs")

    formatter: logging.Formatter = (
        JsonFormatter() if log_format == "json" else logging.Formatter(TEXT_FORMAT)
    )

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                os.path.join(log_dir, "app.log"),
                maxBytes=5_000_000,
                backupCount=3,
                encoding="utf-8",
            )
        )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(RequestIdFilter())
        root.addHandler(handler)

    # Quiet noisy third-party loggers; our middleware logs requests itself.
    for noisy in ("httpx", "httpcore", "urllib3", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


@contextmanager
def log_timing(logger: logging.Logger, step: str, **fields):
    """Log how long a block takes, and log the exception if it fails.

        with log_timing(logger, "qdrant_query", limit=20):
            results = client.query_points(...)
    """
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        logger.exception("%s failed after %.0f ms %s", step, elapsed, extra)
        raise
    else:
        elapsed = (time.perf_counter() - start) * 1000
        logger.info("%s done in %.0f ms %s", step, elapsed, extra)