"""
SCIO Logging System

Strukturiertes Logging mit Unterstützung für:
- JSON-formatierte Logs (für Maschinenverarbeitung)
- Human-readable Logs (für Entwicklung)
- Automatische Kontextanreicherung
- Log-Rotation und -Archivierung
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
from structlog.types import Processor

from scio.core.config import get_config


def add_timestamp(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Fügt ISO-8601 Timestamp hinzu."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_scio_context(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Fügt SCIO-spezifischen Kontext hinzu."""
    event_dict.setdefault("framework", "scio")
    return event_dict


def setup_logging(
    level: Optional[str] = None,
    format_type: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> None:
    """
    Initialisiert das Logging-System.

    Args:
        level: Log-Level (DEBUG, INFO, WARNING, ERROR)
        format_type: 'structured' (JSON) oder 'human' (lesbar)
        log_file: Optionale Log-Datei
    """
    config = get_config()

    level = level or config.logging.level
    format_type = format_type or config.logging.format
    log_file = log_file or config.logging.file

    # Bestimme Renderer basierend auf Format
    if format_type == "structured":
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=sys.stdout.isatty(),
            exception_formatter=structlog.dev.plain_traceback,
        )

    # Prozessoren-Pipeline
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        add_timestamp,
        add_scio_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        renderer,
    ]

    # Konfiguriere structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Standard-Logging auch konfigurieren (für Dependencies)
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: Optional[str] = None, **initial_context: Any) -> structlog.BoundLogger:
    """
    Erstellt einen neuen Logger mit optionalem Kontext.

    Args:
        name: Logger-Name (normalerweise __name__)
        **initial_context: Initiale Kontext-Variablen

    Returns:
        Konfigurierter structlog Logger

    Example:
        logger = get_logger(__name__, experiment_id="exp-001")
        logger.info("Starting experiment", step=1)
    """
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


class LogContext:
    """
    Context Manager für temporären Log-Kontext.

    Example:
        with LogContext(request_id="abc-123"):
            logger.info("Processing request")  # Enthält request_id
    """

    def __init__(self, **context: Any):
        self.context = context
        self._token = None

    def __enter__(self) -> "LogContext":
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token:
            structlog.contextvars.unbind_contextvars(*self.context.keys())
