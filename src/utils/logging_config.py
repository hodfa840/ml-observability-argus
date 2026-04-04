"""Structured, coloured logging for the entire system."""
from __future__ import annotations

import logging
import sys
from typing import Optional


_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_COLOURS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
    "RESET": "\033[0m",
}


class _ColouredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        reset = _COLOURS["RESET"]
        record.levelname = f"{colour}{record.levelname}{reset}"
        return super().format(record)


_root_configured = False


def _configure_root(level: str = "INFO") -> None:
    global _root_configured
    if _root_configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColouredFormatter(fmt=_FMT, datefmt=_DATE_FMT))
    logging.root.setLevel(level)
    logging.root.addHandler(handler)
    _root_configured = True


def get_logger(name: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """Return a named logger with coloured output."""
    _configure_root(level)
    return logging.getLogger(name or "self_healing_ml")
