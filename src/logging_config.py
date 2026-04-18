"""Configuracion de logging con formato estructurado simple."""

from __future__ import annotations

import logging
import sys

from src.config import settings


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    root.setLevel(level)
    root.addHandler(handler)

    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
