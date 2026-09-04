"""Logging utilities for the application."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured application logger."""

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    return logger

