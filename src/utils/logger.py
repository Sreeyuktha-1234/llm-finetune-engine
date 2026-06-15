"""Logging utilities for console and file-based logging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "llm_finetune_engine",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create and configure a logger with console + optional file handlers.

    Args:
        name: Logger name.
        level: Logging level (e.g., logging.INFO).
        log_file: Optional file path for file logging.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers when setup_logger is called multiple times.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger by name using the project's logging configuration."""
    return logging.getLogger(name)
