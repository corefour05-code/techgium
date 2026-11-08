"""Logging configuration for Infusion Pump ML System."""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "infusion_pump_ml",
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure and return a structured logger.

    Args:
        name: Name of the logger.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_file: Optional path to a file where logs will be written.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

