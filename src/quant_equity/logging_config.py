"""Centralized logging configuration."""

from __future__ import annotations

import logging
from pathlib import Path

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure console and optional file logging.

    Parameters
    ----------
    level:
        Logging level, such as DEBUG, INFO, WARNING or ERROR.
    log_file:
        Optional path where logs should also be written.

    Returns
    -------
    logging.Logger
        Logger for the quant_equity package.
    """
    normalized_level = level.upper()
    numeric_level = getattr(logging, normalized_level, None)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level}")

    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
    ]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        handlers.append(
            logging.FileHandler(
                filename=log_path,
                encoding="utf-8",
            )
        )

    logging.basicConfig(
        level=numeric_level,
        format=DEFAULT_LOG_FORMAT,
        handlers=handlers,
        force=True,
    )

    return logging.getLogger("quant_equity")
