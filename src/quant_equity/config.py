"""Project configuration and path management."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REFERENCE_DATA_DIR = DATA_DIR / "reference"

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

DEFAULT_CONFIG_PATH = CONFIG_DIR / "project.yaml"


class ConfigurationError(RuntimeError):
    """Raised when the project configuration is missing or invalid."""


@lru_cache(maxsize=4)
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ConfigurationError
        If the file is empty or required sections are missing.
    """
    config_path = Path(path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ConfigurationError(f"The configuration must contain a YAML mapping: {config_path}")

    required_sections = {
        "project",
        "runtime",
        "storage",
        "research",
        "universe",
        "market_data",
        "portfolio",
        "benchmarks",
    }

    missing_sections = required_sections.difference(config)

    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise ConfigurationError(f"Missing required configuration sections: {missing}")

    return config


def get_random_seed(config: dict[str, Any] | None = None) -> int:
    """Return the global random seed configured for the project."""
    project_config = config if config is not None else load_config()

    try:
        return int(project_config["runtime"]["random_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError("runtime.random_seed must be a valid integer.") from error


def project_path(*parts: str) -> Path:
    """Build an absolute path relative to the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def ensure_project_directories() -> None:
    """Create directories that may not yet exist.

    This function is safe to execute multiple times.
    """
    directories = [
        RAW_DATA_DIR / "market",
        RAW_DATA_DIR / "fundamentals",
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        REFERENCE_DATA_DIR,
        REPORTS_DIR / "figures",
        REPORTS_DIR / "tables",
        REPORTS_DIR / "research",
        REPORTS_DIR / "data_quality",
        REPORTS_DIR / "models",
        REPORTS_DIR / "backtests",
        MODELS_DIR,
        LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
