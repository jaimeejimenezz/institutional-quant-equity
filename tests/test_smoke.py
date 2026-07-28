"""Initial smoke tests for the project."""

import logging

import quant_equity
from quant_equity.config import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    get_random_seed,
    load_config,
)
from quant_equity.logging_config import configure_logging


def test_package_can_be_imported() -> None:
    """The main Python package should be importable."""
    assert quant_equity.__version__ == "0.1.0"


def test_project_root_contains_pyproject() -> None:
    """The detected root should contain pyproject.toml."""
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_default_configuration_exists() -> None:
    """The default YAML configuration should exist."""
    assert DEFAULT_CONFIG_PATH.exists()


def test_configuration_can_be_loaded() -> None:
    """The YAML configuration should contain required values."""
    config = load_config()

    assert config["project"]["slug"] == "institutional-quant-equity"
    assert config["storage"]["tabular_format"] == "parquet"
    assert config["research"]["prediction_horizon_sessions"] == 21


def test_random_seed_is_reproducible() -> None:
    """The project should expose a stable global random seed."""
    config = load_config()

    assert get_random_seed(config) == 42


def test_logging_can_be_configured() -> None:
    """The package logger should be created successfully."""
    logger = configure_logging(level="INFO")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "quant_equity"
