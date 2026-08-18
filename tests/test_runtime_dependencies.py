"""Runtime dependency and packaging contract tests."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import cvxpy as cp
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"


@pytest.mark.parametrize(
    "module_name",
    [
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "sklearn",
        "lightgbm",
        "cvxpy",
        "yfinance",
        "matplotlib",
        "yaml",
        "streamlit",
        "plotly",
    ],
)
def test_declared_runtime_dependency_is_importable(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_core_quantitative_dependencies_are_declared_directly() -> None:
    config = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    dependencies = config["project"]["dependencies"]

    normalized_names = {
        dependency.split(";", maxsplit=1)[0]
        .split("[", maxsplit=1)[0]
        .split("=", maxsplit=1)[0]
        .split("<", maxsplit=1)[0]
        .split(">", maxsplit=1)[0]
        .strip()
        .lower()
        for dependency in dependencies
    }

    assert {"scipy", "lightgbm", "cvxpy"}.issubset(normalized_names)


def test_cvxpy_has_an_available_solver() -> None:
    assert cp.installed_solvers()
