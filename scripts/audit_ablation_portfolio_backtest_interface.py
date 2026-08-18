"""Audit portfolio construction and backtest interfaces for ablation comparison."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any

from quant_equity.config import PROJECT_ROOT, REPORTS_DIR

REPORT_PATH = (
    REPORTS_DIR
    / "robustness"
    / "feature_family_ablation"
    / "portfolio_backtest_interface_audit.txt"
)

SCRIPT_CANDIDATES = (
    PROJECT_ROOT / "scripts" / "build_baseline_portfolios.py",
    PROJECT_ROOT / "scripts" / "build_cvar_portfolios.py",
)

MODULE_CANDIDATES = (
    "quant_equity.portfolio",
    "quant_equity.portfolios",
    "quant_equity.portfolio_construction",
    "quant_equity.backtest",
    "quant_equity.execution",
)

KEYWORDS = (
    "score",
    "weight",
    "portfolio",
    "backtest",
    "turnover",
    "cost",
    "execute",
    "return",
    "spy",
    "benchmark",
)


def _signature(obj: Any) -> str:
    """Return a readable callable signature."""
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "<signature unavailable>"


def _source(obj: Any) -> str:
    """Return source code when available."""
    try:
        return inspect.getsource(obj).rstrip()
    except (OSError, TypeError):
        return "<source unavailable>"


def _interesting(name: str) -> bool:
    """Return whether a symbol is relevant to portfolio/backtest reuse."""
    lowered = name.lower()
    return any(token in lowered for token in KEYWORDS)


def _inspect_module(module_name: str) -> tuple[list[str], list[str]]:
    """Inspect public locally-defined portfolio/backtest members."""
    terminal: list[str] = []
    report: list[str] = []

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return (
            [f"{module_name}: NOT FOUND"],
            [f"{module_name}: NOT FOUND", ""],
        )

    terminal.extend(
        [
            f"Module: {module_name}",
            f"file: {getattr(module, '__file__', '<unknown>')}",
        ]
    )
    report.extend(terminal)

    found = False

    for name in sorted(dir(module)):
        if name.startswith("_") or not _interesting(name):
            continue

        obj = getattr(module, name)

        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue

        if getattr(obj, "__module__", "") != module.__name__:
            continue

        found = True
        line = f"- {name}{_signature(obj)}"
        terminal.append(line)
        report.extend(
            [
                line,
                "source:",
                _source(obj),
                "",
            ]
        )

    if not found:
        terminal.append("- no relevant local callables")
        report.append("- no relevant local callables")

    terminal.append("")
    report.append("")

    return terminal, report


def _script_summary(path: Path) -> tuple[list[str], list[str]]:
    """Summarize exact quant_equity imports and relevant calls."""
    if not path.exists():
        return (
            [f"Script: {path}", "status: NOT FOUND", ""],
            [f"Script: {path}", "status: NOT FOUND", ""],
        )

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports: list[str] = []
    calls: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("quant_equity"):
                imports.append(f"{module}: " + ", ".join(alias.name for alias in node.names))

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue

            if not _interesting(name):
                continue

            keywords = [keyword.arg for keyword in node.keywords if keyword.arg is not None]
            suffix = "" if not keywords else " | keyword args: " + ", ".join(keywords)
            calls.append(f"{name}{suffix}")

    terminal = [
        f"Script: {path}",
        "status: FOUND",
        "quant_equity imports:",
        *([f"- {value}" for value in sorted(set(imports))] if imports else ["- none"]),
        "relevant calls:",
        *([f"- {value}" for value in dict.fromkeys(calls)] if calls else ["- none"]),
        "",
    ]

    report = [
        *terminal,
        "Full script source:",
        source,
        "",
    ]

    return terminal, report


def main() -> None:
    """Write full audit report while keeping terminal output concise."""
    terminal_lines = [
        "Institutional Quant Equity Research Platform",
        "Ablation portfolio/backtest interface audit",
        "================================================",
        "",
    ]
    report_lines = list(terminal_lines)

    for module_name in MODULE_CANDIDATES:
        terminal, report = _inspect_module(module_name)
        terminal_lines.extend(terminal)
        report_lines.extend(report)

    for path in SCRIPT_CANDIDATES:
        terminal, report = _script_summary(path)
        terminal_lines.extend(terminal)
        report_lines.extend(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n".join(terminal_lines))
    print(f"Full audit report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
