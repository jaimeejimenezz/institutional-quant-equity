"""Audit the exact score-weighted portfolio and backtest interfaces."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any

from quant_equity import backtest as backtest_api
from quant_equity import portfolio as portfolio_api
from quant_equity.config import PROJECT_ROOT, REPORTS_DIR

REPORT_PATH = (
    REPORTS_DIR
    / "robustness"
    / "feature_family_ablation"
    / "score_weighted_backtest_contract_audit.txt"
)

BASELINE_SCRIPT = PROJECT_ROOT / "scripts" / "build_baseline_portfolios.py"

KEYWORDS = (
    "backtest",
    "cost",
    "turnover",
    "performance",
    "return",
    "execution",
    "portfolio",
    "benchmark",
    "spy",
)

PORTFOLIO_TARGETS = (
    "BaselinePortfolioConfig",
    "build_score_weighted_portfolios",
    "compute_portfolio_diagnostics",
    "validate_baseline_portfolios",
)


def _signature(obj: Any) -> str:
    """Return a readable signature."""
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "<signature unavailable>"


def _source(obj: Any) -> str:
    """Return source when available."""
    try:
        return inspect.getsource(obj).rstrip()
    except (OSError, TypeError):
        return "<source unavailable>"


def _object_location(obj: Any) -> tuple[str, str]:
    """Return defining module and file for a callable or class."""
    module_name = getattr(
        obj,
        "__module__",
        "<unknown>",
    )

    try:
        module = importlib.import_module(module_name)
        file_path = getattr(
            module,
            "__file__",
            "<unknown>",
        )
    except Exception:
        file_path = "<unknown>"

    return (
        module_name,
        str(file_path),
    )


def _inspect_portfolio_targets() -> tuple[list[str], list[str]]:
    """Inspect the exact portfolio objects used by the baseline script."""
    terminal = [
        "Portfolio construction interfaces",
        "---------------------------------",
    ]
    report = list(terminal)

    for name in PORTFOLIO_TARGETS:
        obj = getattr(
            portfolio_api,
            name,
            None,
        )

        if obj is None:
            terminal.append(f"{name}: NOT FOUND")
            report.extend(
                [
                    f"{name}: NOT FOUND",
                    "",
                ]
            )
            continue

        module_name, file_path = _object_location(obj)

        terminal.extend(
            [
                f"{name}{_signature(obj)}",
                f"  module: {module_name}",
                f"  file: {file_path}",
            ]
        )

        report.extend(
            [
                f"{name}{_signature(obj)}",
                f"module: {module_name}",
                f"file: {file_path}",
                "source:",
                _source(obj),
                "",
                "-" * 72,
                "",
            ]
        )

    terminal.append("")
    report.append("")

    return terminal, report


def _inspect_backtest_exports() -> tuple[list[str], list[str]]:
    """Inspect relevant public exports from quant_equity.backtest."""
    terminal = [
        "Backtest API exports",
        "--------------------",
    ]
    report = list(terminal)

    found = False

    for name in sorted(dir(backtest_api)):
        if name.startswith("_"):
            continue

        if not any(token in name.lower() for token in KEYWORDS):
            continue

        obj = getattr(
            backtest_api,
            name,
        )

        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue

        found = True
        module_name, file_path = _object_location(obj)

        terminal.extend(
            [
                f"{name}{_signature(obj)}",
                f"  module: {module_name}",
                f"  file: {file_path}",
            ]
        )

        report.extend(
            [
                f"{name}{_signature(obj)}",
                f"module: {module_name}",
                f"file: {file_path}",
                "source:",
                _source(obj),
                "",
                "-" * 72,
                "",
            ]
        )

    if not found:
        terminal.append("No relevant public backtest exports found.")
        report.append("No relevant public backtest exports found.")

    terminal.append("")
    report.append("")

    return terminal, report


def _script_calls_and_constants(
    path: Path,
) -> tuple[list[str], list[str]]:
    """Summarize exact construction calls and path constants in a script."""
    terminal = [
        "Baseline portfolio script",
        "-------------------------",
        f"script: {path}",
    ]
    report = list(terminal)

    if not path.exists():
        terminal.append("status: NOT FOUND")
        report.append("status: NOT FOUND")
        return terminal, report

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(
        source,
        filename=str(path),
    )

    assignments: list[str] = []
    call_segments: list[str] = []

    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            continue

        targets = (
            node.targets
            if isinstance(
                node,
                ast.Assign,
            )
            else [node.target]
        )

        names = {
            child.id
            for target in targets
            for child in ast.walk(target)
            if isinstance(
                child,
                ast.Name,
            )
        }

        if any(
            token in name.upper()
            for name in names
            for token in (
                "PATH",
                "SIGNAL",
                "WEIGHT",
                "PRICE",
                "RETURN",
            )
        ):
            segment = ast.get_source_segment(
                source,
                node,
            )

            if segment:
                assignments.append(segment)

    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if isinstance(
            node.func,
            ast.Name,
        ):
            name = node.func.id
        elif isinstance(
            node.func,
            ast.Attribute,
        ):
            name = node.func.attr
        else:
            continue

        if name not in (
            "build_score_weighted_portfolios",
            "compute_portfolio_diagnostics",
            "validate_baseline_portfolios",
            "calculate_portfolio_risk",
        ):
            continue

        segment = ast.get_source_segment(
            source,
            node,
        )

        if segment:
            call_segments.append(segment)

    terminal.append("important assignments:")

    if assignments:
        terminal.extend(f"- {value}" for value in assignments)
    else:
        terminal.append("- none detected")

    terminal.append("exact relevant calls:")

    if call_segments:
        terminal.extend(f"- {value}" for value in call_segments)
    else:
        terminal.append("- none detected")

    report.extend(
        [
            "",
            "Important assignments:",
            *assignments,
            "",
            "Exact relevant calls:",
            *call_segments,
            "",
            "Full script source:",
            source,
            "",
        ]
    )

    terminal.append("")
    return terminal, report


def _scan_backtest_scripts() -> tuple[list[str], list[str]]:
    """Find project scripts that import or invoke the backtest API."""
    terminal = [
        "Backtest/execution scripts discovered",
        "-------------------------------------",
    ]
    report = list(terminal)

    scripts_dir = PROJECT_ROOT / "scripts"

    hits: list[
        tuple[
            Path,
            list[str],
            list[str],
        ]
    ] = []

    for path in sorted(scripts_dir.glob("*.py")):
        source = path.read_text(encoding="utf-8")

        if not any(token in source.lower() for token in KEYWORDS):
            continue

        tree = ast.parse(
            source,
            filename=str(path),
        )

        imports: list[str] = []
        calls: list[str] = []

        for node in ast.walk(tree):
            if isinstance(
                node,
                ast.ImportFrom,
            ):
                module = node.module or ""

                if module.startswith("quant_equity.backtest"):
                    imports.append(f"{module}: " + ", ".join(alias.name for alias in node.names))

            if isinstance(
                node,
                ast.Call,
            ):
                if isinstance(
                    node.func,
                    ast.Name,
                ):
                    name = node.func.id
                elif isinstance(
                    node.func,
                    ast.Attribute,
                ):
                    name = node.func.attr
                else:
                    continue

                if any(token in name.lower() for token in KEYWORDS):
                    calls.append(name)

        filename_hit = any(
            token in path.name.lower()
            for token in (
                "backtest",
                "execution",
                "performance",
                "cost",
            )
        )

        if imports or filename_hit:
            hits.append(
                (
                    path,
                    sorted(set(imports)),
                    list(dict.fromkeys(calls)),
                )
            )

    if not hits:
        terminal.append("No matching scripts found.")
        report.append("No matching scripts found.")
        return terminal, report

    for path, imports, calls in hits:
        terminal.append(f"Script: {path}")

        if imports:
            terminal.extend(f"  import: {value}" for value in imports)

        if calls:
            terminal.append("  calls: " + ", ".join(calls[:20]))

        source = path.read_text(encoding="utf-8")

        report.extend(
            [
                f"Script: {path}",
                "imports:",
                *(imports if imports else ["none"]),
                "calls:",
                *(calls if calls else ["none"]),
                "Full script source:",
                source,
                "",
                "-" * 72,
                "",
            ]
        )

    terminal.append("")
    return terminal, report


def main() -> None:
    """Write the exact portfolio/backtest contract audit."""
    terminal_lines = [
        "Institutional Quant Equity Research Platform",
        "Score-weighted portfolio and backtest contract audit",
        "====================================================",
        "",
    ]
    report_lines = list(terminal_lines)

    for builder in (
        _inspect_portfolio_targets,
        _inspect_backtest_exports,
        lambda: _script_calls_and_constants(BASELINE_SCRIPT),
        _scan_backtest_scripts,
    ):
        terminal, report = builder()
        terminal_lines.extend(terminal)
        report_lines.extend(report)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n".join(terminal_lines))

    print(f"Full audit report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
