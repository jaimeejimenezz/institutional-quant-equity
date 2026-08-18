"""Inspect exact constants and helper signatures from the economic ablation template."""

from __future__ import annotations

import ast
import importlib.util
import inspect
from types import ModuleType
from typing import Any

from quant_equity.config import PROJECT_ROOT, REPORTS_DIR

SOURCE_PATH = PROJECT_ROOT / "scripts" / "run_ensemble_component_ablation.py"

REPORT_PATH = (
    REPORTS_DIR
    / "robustness"
    / "feature_family_ablation"
    / "economic_ablation_runtime_contract.txt"
)

HELPERS = (
    "_portfolio_config",
    "_build_portfolios",
    "_strategy_performance",
)

CONSTANT_NAMES = (
    "CANDIDATE_COUNT",
    "MAX_SECURITY_WEIGHT",
    "MAX_SECTOR_WEIGHT",
    "MINIMUM_POSITIONS",
    "WEIGHT_TOLERANCE",
    "BASELINE_SCENARIO",
    "EXPECTED_ABLATIONS",
    "EXPECTED_SCENARIOS",
)


def _load_module() -> ModuleType:
    """Import the template script without executing main()."""
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Template script not found: {SOURCE_PATH}")

    spec = importlib.util.spec_from_file_location(
        "economic_ablation_template",
        SOURCE_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import the economic ablation template.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


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


def _main_config_assignments() -> list[str]:
    """Extract config-related assignments from main()."""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(
        source,
        filename=str(SOURCE_PATH),
    )

    main_node = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"),
        None,
    )

    if main_node is None:
        return ["<main not found>"]

    results: list[str] = []

    for node in ast.walk(main_node):
        if not isinstance(
            node,
            (ast.Assign, ast.AnnAssign),
        ):
            continue

        segment = ast.get_source_segment(
            source,
            node,
        )

        if not segment:
            continue

        lowered = segment.lower()

        if any(
            token in lowered
            for token in (
                "executioncostconfig",
                "mvpbacktestconfig",
                "performanceevaluationconfig",
                "transaction_cost",
                "linear_cost",
                "initial_capital",
                "execution_config",
                "net_config",
                "evaluation_config",
            )
        ):
            if segment not in results:
                results.append(segment)

    return results if results else ["<no config assignments detected>"]


def main() -> None:
    """Print the exact runtime contract needed for feature-family backtests."""
    module = _load_module()

    terminal_lines = [
        "Institutional Quant Equity Research Platform",
        "Economic ablation runtime contract",
        "================================================",
        f"source: {SOURCE_PATH}",
        "",
        "Frozen construction constants",
        "-----------------------------",
    ]

    report_lines = list(terminal_lines)

    for name in CONSTANT_NAMES:
        if hasattr(module, name):
            value = getattr(
                module,
                name,
            )
            line = f"{name} = {value!r}"
        else:
            line = f"{name}: NOT FOUND"

        terminal_lines.append(line)
        report_lines.append(line)

    terminal_lines.extend(
        [
            "",
            "Reusable helper signatures",
            "--------------------------",
        ]
    )
    report_lines.extend(
        [
            "",
            "Reusable helper signatures",
            "--------------------------",
        ]
    )

    for name in HELPERS:
        obj = getattr(
            module,
            name,
            None,
        )

        if obj is None:
            line = f"{name}: NOT FOUND"
            terminal_lines.append(line)
            report_lines.append(line)
            continue

        line = f"{name}{_signature(obj)}"

        terminal_lines.append(line)

        report_lines.extend(
            [
                line,
                "source:",
                _source(obj),
                "",
            ]
        )

    config_assignments = _main_config_assignments()

    terminal_lines.extend(
        [
            "",
            "Backtest/evaluation config assignments inside main()",
            "-----------------------------------------------",
            *config_assignments,
        ]
    )

    report_lines.extend(
        [
            "",
            "Backtest/evaluation config assignments inside main()",
            "-----------------------------------------------",
            *config_assignments,
            "",
        ]
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n".join(terminal_lines))

    print()
    print(f"Full audit report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
