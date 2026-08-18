"""Inspect the existing end-to-end ablation backtest template."""

from __future__ import annotations

import ast

from quant_equity.config import PROJECT_ROOT, REPORTS_DIR

SOURCE_PATH = PROJECT_ROOT / "scripts" / "run_ensemble_component_ablation.py"

REPORT_PATH = (
    REPORTS_DIR / "robustness" / "feature_family_ablation" / "economic_ablation_template_audit.txt"
)

TARGET_FUNCTIONS = (
    "_portfolio_config",
    "_build_portfolios",
    "_prepare_backtest_targets",
    "_strategy_performance",
    "_execution_metrics",
    "main",
)

RELEVANT_CALLS = (
    "BaselinePortfolioConfig",
    "ExecutionCostConfig",
    "MVPBacktestConfig",
    "PerformanceEvaluationConfig",
    "build_score_weighted_portfolios",
    "compute_portfolio_diagnostics",
    "run_mvp_backtest",
    "build_buy_and_hold_benchmark",
    "evaluate_performance",
)


def _function_source(
    source: str,
    tree: ast.Module,
    name: str,
) -> str:
    """Return exact source for a named top-level function."""
    for node in tree.body:
        if (
            isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            )
            and node.name == name
        ):
            segment = ast.get_source_segment(
                source,
                node,
            )
            return segment or "<source unavailable>"

    return "<not found>"


def _important_assignments(
    source: str,
    tree: ast.Module,
) -> list[str]:
    """Return path and configuration-related module assignments."""
    results: list[str] = []

    for node in tree.body:
        if not isinstance(
            node,
            (ast.Assign, ast.AnnAssign),
        ):
            continue

        targets = node.targets if isinstance(node, ast.Assign) else [node.target]

        names = {
            child.id
            for target in targets
            for child in ast.walk(target)
            if isinstance(child, ast.Name)
        }

        if not names:
            continue

        if any(
            token in name.upper()
            for name in names
            for token in (
                "PATH",
                "DIR",
                "CAPITAL",
                "COST",
                "BPS",
                "SCENARIO",
                "METHOD",
            )
        ):
            segment = ast.get_source_segment(
                source,
                node,
            )
            if segment:
                results.append(segment)

    return results


def _relevant_call_segments(
    source: str,
    tree: ast.Module,
) -> list[str]:
    """Return exact source segments for key construction/backtest calls."""
    results: list[str] = []

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

        if name not in RELEVANT_CALLS:
            continue

        segment = ast.get_source_segment(
            source,
            node,
        )

        if segment and segment not in results:
            results.append(segment)

    return results


def main() -> None:
    """Print the compact contract and persist exact source details."""
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Template script not found: {SOURCE_PATH}")

    source = SOURCE_PATH.read_text(encoding="utf-8")

    tree = ast.parse(
        source,
        filename=str(SOURCE_PATH),
    )

    assignments = _important_assignments(
        source,
        tree,
    )

    calls = _relevant_call_segments(
        source,
        tree,
    )

    terminal_lines = [
        "Institutional Quant Equity Research Platform",
        "Economic ablation template audit",
        "================================================",
        f"source: {SOURCE_PATH}",
        "",
        "Important assignments",
        "---------------------",
        *(assignments if assignments else ["<none found>"]),
        "",
        "Relevant calls",
        "--------------",
        *(calls if calls else ["<none found>"]),
        "",
        "Target helper availability",
        "--------------------------",
    ]

    report_lines = list(terminal_lines)

    for name in TARGET_FUNCTIONS:
        function_source = _function_source(
            source,
            tree,
            name,
        )

        status = (
            "FOUND"
            if function_source
            not in (
                "<not found>",
                "<source unavailable>",
            )
            else function_source.upper().strip("<>")
        )

        terminal_lines.append(f"{name}: {status}")

        report_lines.extend(
            [
                "",
                "=" * 72,
                f"{name}: {status}",
                "=" * 72,
                function_source,
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
