"""Audit the exact predictive-evaluation contract used by the frozen full ensemble."""

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
    / "predictive_evaluation_contract_audit.txt"
)

SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_final_alpha_signal.py"

MODULE_CANDIDATES = (
    "quant_equity.models",
    "quant_equity.models.evaluation",
    "quant_equity.models.model_evaluation",
)


def _safe_signature(obj: Any) -> str:
    """Return a readable signature."""
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "<signature unavailable>"


def _safe_source(obj: Any) -> str:
    """Return source code when available."""
    try:
        return inspect.getsource(obj).rstrip()
    except (OSError, TypeError):
        return "<source unavailable>"


def _find_evaluator() -> tuple[Any, Any]:
    """Locate evaluate_model_predictions and its defining module."""
    for module_name in MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        evaluator = getattr(
            module,
            "evaluate_model_predictions",
            None,
        )

        if evaluator is not None:
            defining_module = importlib.import_module(evaluator.__module__)
            return evaluator, defining_module

    raise RuntimeError("evaluate_model_predictions could not be located.")


def _called_local_helpers(
    source: str,
    module: Any,
) -> list[tuple[str, Any]]:
    """Return locally defined helpers called by the evaluator."""
    tree = ast.parse(source)
    names: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Name):
            name = node.func.id
        else:
            continue

        if name in names:
            continue

        obj = getattr(module, name, None)

        if obj is None:
            continue

        if not inspect.isfunction(obj):
            continue

        if getattr(obj, "__module__", "") != module.__name__:
            continue

        names.append(name)

    return [(name, getattr(module, name)) for name in names]


def _extract_script_calls(path: Path) -> list[str]:
    """Extract exact source segments that call the evaluator."""
    if not path.exists():
        return [f"Script not found: {path}"]

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(
        source,
        filename=str(path),
    )
    lines = source.splitlines()

    segments: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue

        if name != "evaluate_model_predictions":
            continue

        start = max(
            0,
            node.lineno - 5,
        )
        end = min(
            len(lines),
            getattr(node, "end_lineno", node.lineno) + 4,
        )

        numbered = "\n".join(f"{index + 1:04d}: {lines[index]}" for index in range(start, end))

        segments.append(numbered)

    if not segments:
        return ["No evaluate_model_predictions call found."]

    return segments


def main() -> None:
    """Write a full audit and keep terminal output concise."""
    evaluator, defining_module = _find_evaluator()

    signature = _safe_signature(evaluator)
    source = _safe_source(evaluator)

    helpers = (
        _called_local_helpers(
            source,
            defining_module,
        )
        if source != "<source unavailable>"
        else []
    )

    script_calls = _extract_script_calls(SCRIPT_PATH)

    terminal_lines = [
        "Institutional Quant Equity Research Platform",
        "Predictive evaluation contract audit",
        "------------------------------------------------",
        (f"evaluator: {evaluator.__module__}.{evaluator.__name__}{signature}"),
        (f"defining_file: {getattr(defining_module, '__file__', '<unknown>')}"),
        "",
        "Local helpers called by evaluator:",
    ]

    if helpers:
        terminal_lines.extend(f"- {name}{_safe_signature(obj)}" for name, obj in helpers)
    else:
        terminal_lines.append("- none detected")

    terminal_lines.extend(
        [
            "",
            "Exact evaluator call(s) in build_final_alpha_signal.py:",
        ]
    )

    terminal_lines.extend(script_calls)

    report_lines = [
        *terminal_lines,
        "",
        "=" * 72,
        "Full evaluator source",
        "=" * 72,
        source,
        "",
    ]

    for name, obj in helpers:
        report_lines.extend(
            [
                "=" * 72,
                f"Helper: {name}{_safe_signature(obj)}",
                "=" * 72,
                _safe_source(obj),
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
