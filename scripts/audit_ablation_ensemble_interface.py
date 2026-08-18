"""Audit the final ensemble interface before ablation reconstruction."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from pathlib import Path
from typing import Any

from quant_equity.config import PROJECT_ROOT, REPORTS_DIR

REPORT_PATH = REPORTS_DIR / "robustness" / "ablation_ensemble_interface_audit.txt"

ENSEMBLE_MODULE = "quant_equity.models.ensemble"

TARGET_NAMES = (
    "EnsembleConfig",
    "build_validation_weights",
    "build_component_scores",
    "build_ensemble_candidates",
    "build_final_alpha_signal",
    "build_ablation_candidates",
)

SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_final_alpha_signal.py"


def _safe_signature(obj: Any) -> str:
    """Return a readable callable signature."""
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "<signature unavailable>"


def _describe_dataclass(obj: Any) -> list[str]:
    """Return dataclass field defaults."""
    if not (inspect.isclass(obj) and dataclasses.is_dataclass(obj)):
        return []

    lines = []

    for field in dataclasses.fields(obj):
        if field.default is not dataclasses.MISSING:
            default = repr(field.default)
        elif field.default_factory is not dataclasses.MISSING:
            default = "<default_factory>"
        else:
            default = "<required>"

        lines.append(f"    {field.name}: {default}")

    return lines


def _source(obj: Any) -> str:
    """Return source code when available."""
    try:
        return inspect.getsource(obj).rstrip()
    except (OSError, TypeError):
        return "<source unavailable>"


def _script_summary(path: Path) -> list[str]:
    """Return imports and relevant calls from the ensemble script."""
    if not path.exists():
        return [
            f"script: {path}",
            "status: NOT FOUND",
        ]

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    imports: list[str] = []
    calls: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""

            if module.startswith("quant_equity"):
                names = ", ".join(alias.name for alias in node.names)
                imports.append(f"{module}: {names}")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue

            if any(
                token in name.lower()
                for token in (
                    "ensemble",
                    "validation",
                    "component",
                    "alpha",
                    "weight",
                )
            ):
                keywords = [keyword.arg for keyword in node.keywords if keyword.arg is not None]

                suffix = "" if not keywords else " | keyword args: " + ", ".join(keywords)

                calls.append(f"{name}{suffix}")

    lines = [
        f"script: {path}",
        "status: FOUND",
        "quant_equity imports:",
    ]

    if imports:
        lines.extend(f"  - {value}" for value in sorted(set(imports)))
    else:
        lines.append("  - none")

    lines.append("relevant calls:")

    if calls:
        lines.extend(f"  - {value}" for value in dict.fromkeys(calls))
    else:
        lines.append("  - none")

    return lines


def main() -> None:
    """Write a full audit while keeping terminal output concise."""
    module = importlib.import_module(ENSEMBLE_MODULE)

    terminal_lines = [
        "Institutional Quant Equity Research Platform",
        "Ablation ensemble interface audit",
        "------------------------------------------------",
        f"module: {ENSEMBLE_MODULE}",
        (f"file: {getattr(module, '__file__', '<unknown>')}"),
        "",
        "Interfaces",
        "----------",
    ]

    report_lines = [
        "Institutional Quant Equity Research Platform",
        "Ablation ensemble interface audit",
        "=" * 72,
        "",
        f"Module: {ENSEMBLE_MODULE}",
        (f"File: {getattr(module, '__file__', '<unknown>')}"),
        "",
    ]

    found: dict[str, bool] = {}

    for name in TARGET_NAMES:
        obj = getattr(
            module,
            name,
            None,
        )

        found[name] = obj is not None

        if obj is None:
            terminal_lines.append(f"{name}: NOT FOUND")
            report_lines.extend(
                [
                    f"{name}: NOT FOUND",
                    "",
                ]
            )
            continue

        signature = _safe_signature(obj)

        terminal_lines.append(f"{name}{signature}")

        dataclass_lines = _describe_dataclass(obj)

        terminal_lines.extend(dataclass_lines)

        report_lines.extend(
            [
                f"{name}{signature}",
                *dataclass_lines,
                "",
                "source:",
                _source(obj),
                "",
                "-" * 72,
                "",
            ]
        )

    script_lines = _script_summary(SCRIPT_PATH)

    terminal_lines.extend(
        [
            "",
            "Ensemble script",
            "---------------",
            *script_lines,
        ]
    )

    report_lines.extend(
        [
            "Ensemble script",
            "=" * 72,
            *script_lines,
            "",
        ]
    )

    if SCRIPT_PATH.exists():
        report_lines.extend(
            [
                "Full script source:",
                SCRIPT_PATH.read_text(encoding="utf-8"),
                "",
            ]
        )

    checks = {
        "ensemble_module": True,
        "ensemble_config": found.get(
            "EnsembleConfig",
            False,
        ),
        "validation_weight_builder": found.get(
            "build_validation_weights",
            False,
        ),
        "final_alpha_builder": found.get(
            "build_final_alpha_signal",
            False,
        ),
        "ensemble_script": SCRIPT_PATH.exists(),
    }

    terminal_lines.extend(
        [
            "",
            "Audit checks",
            "------------",
        ]
    )

    failed = 0

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"

        if not passed:
            failed += 1

        terminal_lines.append(f"{name}: {status}")

    terminal_lines.extend(
        [
            "",
            f"readiness_checks: {len(checks)}",
            f"failed_readiness_checks: {failed}",
        ]
    )

    report_lines.extend(
        [
            "Audit checks",
            "=" * 72,
            *terminal_lines[terminal_lines.index("Audit checks") + 2 :],
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
