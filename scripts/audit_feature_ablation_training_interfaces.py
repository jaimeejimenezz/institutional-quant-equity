"""Audit local model-training interfaces before feature-family retraining."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from pathlib import Path
from typing import Any

from quant_equity.config import (
    PROJECT_ROOT,
    REPORTS_DIR,
)

REPORT_PATH = REPORTS_DIR / "robustness" / "feature_ablation_training_interface_audit.txt"

MODULE_CANDIDATES = (
    "quant_equity.models.regularized_linear",
    "quant_equity.models.lightgbm_ranker",
    "quant_equity.models.lightgbm_ranking",
    "quant_equity.models.model_baselines",
    "quant_equity.models",
)

SCRIPT_PATTERNS = (
    "*regularized*.py",
    "*elastic*.py",
    "*ranker*.py",
    "*ranking*.py",
)

NAME_KEYWORDS = (
    "train",
    "fit",
    "predict",
    "evaluate",
    "score_technical_composite",
    "config",
    "output",
)


def _safe_signature(
    obj: Any,
) -> str:
    """Return a readable callable signature."""
    try:
        return str(inspect.signature(obj))
    except (
        TypeError,
        ValueError,
    ):
        return "<signature unavailable>"


def _public_relevant_members(
    module: Any,
) -> list[
    tuple[
        str,
        Any,
    ]
]:
    """Return locally defined functions and classes relevant to training."""
    rows = []

    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue

        if not any(keyword in name.lower() for keyword in NAME_KEYWORDS):
            continue

        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue

        owner = getattr(
            obj,
            "__module__",
            "",
        )

        if owner != module.__name__:
            continue

        rows.append(
            (
                name,
                obj,
            )
        )

    return rows


def _describe_dataclass(
    obj: Any,
) -> list[str]:
    """Describe dataclass constructor fields and defaults."""
    if not (inspect.isclass(obj) and dataclasses.is_dataclass(obj)):
        return []

    lines = [
        "    dataclass fields:",
    ]

    for field in dataclasses.fields(obj):
        if field.default is not dataclasses.MISSING:
            default = repr(field.default)
        elif field.default_factory is not dataclasses.MISSING:
            default = "<default_factory>"
        else:
            default = "<required>"

        lines.append(f"      - {field.name}: default={default}")

    return lines


def _source_for_named_function(
    module: Any,
    function_name: str,
) -> list[str]:
    """Return source for one function when available."""
    obj = getattr(
        module,
        function_name,
        None,
    )

    if obj is None:
        return []

    try:
        source = inspect.getsource(obj)
    except (
        OSError,
        TypeError,
    ):
        return [(f"Source unavailable for {module.__name__}.{function_name}")]

    return [
        (f"Source for {module.__name__}.{function_name}:"),
        source.rstrip(),
    ]


def _script_ast_summary(
    path: Path,
) -> list[str]:
    """Summarize imports and relevant function calls in one training script."""
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text(encoding="utf-8-sig")

    tree = ast.parse(
        source,
        filename=str(path),
    )

    imports = []

    calls = []

    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.ImportFrom,
        ):
            module = node.module or ""

            if module.startswith("quant_equity"):
                imports.append(
                    (
                        module,
                        tuple(alias.name for alias in node.names),
                    )
                )

        if isinstance(
            node,
            ast.Call,
        ):
            name = ""

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

            if name and any(
                keyword in name.lower()
                for keyword in (
                    "train",
                    "fit",
                    "evaluate",
                    "score",
                )
            ):
                keyword_names = tuple(
                    keyword.arg for keyword in node.keywords if keyword.arg is not None
                )

                calls.append(
                    (
                        name,
                        keyword_names,
                    )
                )

    lines = [
        f"Script: {path}",
    ]

    if imports:
        lines.append("  quant_equity imports:")

        for module, names in sorted(set(imports)):
            lines.append(f"    - {module}: " + ", ".join(names))

    else:
        lines.append("  quant_equity imports: none")

    if calls:
        lines.append("  relevant calls:")

        seen = set()

        for name, keywords in calls:
            key = (
                name,
                keywords,
            )

            if key in seen:
                continue

            seen.add(key)

            suffix = "" if not keywords else (" | keyword args: " + ", ".join(keywords))

            lines.append(f"    - {name}{suffix}")

    else:
        lines.append("  relevant calls: none")

    return lines


def main() -> None:
    """Write and display the local feature-ablation training interface audit."""
    lines = [
        ("Institutional Quant Equity Research Platform"),
        ("Feature-ablation training interface audit"),
        ("================================================"),
        "",
    ]

    loaded_modules = {}

    for module_name in MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                lines.extend(
                    [
                        f"Module: {module_name}",
                        "  status: NOT FOUND",
                        "",
                    ]
                )
                continue

            raise

        loaded_modules[module_name] = module

        lines.extend(
            [
                f"Module: {module_name}",
                (f"  file: {getattr(module, '__file__', '<unknown>')}"),
                "  relevant local members:",
            ]
        )

        members = _public_relevant_members(module)

        if not members:
            lines.append("    - none")

        for name, obj in members:
            kind = "class" if inspect.isclass(obj) else "function"

            lines.append(f"    - {kind} {name}{_safe_signature(obj)}")

            lines.extend(_describe_dataclass(obj))

        lines.append("")

    baseline_module = loaded_modules.get("quant_equity.models.model_baselines")

    if baseline_module is not None:
        lines.extend(
            _source_for_named_function(
                baseline_module,
                "score_technical_composite",
            )
        )
        lines.append("")

    scripts_dir = PROJECT_ROOT / "scripts"

    candidate_scripts = set()

    for pattern in SCRIPT_PATTERNS:
        candidate_scripts.update(scripts_dir.glob(pattern))

    lines.extend(
        [
            ("Training scripts discovered"),
            ("---------------------------"),
        ]
    )

    if not candidate_scripts:
        lines.append("No matching training scripts found.")

    for path in sorted(candidate_scripts):
        lines.extend(_script_ast_summary(path))
        lines.append("")

    models_package = loaded_modules.get("quant_equity.models")

    if models_package is not None:
        exports = sorted(
            name
            for name in dir(models_package)
            if not name.startswith("_")
            and any(
                keyword in name.lower()
                for keyword in (
                    "train",
                    "fit",
                    "rank",
                    "elastic",
                    "regular",
                    "composite",
                )
            )
        )

        lines.extend(
            [
                ("Relevant quant_equity.models exports"),
                ("-----------------------------------"),
            ]
        )

        for name in exports:
            obj = getattr(
                models_package,
                name,
            )

            if inspect.isfunction(obj) or inspect.isclass(obj):
                lines.append(f"- {name}{_safe_signature(obj)}")
            else:
                lines.append(f"- {name}")

        lines.append("")

    required_findings = {
        "regularized_linear_module": ("quant_equity.models.regularized_linear" in loaded_modules),
        "ranking_module": any(
            module_name in loaded_modules
            for module_name in (
                "quant_equity.models.lightgbm_ranker",
                "quant_equity.models.lightgbm_ranking",
            )
        ),
        "model_baselines_module": (baseline_module is not None),
        "technical_composite_callable": bool(
            baseline_module is not None
            and callable(
                getattr(
                    baseline_module,
                    "score_technical_composite",
                    None,
                )
            )
        ),
        "training_scripts_found": bool(candidate_scripts),
    }

    lines.extend(
        [
            "Audit checks",
            "------------",
        ]
    )

    failed = 0

    for name, passed in required_findings.items():
        status = "PASS" if passed else "FAIL"

        if not passed:
            failed += 1

        lines.append(f"{name}: {status}")

    lines.extend(
        [
            "",
            (f"readiness_checks: {len(required_findings)}"),
            (f"failed_readiness_checks: {failed}"),
            "",
        ]
    )

    report = "\n".join(lines)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print(report)

    print(f"Audit report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
