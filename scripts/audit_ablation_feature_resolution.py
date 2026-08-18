"""Inspect how model trainers resolve feature columns for ablation retraining."""

from __future__ import annotations

import ast
import importlib
import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from quant_equity.config import REPORTS_DIR

REPORT_PATH = REPORTS_DIR / "robustness" / "ablation_feature_resolution_audit.txt"

TARGETS = (
    (
        "quant_equity.models.regularized_linear",
        "train_regularized_linear_models",
    ),
    (
        "quant_equity.models.lightgbm_ranking",
        "train_lightgbm_ranking",
    ),
    (
        "quant_equity.models.model_baselines",
        "generate_baseline_predictions",
    ),
    (
        "quant_equity.models.model_baselines",
        "score_technical_composite",
    ),
)

IMPORTANT_EXPORTS = (
    "COMPOSITE_FEATURE_DIRECTIONS",
    "TECHNICAL_COMPOSITE_DIRECTIONS",
)


def _safe_repr(
    value: Any,
    *,
    max_length: int = 12000,
) -> str:
    """Return a bounded representation suitable for an audit report."""
    text = repr(value)

    if len(text) <= max_length:
        return text

    return text[:max_length] + "\n... <representation truncated> ..."


def _source(
    obj: Any,
) -> str:
    """Return Python source when inspect can recover it."""
    try:
        return inspect.getsource(obj).rstrip()
    except (
        OSError,
        TypeError,
    ):
        return "<source unavailable>"


def _referenced_names(
    obj: Any,
) -> tuple[str, ...]:
    """Return global-style names referenced by a function body."""
    try:
        source = inspect.getsource(obj)
    except (
        OSError,
        TypeError,
    ):
        return ()

    tree = ast.parse(source)

    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.Name,
        )
        and (node.id.isupper() or "feature" in node.id.lower() or "column" in node.id.lower())
    }

    return tuple(sorted(names))


def _describe_value(
    value: Any,
) -> list[str]:
    """Describe a referenced global value without mutating it."""
    lines = [
        (f"      type: {type(value).__name__}"),
    ]

    if isinstance(
        value,
        Mapping,
    ):
        lines.append(f"      length: {len(value)}")
        lines.append(f"      value: {_safe_repr(dict(value))}")
        return lines

    if isinstance(
        value,
        Sequence,
    ) and not isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        lines.append(f"      length: {len(value)}")
        lines.append(f"      value: {_safe_repr(tuple(value))}")
        return lines

    lines.append(f"      value: {_safe_repr(value)}")

    return lines


def _inspect_target(
    module_name: str,
    object_name: str,
) -> list[str]:
    """Inspect one training or baseline callable."""
    module = importlib.import_module(module_name)

    obj = getattr(
        module,
        object_name,
        None,
    )

    lines = [
        (f"{module_name}.{object_name}"),
        "-" * 72,
    ]

    if obj is None:
        lines.extend(
            [
                "status: NOT FOUND",
                "",
            ]
        )
        return lines

    lines.extend(
        [
            "status: FOUND",
            (f"signature: {inspect.signature(obj)}"),
            "",
            "source:",
            _source(obj),
            "",
            "referenced feature/global names:",
        ]
    )

    names = _referenced_names(obj)

    if not names:
        lines.append("  - none detected")

    for name in names:
        lines.append(f"  - {name}")

        if hasattr(
            module,
            name,
        ):
            value = getattr(
                module,
                name,
            )

            lines.extend(_describe_value(value))
        else:
            lines.append("      module value: not defined locally")

    lines.append("")

    return lines


def main() -> None:
    """Write the feature-resolution audit used before ablation retraining."""
    lines = [
        "Institutional Quant Equity Research Platform",
        "Ablation feature-resolution audit",
        "=" * 72,
        "",
    ]

    for module_name, object_name in TARGETS:
        lines.extend(
            _inspect_target(
                module_name,
                object_name,
            )
        )

    models_module = importlib.import_module("quant_equity.models")

    lines.extend(
        [
            "Frozen technical-composite exports",
            "-" * 72,
        ]
    )

    for name in IMPORTANT_EXPORTS:
        lines.append(f"{name}:")

        if not hasattr(
            models_module,
            name,
        ):
            lines.append("  NOT FOUND")
            continue

        value = getattr(
            models_module,
            name,
        )

        lines.extend(_describe_value(value))

    lines.append("")

    checks = {
        "regularized_linear_source_available": (
            "<source unavailable>"
            not in _source(
                importlib.import_module(
                    "quant_equity.models.regularized_linear"
                ).train_regularized_linear_models
            )
        ),
        "lightgbm_ranking_source_available": (
            "<source unavailable>"
            not in _source(
                importlib.import_module(
                    "quant_equity.models.lightgbm_ranking"
                ).train_lightgbm_ranking
            )
        ),
        "baseline_generation_source_available": (
            "<source unavailable>"
            not in _source(
                importlib.import_module(
                    "quant_equity.models.model_baselines"
                ).generate_baseline_predictions
            )
        ),
        "composite_directions_available": hasattr(
            models_module,
            "COMPOSITE_FEATURE_DIRECTIONS",
        ),
    }

    lines.extend(
        [
            "Audit checks",
            "-" * 72,
        ]
    )

    failed = 0

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"

        if not passed:
            failed += 1

        lines.append(f"{name}: {status}")

    lines.extend(
        [
            "",
            f"readiness_checks: {len(checks)}",
            f"failed_readiness_checks: {failed}",
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
