"""Freeze the feature-family contract used by robustness ablations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from quant_equity.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from quant_equity.features.technical_processing import (
    TECHNICAL_MODEL_FEATURE_COLUMNS,
)
from quant_equity.logging_config import configure_logging
from quant_equity.models.modeling_panel import (
    MODEL_FEATURE_COLUMNS,
)

MODELING_PANEL_PATH = PROCESSED_DATA_DIR / "modeling_panel.parquet"

TABLES_DIR = REPORTS_DIR / "tables"

CONTRACT_PATH = TABLES_DIR / "robustness_feature_family_contract.csv"

CHECKS_PATH = TABLES_DIR / "robustness_feature_family_contract_checks.csv"

REPORT_PATH = REPORTS_DIR / "robustness" / "feature_family_ablation_contract.md"

EXPECTED_MODEL_FEATURES = 91

MOMENTUM_FEATURE_ROOTS = (
    "return_1w",
    "return_1m",
    "return_3m",
    "momentum_6_1",
    "momentum_12_1",
    "reversal_1m",
)


def _write_csv(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write one CSV table."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    data.to_csv(
        path,
        index=False,
    )


def _canonical_technical_name(
    model_feature: str,
) -> str | None:
    """Recover the exact canonical technical name from the modeling namespace."""
    feature = str(model_feature).strip()

    if feature.startswith("tech__"):
        return feature[len("tech__") :]

    if feature in {str(value) for value in TECHNICAL_MODEL_FEATURE_COLUMNS}:
        return feature

    return None


def _technical_mapping() -> dict[str, str]:
    """Map model-panel technical features by exact canonical name."""
    expected_canonical = {str(feature) for feature in TECHNICAL_MODEL_FEATURE_COLUMNS}

    mapping: dict[
        str,
        str,
    ] = {}

    for model_feature in MODEL_FEATURE_COLUMNS:
        feature_name = str(model_feature)

        candidate = _canonical_technical_name(feature_name)

        if candidate is not None and candidate in expected_canonical:
            mapping[feature_name] = candidate

    mapped_values = list(mapping.values())

    if len(mapped_values) != len(set(mapped_values)):
        duplicates = sorted({value for value in mapped_values if mapped_values.count(value) > 1})

        raise ValueError(
            "Multiple modeling features map to the same canonical "
            f"technical predictor: {duplicates}."
        )

    return mapping


def _is_momentum_canonical_feature(
    canonical_feature: str,
) -> bool:
    """Identify the explicitly frozen momentum family."""
    return any(canonical_feature == (root + "_sector_neutral") for root in MOMENTUM_FEATURE_ROOTS)


def _build_contract() -> pd.DataFrame:
    """Build the frozen feature-family classification."""
    technical_mapping = _technical_mapping()

    rows = []

    for feature in MODEL_FEATURE_COLUMNS:
        feature_name = str(feature)

        canonical_technical_feature = technical_mapping.get(feature_name)

        is_technical = canonical_technical_feature is not None

        is_fundamental = not is_technical

        is_momentum = bool(
            is_technical and _is_momentum_canonical_feature(canonical_technical_feature)
        )

        if is_fundamental:
            family = "fundamental"
        elif is_momentum:
            family = "technical_momentum"
        else:
            family = "technical_other"

        rows.append(
            {
                "feature": feature_name,
                "canonical_technical_feature": (canonical_technical_feature or ""),
                "family": family,
                "is_fundamental": bool(is_fundamental),
                "is_technical": bool(is_technical),
                "is_momentum": bool(is_momentum),
                "included_full_model": True,
                "included_no_fundamentals": bool(not is_fundamental),
                "included_no_momentum": bool(not is_momentum),
            }
        )

    return pd.DataFrame(rows)


def _build_checks(
    panel: pd.DataFrame,
    contract: pd.DataFrame,
) -> pd.DataFrame:
    """Audit the feature-family contract."""
    model_features = list(MODEL_FEATURE_COLUMNS)

    panel_columns = set(panel.columns)

    technical = contract.loc[contract["is_technical"]]

    fundamental = contract.loc[contract["is_fundamental"]]

    momentum = contract.loc[contract["is_momentum"]]

    no_fundamentals = contract.loc[contract["included_no_fundamentals"]]

    no_momentum = contract.loc[contract["included_no_momentum"]]

    mapped_canonical = set(technical["canonical_technical_feature"])

    expected_canonical = {str(feature) for feature in TECHNICAL_MODEL_FEATURE_COLUMNS}

    checks = [
        (
            "expected_model_feature_count",
            int(len(model_features) != EXPECTED_MODEL_FEATURES),
            (f"The frozen modeling contract should contain {EXPECTED_MODEL_FEATURES} predictors."),
        ),
        (
            "unique_model_feature_names",
            int(len(model_features) != len(set(model_features))),
            "Frozen model feature names must be unique.",
        ),
        (
            "all_model_features_in_panel",
            int(len(set(model_features).difference(panel_columns))),
            "Every frozen model feature must exist in modeling_panel.parquet.",
        ),
        (
            "technical_contract_complete",
            int(mapped_canonical != expected_canonical),
            (
                "Every canonical technical model feature must map to "
                "exactly one frozen modeling-panel predictor."
            ),
        ),
        (
            "expected_technical_feature_count",
            int(len(technical) != len(TECHNICAL_MODEL_FEATURE_COLUMNS)),
            "The technical family must match the canonical technical-model contract.",
        ),
        (
            "fundamental_family_nonempty",
            int(fundamental.empty),
            "The no-fundamentals ablation requires fundamental predictors.",
        ),
        (
            "momentum_family_nonempty",
            int(momentum.empty),
            "The no-momentum ablation requires explicit momentum predictors.",
        ),
        (
            "families_partition_model_features",
            int((len(technical) + len(fundamental)) != len(contract)),
            "Technical and fundamental families must partition the model predictors.",
        ),
        (
            "no_fundamentals_retains_only_technical",
            int((~no_fundamentals["is_technical"]).sum()),
            "The no-fundamentals contract must retain only technical predictors.",
        ),
        (
            "no_momentum_removes_all_momentum",
            int(no_momentum["is_momentum"].sum()),
            "The no-momentum contract must remove every frozen momentum predictor.",
        ),
        (
            "ablations_retain_predictors",
            int(len(no_fundamentals) < 2 or len(no_momentum) < 2),
            "Each ablation must retain a usable predictor set.",
        ),
    ]

    return pd.DataFrame(
        [
            {
                "check": name,
                "status": ("PASS" if violations == 0 else "FAIL"),
                "violations": int(violations),
                "description": description,
            }
            for (
                name,
                violations,
                description,
            ) in checks
        ]
    )


def _format_value(
    value: Any,
) -> str:
    """Format one value for Markdown."""
    if value is None or pd.isna(value):
        return ""

    return str(value).replace(
        "|",
        "\\|",
    )


def _to_markdown(
    data: pd.DataFrame,
) -> str:
    """Convert a dataframe to Markdown."""
    if data.empty:
        return "_No observations._"

    columns = [str(column) for column in data.columns]

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in data.itertuples(
        index=False,
        name=None,
    ):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")

    return "\n".join(lines)


def _build_report(
    contract: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    """Build the frozen feature-family ablation contract report."""
    summary = contract.groupby(
        "family",
        as_index=False,
    ).agg(
        feature_count=(
            "feature",
            "size",
        )
    )

    fundamental_features = contract.loc[
        contract["is_fundamental"],
        [
            "feature",
        ],
    ]

    momentum_features = contract.loc[
        contract["is_momentum"],
        [
            "feature",
        ],
    ]

    return "\n".join(
        [
            "# Feature-Family Ablation Contract",
            "",
            "## Purpose",
            "",
            (
                "This document freezes the predictor families used by the "
                "feature ablation experiments before any ablation model is trained."
            ),
            ("The classification changes no prediction, portfolio or backtest result."),
            "",
            "## Rules",
            "",
            (
                "- Technical predictors are identified from the canonical "
                "`TECHNICAL_MODEL_FEATURE_COLUMNS` contract created by the "
                "technical processing pipeline."
            ),
            (
                "- Fundamental predictors are the remaining frozen model "
                "predictors after the complete technical contract is mapped."
            ),
            (
                "- Momentum predictors are frozen inside the technical family as "
                "return, momentum and reversal roots: " + ", ".join(MOMENTUM_FEATURE_ROOTS) + "."
            ),
            (
                "- `no_fundamentals` removes all fundamental predictors and "
                "retains every technical predictor."
            ),
            (
                "- `no_momentum` removes only the explicitly frozen momentum "
                "predictors and retains fundamentals and other technical predictors."
            ),
            ("- These families must not be changed after viewing the ablation results."),
            "",
            "## Family counts",
            "",
            _to_markdown(summary),
            "",
            "## Fundamental predictors removed",
            "",
            _to_markdown(fundamental_features),
            "",
            "## Momentum predictors removed",
            "",
            _to_markdown(momentum_features),
            "",
            "## Full contract",
            "",
            _to_markdown(contract),
            "",
            "## Readiness checks",
            "",
            _to_markdown(checks),
            "",
        ]
    )


def main() -> None:
    """Freeze and audit feature families for model ablations."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    if not MODELING_PANEL_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {MODELING_PANEL_PATH}")

    panel = pd.read_parquet(MODELING_PANEL_PATH)

    contract = _build_contract()

    checks = _build_checks(
        panel,
        contract,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    _write_csv(
        contract,
        CONTRACT_PATH,
    )

    _write_csv(
        checks,
        CHECKS_PATH,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        _build_report(
            contract,
            checks,
        ),
        encoding="utf-8",
    )

    if failed_checks:
        raise ValueError(
            "Feature-family ablation contract validation failed with "
            f"{failed_checks} failed checks."
        )

    fundamental_count = int(contract["is_fundamental"].sum())

    momentum_count = int(contract["is_momentum"].sum())

    technical_count = int(contract["is_technical"].sum())

    logger.info("Feature-family ablation contract completed.")

    print()

    print("Institutional Quant Equity Research Platform")

    print("Feature-family ablation contract")

    print("------------------------------------------------")

    print(f"full_model_features: {len(contract)}")

    print(f"fundamental_features: {fundamental_count}")

    print(f"technical_features: {technical_count}")

    print(f"momentum_features: {momentum_count}")

    print(f"no_fundamentals_features: {int(contract['included_no_fundamentals'].sum())}")

    print(f"no_momentum_features: {int(contract['included_no_momentum'].sum())}")

    print()

    print("Canonical technical predictors mapped:")

    print(
        contract.loc[
            contract["is_technical"],
            [
                "feature",
                "canonical_technical_feature",
            ],
        ].to_string(index=False)
    )

    print()

    print("Momentum predictors removed:")

    print(
        contract.loc[
            contract["is_momentum"],
            [
                "feature",
            ],
        ].to_string(index=False)
    )

    print()

    print("Feature-family counts:")

    print(contract.groupby("family")["feature"].size().rename("count").to_string())

    print()

    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()

    print(f"Contract table: {CONTRACT_PATH}")

    print(f"Checks table: {CHECKS_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
