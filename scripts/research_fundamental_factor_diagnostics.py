"""Research diagnostics for processed fundamental factors."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from quant_equity.config import (
    PROJECT_ROOT,
    REPORTS_DIR,
    load_config,
)
from quant_equity.features import (
    FundamentalAuditConfig,
    build_cross_sectional_correlations,
    build_fundamental_feature_summary,
    build_zscore_audit,
    select_high_correlations,
    update_feature_dictionary,
    validate_feature_metadata,
)
from quant_equity.features.fundamental_transforms import (
    FUNDAMENTAL_FACTOR_COLUMNS,
)
from quant_equity.logging_config import (
    configure_logging,
)

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features_fundamental_monthly.parquet"

SUMMARY_PATH = REPORTS_DIR / "tables" / "fundamental_feature_summary.csv"

CORRELATION_PATH = REPORTS_DIR / "tables" / "fundamental_factor_correlations.csv"

HIGH_CORRELATION_PATH = REPORTS_DIR / "tables" / "fundamental_high_correlations.csv"

ZSCORE_AUDIT_PATH = REPORTS_DIR / "tables" / "fundamental_zscore_audit.csv"

REPORT_PATH = REPORTS_DIR / "research" / "fundamental_factor_audit.md"

DICTIONARY_PATH = PROJECT_ROOT / "docs" / "FEATURE_DICTIONARY.md"


def _write_csv(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write one report table."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        path,
        index=False,
    )


def _coverage_tier(
    coverage: float,
) -> str:
    """Give coverage a descriptive non-predictive tier."""
    if coverage >= 0.90:
        return "high"

    if coverage >= 0.70:
        return "moderate"

    return "limited"


def _write_report(
    *,
    data: pd.DataFrame,
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
    high_correlations: pd.DataFrame,
    zscore_audit: pd.DataFrame,
    config: FundamentalAuditConfig,
    path: Path,
) -> None:
    """Write the Step 10E research report."""
    latest_date = data["as_of_date"].max()

    duplicates = int(
        data.duplicated(
            [
                "as_of_date",
                "ticker",
            ]
        ).sum()
    )

    global_violations = int(
        zscore_audit.loc[
            zscore_audit["version"].eq("zscore"),
            "centering_violations",
        ].sum()
    )

    sector_violations = int(
        zscore_audit.loc[
            zscore_audit["version"].eq("sector_zscore"),
            "centering_violations",
        ].sum()
    )

    latest_summary = summary.loc[
        :,
        [
            "factor",
            "family",
            "latest_raw_coverage",
            "overall_global_zscore_coverage",
            "overall_sector_zscore_coverage",
        ],
    ].copy()

    latest_summary["coverage_tier"] = latest_summary["latest_raw_coverage"].map(_coverage_tier)

    lines = [
        "# Fundamental factor audit",
        "",
        "## Step",
        "",
        ("Step 10E — Final non-predictive audit of fundamental features."),
        "",
        "## Dataset",
        "",
        f"- Rows: {len(data)}",
        (f"- Rebalance dates: {data['as_of_date'].nunique()}"),
        (f"- Companies: {data['ticker'].nunique()}"),
        (f"- Fundamental factors: {len(FUNDAMENTAL_FACTOR_COLUMNS)}"),
        (f"- Latest date: {latest_date.date()}"),
        (f"- Duplicate date-ticker rows: {duplicates}"),
        "",
        "## Z-score audit",
        "",
        (f"- Global centering violations: {global_violations}"),
        (f"- Sector centering violations: {sector_violations}"),
        "",
        "## Correlation diagnostics",
        "",
        (f"- Factor pairs evaluated: {len(correlations)}"),
        (f"- High-correlation threshold: {config.high_correlation_threshold:.2f}"),
        (f"- Highly correlated pairs: {len(high_correlations)}"),
        "",
        (
            "High correlation is treated only as a "
            "redundancy warning. No factor is removed "
            "in Step 10E."
        ),
        "",
        "## Coverage summary",
        "",
        "```text",
        latest_summary.to_string(index=False),
        "```",
        "",
        "## Highly correlated pairs",
        "",
        "```text",
        (high_correlations.to_string(index=False) if not high_correlations.empty else "None"),
        "```",
        "",
        "## Methodological boundary",
        "",
        (
            "Step 10E does not use future returns, "
            "target_21d, target_21d_excess, model "
            "performance or backtest results."
        ),
        "",
        (
            "Predictive relevance and final feature "
            "selection are deferred to the modeling "
            "panel and walk-forward research stages."
        ),
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Execute Step 10E."""
    project_config = load_config()

    configure_logging()

    logger = logging.getLogger("quant_equity")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {INPUT_PATH}")

    data = pd.read_parquet(INPUT_PATH)

    data["as_of_date"] = pd.to_datetime(
        data["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    audit_config = FundamentalAuditConfig.from_mapping(
        project_config["fundamental_factors"]["audit"]
    )

    validate_feature_metadata()

    summary = build_fundamental_feature_summary(data)

    correlations = build_cross_sectional_correlations(
        data,
        config=audit_config,
    )

    high_correlations = select_high_correlations(
        correlations,
        config=audit_config,
    )

    zscore_audit = build_zscore_audit(
        data,
        config=audit_config,
    )

    _write_csv(
        summary,
        SUMMARY_PATH,
    )

    _write_csv(
        correlations,
        CORRELATION_PATH,
    )

    _write_csv(
        high_correlations,
        HIGH_CORRELATION_PATH,
    )

    _write_csv(
        zscore_audit,
        ZSCORE_AUDIT_PATH,
    )

    update_feature_dictionary(DICTIONARY_PATH)

    _write_report(
        data=data,
        summary=summary,
        correlations=correlations,
        high_correlations=high_correlations,
        zscore_audit=zscore_audit,
        config=audit_config,
        path=REPORT_PATH,
    )

    duplicates = int(
        data.duplicated(
            [
                "as_of_date",
                "ticker",
            ]
        ).sum()
    )

    factor_columns = []

    for factor in FUNDAMENTAL_FACTOR_COLUMNS:
        factor_columns.extend(
            [
                factor,
                f"{factor}_zscore",
                f"{factor}_sector_zscore",
            ]
        )

    infinite_values = int(np.isinf(data[factor_columns].to_numpy(dtype=float)).sum())

    centering_violations = int(zscore_audit["centering_violations"].sum())

    latest_date = data["as_of_date"].max()

    latest_summary = summary.sort_values(
        "latest_raw_coverage",
        ascending=False,
    )

    logger.info("Fundamental factor diagnostics completed.")

    print()
    print("Institutional Quant Equity Research Platform")

    print("Fundamental factor diagnostics - Step 10E")

    print("------------------------------------------------")

    print(f"Rows: {len(data)}")

    print(f"Rebalance dates: {data['as_of_date'].nunique()}")

    print(f"Companies: {data['ticker'].nunique()}")

    print(f"Fundamental factors: {len(FUNDAMENTAL_FACTOR_COLUMNS)}")

    print(f"Factor pairs evaluated: {len(correlations)}")

    print(f"Highly correlated pairs: {len(high_correlations)}")

    print(f"Duplicate date-ticker rows: {duplicates}")

    print(f"Infinite factor values: {infinite_values}")

    print(f"Z-score centering violations: {centering_violations}")

    print()

    print(f"Latest raw coverage ({latest_date.date()}):")

    print(
        latest_summary.loc[
            :,
            [
                "factor",
                "family",
                "latest_raw_coverage",
            ],
        ].to_string(index=False)
    )

    print()

    print("Highly correlated factor pairs:")

    if high_correlations.empty:
        print("None")
    else:
        print(high_correlations.to_string(index=False))

    print()

    print(f"Summary: {SUMMARY_PATH}")

    print(f"Correlations: {CORRELATION_PATH}")

    print(f"High correlations: {HIGH_CORRELATION_PATH}")

    print(f"Z-score audit: {ZSCORE_AUDIT_PATH}")

    print(f"Research report: {REPORT_PATH}")

    print(f"Feature dictionary: {DICTIONARY_PATH}")

    print()

    print("Fundamental factor diagnostics: OK")


if __name__ == "__main__":
    main()
