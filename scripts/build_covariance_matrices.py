"""Build rolling point-in-time shrinkage covariance matrices."""

from __future__ import annotations

import logging

import pandas as pd

from quant_equity.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    load_config,
)
from quant_equity.logging_config import (
    configure_logging,
)
from quant_equity.risk import (
    CovarianceConfig,
    build_covariance_matrices,
    validate_covariance_matrices,
)

MARKET_PATH = PROCESSED_DATA_DIR / "market_daily.parquet"

SIGNAL_PATH = PROCESSED_DATA_DIR / "final_alpha_signal.parquet"

MATRIX_DIRECTORY = PROCESSED_DATA_DIR / "covariance_matrices"

COMBINED_MATRIX_PATH = PROCESSED_DATA_DIR / "covariance_matrices.parquet"

DIAGNOSTICS_PATH = REPORTS_DIR / "tables" / "covariance_diagnostics.csv"

CHECKS_PATH = REPORTS_DIR / "tables" / "covariance_checks.csv"

REPORT_PATH = REPORTS_DIR / "risk" / "covariance_report.md"


def _write_report(
    matrices: pd.DataFrame,
    diagnostics: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    """Write the covariance estimation report."""
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostic_summary = diagnostics[
        [
            "observations",
            "shrinkage",
            "minimum_eigenvalue",
            "maximum_eigenvalue",
            "sample_condition_number",
            "shrinkage_condition_number",
            "mean_pairwise_correlation",
        ]
    ].describe()

    latest_date = diagnostics["as_of_date"].max()

    latest_diagnostics = diagnostics.loc[diagnostics["as_of_date"].eq(latest_date)]

    latest_correlations = (
        matrices.loc[
            matrices["as_of_date"].eq(latest_date) & matrices["ticker_a"].ne(matrices["ticker_b"])
        ]
        .assign(absolute_correlation=lambda data: data["correlation"].abs())
        .nlargest(
            15,
            "absolute_correlation",
        )[
            [
                "ticker_a",
                "ticker_b",
                "correlation",
            ]
        ]
    )

    lines = [
        "# Shrinkage Covariance Model",
        "",
        "## Methodology",
        "",
        ("Covariance matrices are estimated independently for every monthly signal date."),
        "",
        ("Only daily returns available on or before the corresponding as-of date are used."),
        "",
        (
            "Ledoit-Wolf shrinkage is applied to reduce "
            "sampling instability in the covariance matrix."
        ),
        "",
        ("Daily covariance estimates are annualized using 252 trading sessions."),
        "",
        "## Coverage",
        "",
        "```text",
        (f"dates: {diagnostics['as_of_date'].nunique()}"),
        (f"matrix_rows: {len(matrices)}"),
        (f"assets_per_date_min: {diagnostics['assets'].min()}"),
        (f"assets_per_date_max: {diagnostics['assets'].max()}"),
        (f"observations_min: {diagnostics['observations'].min()}"),
        (f"observations_max: {diagnostics['observations'].max()}"),
        "```",
        "",
        "## Readiness checks",
        "",
        "```text",
        checks.to_string(index=False),
        "```",
        "",
        "## Diagnostic distribution",
        "",
        "```text",
        diagnostic_summary.to_string(),
        "```",
        "",
        (f"## Latest-date diagnostics ({latest_date.date()})"),
        "",
        "```text",
        latest_diagnostics.to_string(index=False),
        "```",
        "",
        (f"## Strongest latest-date correlations ({latest_date.date()})"),
        "",
        "```text",
        latest_correlations.to_string(index=False),
        "```",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Build and audit rolling covariance matrices."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    for path in (
        MARKET_PATH,
        SIGNAL_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    project_config = load_config()

    covariance_config = CovarianceConfig.from_mapping(
        project_config.get(
            "risk_model",
            {},
        )
    )

    market_data = pd.read_parquet(MARKET_PATH)

    final_signal = pd.read_parquet(SIGNAL_PATH)

    (
        matrices,
        diagnostics,
    ) = build_covariance_matrices(
        market_data,
        final_signal,
        config=covariance_config,
    )

    checks = validate_covariance_matrices(
        matrices,
        diagnostics,
        final_signal,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    DIAGNOSTICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics.to_csv(
        DIAGNOSTICS_PATH,
        index=False,
    )

    checks.to_csv(
        CHECKS_PATH,
        index=False,
    )

    _write_report(
        matrices,
        diagnostics,
        checks,
    )

    if failed_checks:
        raise ValueError(f"Covariance validation failed with {failed_checks} failed checks.")

    MATRIX_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for existing_file in MATRIX_DIRECTORY.glob("*.parquet"):
        existing_file.unlink()

    for (
        as_of_date,
        date_matrix,
    ) in matrices.groupby(
        "as_of_date",
        sort=True,
    ):
        output_path = MATRIX_DIRECTORY / (
            pd.Timestamp(as_of_date).strftime("%Y-%m-%d") + ".parquet"
        )

        date_matrix.to_parquet(
            output_path,
            index=False,
        )

    matrices.to_parquet(
        COMBINED_MATRIX_PATH,
        index=False,
    )

    logger.info("Rolling shrinkage covariance matrices completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("Rolling shrinkage covariance matrices")
    print("------------------------------------------------")

    print(f"dates: {diagnostics['as_of_date'].nunique()}")

    print(f"assets_per_date_min: {diagnostics['assets'].min()}")

    print(f"assets_per_date_max: {diagnostics['assets'].max()}")

    print(f"matrix_rows: {len(matrices)}")

    print(f"minimum_joint_observations: {diagnostics['observations'].min()}")

    print(f"maximum_joint_observations: {diagnostics['observations'].max()}")

    print(f"mean_shrinkage: {diagnostics['shrinkage'].mean():.6f}")

    print(f"minimum_eigenvalue: {diagnostics['minimum_eigenvalue'].min():.12f}")

    print(f"mean_pairwise_correlation: {diagnostics['mean_pairwise_correlation'].mean():.6f}")

    print(f"median_sample_condition_number: {diagnostics['sample_condition_number'].median():.4f}")

    print(
        "median_shrinkage_condition_number: "
        f"{diagnostics['shrinkage_condition_number'].median():.4f}"
    )

    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()
    print(f"Matrices directory: {MATRIX_DIRECTORY}")

    print(f"Combined matrix: {COMBINED_MATRIX_PATH}")

    print(f"Diagnostics: {DIAGNOSTICS_PATH}")

    print(f"Checks: {CHECKS_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
