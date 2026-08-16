"""Build median-MAD portfolios and compare portfolio-construction methods."""

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
from quant_equity.portfolio import (
    BaselinePortfolioConfig,
    CvarRiskConfig,
    MedianMadConfig,
    PortfolioOptimizerConfig,
    build_median_mad_portfolios,
    compute_portfolio_diagnostics,
    validate_baseline_portfolios,
    validate_cvar_diagnostics,
    validate_median_mad_diagnostics,
    validate_optimizer_diagnostics,
)
from quant_equity.risk import (
    PortfolioRiskConfig,
    calculate_portfolio_risk,
)

SIGNAL_PATH = PROCESSED_DATA_DIR / "final_alpha_signal.parquet"

MARKET_PATH = PROCESSED_DATA_DIR / "market_daily.parquet"

RISK_ESTIMATES_PATH = PROCESSED_DATA_DIR / "risk_estimates.parquet"

COVARIANCE_PATH = PROCESSED_DATA_DIR / "covariance_matrices.parquet"

FOUR_METHOD_WEIGHTS_PATH = PROCESSED_DATA_DIR / "target_weights_four_methods.parquet"

OPTIMIZER_DIAGNOSTICS_PATH = REPORTS_DIR / "tables" / "portfolio_optimizer_diagnostics.csv"

CVAR_DIAGNOSTICS_PATH = REPORTS_DIR / "tables" / "cvar_portfolio_diagnostics.csv"

MEDIAN_MAD_WEIGHTS_PATH = PROCESSED_DATA_DIR / "target_weights_median_mad.parquet"

ALL_WEIGHTS_PATH = PROCESSED_DATA_DIR / "target_weights_all_methods.parquet"

MEDIAN_MAD_DIAGNOSTICS_PATH = REPORTS_DIR / "tables" / "median_mad_portfolio_diagnostics.csv"

CONSTRUCTION_DIAGNOSTICS_PATH = REPORTS_DIR / "tables" / "all_method_portfolio_diagnostics.csv"

RISK_SUMMARY_PATH = REPORTS_DIR / "tables" / "all_method_risk_summary.csv"

CHECKS_PATH = REPORTS_DIR / "tables" / "all_method_portfolio_checks.csv"

REPORT_PATH = REPORTS_DIR / "portfolio" / "portfolio_construction_comparison.md"


def evaluate_risk(
    weights: pd.DataFrame,
    risk_estimates: pd.DataFrame,
    covariance: pd.DataFrame,
    *,
    config: PortfolioRiskConfig,
) -> pd.DataFrame:
    """Evaluate predicted risk for each construction method."""
    blocks = []

    for (
        method,
        method_weights,
    ) in weights.groupby(
        "method",
        sort=True,
    ):
        (
            summary,
            _,
            _,
        ) = calculate_portfolio_risk(
            method_weights[
                [
                    "as_of_date",
                    "ticker",
                    "weight",
                ]
            ],
            risk_estimates,
            covariance,
            config=config,
        )

        summary["method"] = method

        blocks.append(summary)

    return pd.concat(
        blocks,
        ignore_index=True,
    )


def main() -> None:
    """Build and evaluate median-MAD portfolios."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    required_paths = (
        SIGNAL_PATH,
        MARKET_PATH,
        RISK_ESTIMATES_PATH,
        COVARIANCE_PATH,
        FOUR_METHOD_WEIGHTS_PATH,
        OPTIMIZER_DIAGNOSTICS_PATH,
        CVAR_DIAGNOSTICS_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    project_config = load_config()

    construction_config = BaselinePortfolioConfig.from_mapping(
        project_config.get(
            "portfolio_construction",
            {},
        )
    )

    optimizer_config = PortfolioOptimizerConfig.from_mapping(
        project_config.get(
            "portfolio_optimizer",
            {},
        )
    )

    cvar_config = CvarRiskConfig.from_mapping(
        project_config.get(
            "portfolio_cvar",
            {},
        )
    )

    median_mad_config = MedianMadConfig.from_mapping(
        project_config.get(
            "portfolio_median_mad",
            {},
        )
    )

    risk_config = PortfolioRiskConfig.from_mapping(
        project_config.get(
            "portfolio_risk",
            {},
        )
    )

    final_signal = pd.read_parquet(SIGNAL_PATH)

    market_daily = pd.read_parquet(MARKET_PATH)

    risk_estimates = pd.read_parquet(RISK_ESTIMATES_PATH)

    covariance = pd.read_parquet(COVARIANCE_PATH)

    four_method_weights = pd.read_parquet(FOUR_METHOD_WEIGHTS_PATH)

    optimizer_diagnostics = pd.read_csv(OPTIMIZER_DIAGNOSTICS_PATH)

    cvar_diagnostics = pd.read_csv(CVAR_DIAGNOSTICS_PATH)

    (
        median_mad_weights,
        median_mad_diagnostics,
    ) = build_median_mad_portfolios(
        final_signal,
        market_daily,
        risk_estimates,
        portfolio_config=optimizer_config,
        median_mad_config=median_mad_config,
    )

    combined_weights = (
        pd.concat(
            [
                four_method_weights,
                median_mad_weights,
            ],
            ignore_index=True,
            sort=False,
        )
        .sort_values(
            [
                "method",
                "as_of_date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    construction_diagnostics = compute_portfolio_diagnostics(
        combined_weights,
        weight_tolerance=(construction_config.weight_tolerance),
    )

    general_checks = validate_baseline_portfolios(
        combined_weights,
        config=construction_config,
    )

    optimizer_checks = validate_optimizer_diagnostics(
        optimizer_diagnostics,
        config=optimizer_config,
    )

    cvar_checks = validate_cvar_diagnostics(
        cvar_diagnostics,
        portfolio_config=optimizer_config,
        cvar_config=cvar_config,
    )

    median_mad_checks = validate_median_mad_diagnostics(
        median_mad_diagnostics,
        portfolio_config=optimizer_config,
        median_mad_config=median_mad_config,
    )

    general_checks.insert(
        0,
        "scope",
        "all_methods",
    )

    optimizer_checks.insert(
        0,
        "scope",
        "alpha_risk_turnover",
    )

    cvar_checks.insert(
        0,
        "scope",
        "cvar",
    )

    median_mad_checks.insert(
        0,
        "scope",
        "median_mad_de",
    )

    checks = pd.concat(
        [
            general_checks,
            optimizer_checks,
            cvar_checks,
            median_mad_checks,
        ],
        ignore_index=True,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    risk_summary = evaluate_risk(
        combined_weights,
        risk_estimates,
        covariance,
        config=risk_config,
    )

    MEDIAN_MAD_WEIGHTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MEDIAN_MAD_DIAGNOSTICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    median_mad_weights.to_parquet(
        MEDIAN_MAD_WEIGHTS_PATH,
        index=False,
    )

    combined_weights.to_parquet(
        ALL_WEIGHTS_PATH,
        index=False,
    )

    median_mad_diagnostics.to_csv(
        MEDIAN_MAD_DIAGNOSTICS_PATH,
        index=False,
    )

    construction_diagnostics.to_csv(
        CONSTRUCTION_DIAGNOSTICS_PATH,
        index=False,
    )

    risk_summary.to_csv(
        RISK_SUMMARY_PATH,
        index=False,
    )

    checks.to_csv(
        CHECKS_PATH,
        index=False,
    )

    comparison = construction_diagnostics.groupby("method").agg(
        positions=(
            "positions",
            "mean",
        ),
        maximum_weight=(
            "maximum_weight",
            "max",
        ),
        maximum_sector_weight=(
            "maximum_sector_weight",
            "max",
        ),
        effective_positions=(
            "effective_positions",
            "mean",
        ),
        one_way_turnover=(
            "one_way_turnover",
            "mean",
        ),
    )

    risk_comparison = risk_summary.groupby("method").agg(
        predicted_volatility=(
            "predicted_volatility",
            "mean",
        ),
        beta_vs_spy=(
            "portfolio_beta_vs_spy",
            "mean",
        ),
        maximum_sector_weight=(
            "maximum_sector_weight",
            "max",
        ),
    )

    mad_exceedance_rate = float(median_mad_diagnostics["mad_violation"].gt(0.0).mean())

    report_lines = [
        "# Portfolio Construction Comparison",
        "",
        "## Construction comparison",
        "",
        "```text",
        comparison.to_string(),
        "```",
        "",
        "## Predicted risk",
        "",
        "```text",
        risk_comparison.to_string(),
        "```",
        "",
        "## Median-MAD diagnostics",
        "",
        "```text",
        median_mad_diagnostics[
            [
                "median_daily_return",
                "mad_daily",
                "mad_violation",
                "turnover_l1",
                "one_way_turnover",
                "portfolio_beta_vs_spy",
            ]
        ]
        .describe()
        .to_string(),
        "```",
        "",
        f"MAD limit exceedance rate: {mad_exceedance_rate:.2%}",
        "",
        "## Readiness checks",
        "",
        "```text",
        checks.to_string(index=False),
        "```",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    if failed_checks:
        raise ValueError(f"Portfolio validation failed with {failed_checks} failed checks.")

    logger.info("Median-MAD portfolio construction completed.")

    print()

    print("Institutional Quant Equity Research Platform")

    print("Median-MAD portfolio construction")

    print("------------------------------------------------")

    print(f"dates: {combined_weights['as_of_date'].nunique()}")

    print(f"methods: {combined_weights['method'].nunique()}")

    print()

    print("Construction comparison:")

    print(comparison.to_string())

    print()

    print("Predicted risk:")

    print(risk_comparison.to_string())

    print()

    print("Median-MAD diagnostics:")

    print(
        median_mad_diagnostics[
            [
                "median_daily_return",
                "mad_daily",
                "mad_violation",
                "turnover_l1",
                "one_way_turnover",
                "portfolio_beta_vs_spy",
            ]
        ]
        .mean()
        .to_string()
    )

    print()

    print(f"MAD limit exceedance rate: {mad_exceedance_rate:.2%}")

    print()

    print(f"optimizer_success_rate: {median_mad_diagnostics['optimizer_success'].mean():.2%}")

    print()

    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()

    print(f"Median-MAD weights: {MEDIAN_MAD_WEIGHTS_PATH}")

    print(f"All weights: {ALL_WEIGHTS_PATH}")

    print(f"Median-MAD diagnostics: {MEDIAN_MAD_DIAGNOSTICS_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
