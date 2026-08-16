"""Evaluate portfolio risk using the final alpha ranking and risk model."""

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
    PortfolioRiskConfig,
    build_top_n_equal_weights,
    calculate_portfolio_risk,
    validate_portfolio_risk,
)

SIGNAL_PATH = PROCESSED_DATA_DIR / "final_alpha_signal.parquet"

RISK_ESTIMATES_PATH = PROCESSED_DATA_DIR / "risk_estimates.parquet"

COVARIANCE_PATH = PROCESSED_DATA_DIR / "covariance_matrices.parquet"

REFERENCE_WEIGHTS_PATH = PROCESSED_DATA_DIR / "reference_portfolio_weights.parquet"

SUMMARY_PATH = REPORTS_DIR / "tables" / "reference_portfolio_risk_summary.csv"

CONTRIBUTIONS_PATH = REPORTS_DIR / "tables" / "reference_portfolio_risk_contributions.csv"

SECTORS_PATH = REPORTS_DIR / "tables" / "reference_portfolio_sector_exposures.csv"

CHECKS_PATH = REPORTS_DIR / "tables" / "portfolio_risk_checks.csv"

REPORT_PATH = REPORTS_DIR / "risk" / "portfolio_risk_report.md"


def _write_report(
    summary: pd.DataFrame,
    contributions: pd.DataFrame,
    sectors: pd.DataFrame,
    checks: pd.DataFrame,
    *,
    top_n: int,
) -> None:
    """Write the portfolio-level risk report."""
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_date = summary["as_of_date"].max()

    latest_summary = summary.loc[summary["as_of_date"].eq(latest_date)]

    latest_contributions = contributions.loc[contributions["as_of_date"].eq(latest_date)].nlargest(
        10,
        "risk_contribution_share",
    )

    latest_sectors = sectors.loc[sectors["as_of_date"].eq(latest_date)].sort_values(
        "portfolio_weight",
        ascending=False,
    )

    distribution = summary[
        [
            "predicted_volatility",
            "portfolio_beta_vs_spy",
            "concentration_hhi",
            "effective_positions",
            "maximum_sector_weight",
            "maximum_position_adv_fraction",
            "maximum_liquidation_days",
        ]
    ].describe()

    lines = [
        "# Portfolio Risk Model",
        "",
        "## Reference portfolio",
        "",
        (
            f"The diagnostic portfolio selects the top {top_n} "
            "securities from the final alpha ranking and assigns "
            "equal weights."
        ),
        "",
        (
            "The reference portfolio is used only to validate "
            "the risk engine. Portfolio construction methods "
            "are evaluated separately downstream."
        ),
        "",
        "## Risk methodology",
        "",
        ("- Portfolio volatility is calculated from the Ledoit-Wolf annualized covariance matrix."),
        ("- Portfolio beta is the weighted average of security betas versus SPY."),
        ("- Security risk contributions use the Euler decomposition of portfolio volatility."),
        ("- Sector exposure is compared with the equal-weight universe sector allocation."),
        ("- Liquidity diagnostics compare position notional with trailing Average Dollar Volume."),
        "",
        "## Readiness checks",
        "",
        "```text",
        checks.to_string(index=False),
        "```",
        "",
        "## Distribution across dates",
        "",
        "```text",
        distribution.to_string(),
        "```",
        "",
        (f"## Latest portfolio risk ({latest_date.date()})"),
        "",
        "```text",
        latest_summary.to_string(index=False),
        "```",
        "",
        (f"## Largest security risk contributions ({latest_date.date()})"),
        "",
        "```text",
        latest_contributions[
            [
                "ticker",
                "sector",
                "weight",
                "beta_vs_spy",
                "risk_contribution_share",
                "position_adv_fraction",
                "liquidation_days",
            ]
        ].to_string(index=False),
        "```",
        "",
        (f"## Sector exposures ({latest_date.date()})"),
        "",
        "```text",
        latest_sectors[
            [
                "sector",
                "portfolio_weight",
                "universe_equal_weight",
                "active_weight",
                "positions",
                "risk_contribution_share",
            ]
        ].to_string(index=False),
        "```",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Evaluate the reference alpha portfolio using the risk model."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    for path in (
        SIGNAL_PATH,
        RISK_ESTIMATES_PATH,
        COVARIANCE_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    project_config = load_config()

    risk_settings = project_config.get(
        "portfolio_risk",
        {},
    )

    top_n = int(
        risk_settings.get(
            "reference_top_n",
            20,
        )
    )

    config = PortfolioRiskConfig.from_mapping(risk_settings)

    final_signal = pd.read_parquet(SIGNAL_PATH)

    risk_estimates = pd.read_parquet(RISK_ESTIMATES_PATH)

    covariance_matrices = pd.read_parquet(COVARIANCE_PATH)

    weights = build_top_n_equal_weights(
        final_signal,
        top_n=top_n,
    )

    (
        summary,
        contributions,
        sectors,
    ) = calculate_portfolio_risk(
        weights,
        risk_estimates,
        covariance_matrices,
        config=config,
    )

    checks = validate_portfolio_risk(
        summary,
        contributions,
        sectors,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weights.to_parquet(
        REFERENCE_WEIGHTS_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    contributions.to_csv(
        CONTRIBUTIONS_PATH,
        index=False,
    )

    sectors.to_csv(
        SECTORS_PATH,
        index=False,
    )

    checks.to_csv(
        CHECKS_PATH,
        index=False,
    )

    _write_report(
        summary,
        contributions,
        sectors,
        checks,
        top_n=top_n,
    )

    if failed_checks:
        raise ValueError(f"Portfolio risk validation failed with {failed_checks} failed checks.")

    logger.info("Reference portfolio risk evaluation completed.")

    latest_date = summary["as_of_date"].max()

    latest = summary.loc[summary["as_of_date"].eq(latest_date)].iloc[0]

    print()
    print("Institutional Quant Equity Research Platform")
    print("Reference portfolio risk evaluation")
    print("------------------------------------------------")

    print(f"dates: {len(summary)}")

    print(f"reference_positions: {top_n}")

    print(f"portfolio_value: {config.portfolio_value:.2f}")

    print(f"mean_predicted_volatility: {summary['predicted_volatility'].mean():.6f}")

    print(f"mean_beta_vs_spy: {summary['portfolio_beta_vs_spy'].mean():.6f}")

    print(f"mean_effective_positions: {summary['effective_positions'].mean():.6f}")

    print(f"maximum_sector_weight_observed: {summary['maximum_sector_weight'].max():.6f}")

    print(f"maximum_position_adv_fraction: {summary['maximum_position_adv_fraction'].max():.8f}")

    print(f"maximum_liquidation_days: {summary['maximum_liquidation_days'].max():.6f}")

    print()
    print(f"Latest date: {latest_date.date()}")

    print(f"predicted_volatility: {latest['predicted_volatility']:.6f}")

    print(f"portfolio_beta_vs_spy: {latest['portfolio_beta_vs_spy']:.6f}")

    print(f"maximum_sector_weight: {latest['maximum_sector_weight']:.6f}")

    print(f"effective_positions: {latest['effective_positions']:.6f}")

    print()
    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()
    print(f"Reference weights: {REFERENCE_WEIGHTS_PATH}")

    print(f"Summary: {SUMMARY_PATH}")

    print(f"Contributions: {CONTRIBUTIONS_PATH}")

    print(f"Sector exposures: {SECTORS_PATH}")

    print(f"Checks: {CHECKS_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
