"""Build point-in-time security-level risk and liquidity estimates."""

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
    RiskEstimateConfig,
    build_risk_estimates,
    validate_risk_estimates,
)

MARKET_PATH = PROCESSED_DATA_DIR / "market_daily.parquet"

SPY_PATH = PROCESSED_DATA_DIR / "benchmark_spy_daily.parquet"

SIGNAL_PATH = PROCESSED_DATA_DIR / "final_alpha_signal.parquet"

RISK_ESTIMATES_PATH = PROCESSED_DATA_DIR / "risk_estimates.parquet"

CHECKS_PATH = REPORTS_DIR / "tables" / "risk_estimate_checks.csv"

REPORT_PATH = REPORTS_DIR / "risk" / "risk_estimates_report.md"


def _write_report(
    estimates: pd.DataFrame,
    checks: pd.DataFrame,
    *,
    config: RiskEstimateConfig,
) -> None:
    """Write the security-level risk estimate report."""
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptive = estimates[
        [
            "annualized_volatility",
            "annualized_downside_volatility",
            "beta_vs_spy",
            "correlation_vs_spy",
            "average_dollar_volume",
        ]
    ].describe()

    latest_date = estimates["as_of_date"].max()

    latest = estimates.loc[estimates["as_of_date"].eq(latest_date)].copy()

    highest_risk = latest.nlargest(
        10,
        "annualized_volatility",
    )[
        [
            "ticker",
            "sector",
            "annualized_volatility",
            "beta_vs_spy",
            "average_dollar_volume",
        ]
    ]

    lowest_liquidity = latest.nsmallest(
        10,
        "average_dollar_volume",
    )[
        [
            "ticker",
            "sector",
            "annualized_volatility",
            "beta_vs_spy",
            "average_dollar_volume",
        ]
    ]

    lines = [
        "# Security-Level Risk Estimates",
        "",
        "## Methodology",
        "",
        (
            "- Volatility uses the latest "
            f"{config.volatility_window_sessions} "
            "available daily adjusted-close returns."
        ),
        (f"- Beta uses up to {config.beta_window_sessions} daily returns aligned with SPY."),
        (
            "- Liquidity uses a "
            f"{config.liquidity_window_sessions}-session "
            "average dollar volume window."
        ),
        ("- Every estimation window ends on or before the signal as-of date."),
        ("- No forward return or future target is used in this artifact."),
        "",
        "## Coverage",
        "",
        "```text",
        f"rows: {len(estimates)}",
        (f"dates: {estimates['as_of_date'].nunique()}"),
        (f"tickers: {estimates['ticker'].nunique()}"),
        (f"first_date: {estimates['as_of_date'].min().date()}"),
        (f"last_date: {estimates['as_of_date'].max().date()}"),
        "```",
        "",
        "## Readiness checks",
        "",
        "```text",
        checks.to_string(index=False),
        "```",
        "",
        "## Distribution summary",
        "",
        "```text",
        descriptive.to_string(),
        "```",
        "",
        (f"## Highest volatility on {latest_date.date()}"),
        "",
        "```text",
        highest_risk.to_string(index=False),
        "```",
        "",
        (f"## Lowest dollar volume on {latest_date.date()}"),
        "",
        "```text",
        lowest_liquidity.to_string(index=False),
        "```",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Build and validate monthly security-level risk estimates."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    for path in (
        MARKET_PATH,
        SPY_PATH,
        SIGNAL_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    project_config = load_config()

    risk_config = RiskEstimateConfig.from_mapping(
        project_config.get(
            "risk_model",
            {},
        )
    )

    market_data = pd.read_parquet(MARKET_PATH)

    spy_data = pd.read_parquet(SPY_PATH)

    final_signal = pd.read_parquet(SIGNAL_PATH)

    estimates = build_risk_estimates(
        market_data,
        spy_data,
        final_signal,
        config=risk_config,
    )

    checks = validate_risk_estimates(
        estimates,
        final_signal,
        config=risk_config,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    CHECKS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checks.to_csv(
        CHECKS_PATH,
        index=False,
    )

    _write_report(
        estimates,
        checks,
        config=risk_config,
    )

    if failed_checks:
        raise ValueError(f"Risk estimate validation failed with {failed_checks} failed checks.")

    RISK_ESTIMATES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    estimates.to_parquet(
        RISK_ESTIMATES_PATH,
        index=False,
    )

    logger.info("Security-level risk estimates completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("Security-level risk estimates")
    print("------------------------------------------------")

    print(f"rows: {len(estimates)}")

    print(f"dates: {estimates['as_of_date'].nunique()}")

    print(f"tickers: {estimates['ticker'].nunique()}")

    print(f"first_date: {estimates['as_of_date'].min().date()}")

    print(f"last_date: {estimates['as_of_date'].max().date()}")

    print(f"minimum_return_observations: {estimates['return_observations'].min()}")

    print(f"minimum_beta_observations: {estimates['beta_observations'].min()}")

    print(f"minimum_liquidity_observations: {estimates['liquidity_observations'].min()}")

    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()
    print(f"Risk estimates: {RISK_ESTIMATES_PATH}")

    print(f"Checks: {CHECKS_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
