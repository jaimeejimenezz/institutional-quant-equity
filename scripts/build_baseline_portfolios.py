"""Build and evaluate constrained baseline target portfolios."""

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
    build_equal_weight_portfolios,
    build_score_weighted_portfolios,
    compute_portfolio_diagnostics,
    validate_baseline_portfolios,
)
from quant_equity.risk import (
    PortfolioRiskConfig,
    calculate_portfolio_risk,
)

SIGNAL_PATH = PROCESSED_DATA_DIR / "final_alpha_signal.parquet"

RISK_ESTIMATES_PATH = PROCESSED_DATA_DIR / "risk_estimates.parquet"

COVARIANCE_PATH = PROCESSED_DATA_DIR / "covariance_matrices.parquet"

OUTPUT_PATH = PROCESSED_DATA_DIR / "target_weights_baseline_methods.parquet"

DIAGNOSTICS_PATH = REPORTS_DIR / "tables" / "baseline_portfolio_diagnostics.csv"

RISK_SUMMARY_PATH = REPORTS_DIR / "tables" / "baseline_portfolio_risk_summary.csv"

CHECKS_PATH = REPORTS_DIR / "tables" / "baseline_portfolio_checks.csv"

REPORT_PATH = REPORTS_DIR / "portfolio" / "baseline_portfolio_comparison.md"


def _evaluate_risk(
    weights: pd.DataFrame,
    risk_estimates: pd.DataFrame,
    covariance: pd.DataFrame,
    *,
    config: PortfolioRiskConfig,
) -> pd.DataFrame:
    """Evaluate portfolio risk independently for each construction method."""
    blocks = []

    for (
        method,
        method_weights,
    ) in weights.groupby(
        "method",
        sort=True,
    ):
        summary, _, _ = calculate_portfolio_risk(
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


def _write_report(
    diagnostics: pd.DataFrame,
    risk_summary: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    """Write baseline portfolio-construction comparison."""
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    construction_summary = diagnostics.groupby(
        "method",
        as_index=False,
    ).agg(
        mean_positions=(
            "positions",
            "mean",
        ),
        mean_maximum_weight=(
            "maximum_weight",
            "mean",
        ),
        maximum_sector_weight=(
            "maximum_sector_weight",
            "max",
        ),
        mean_effective_positions=(
            "effective_positions",
            "mean",
        ),
        mean_one_way_turnover=(
            "one_way_turnover",
            "mean",
        ),
    )

    risk_comparison = risk_summary.groupby(
        "method",
        as_index=False,
    ).agg(
        mean_predicted_volatility=(
            "predicted_volatility",
            "mean",
        ),
        mean_beta_vs_spy=(
            "portfolio_beta_vs_spy",
            "mean",
        ),
        maximum_predicted_volatility=(
            "predicted_volatility",
            "max",
        ),
        mean_maximum_sector_weight=(
            "maximum_sector_weight",
            "mean",
        ),
    )

    lines = [
        "# Baseline Portfolio Construction",
        "",
        "## Methods",
        "",
        "- Sector-controlled top-N equal weight",
        "- Constrained alpha score-weighted portfolio",
        "",
        "## Construction diagnostics",
        "",
        "```text",
        construction_summary.to_string(index=False),
        "```",
        "",
        "## Predicted risk comparison",
        "",
        "```text",
        risk_comparison.to_string(index=False),
        "```",
        "",
        "## Readiness checks",
        "",
        "```text",
        checks.to_string(index=False),
        "```",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Build baseline target portfolios and evaluate predicted risk."""
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

    construction_config = BaselinePortfolioConfig.from_mapping(
        project_config.get(
            "portfolio_construction",
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

    risk_estimates = pd.read_parquet(RISK_ESTIMATES_PATH)

    covariance = pd.read_parquet(COVARIANCE_PATH)

    equal_weights = build_equal_weight_portfolios(
        final_signal,
        config=construction_config,
    )

    score_weights = build_score_weighted_portfolios(
        final_signal,
        config=construction_config,
    )

    weights = (
        pd.concat(
            [
                equal_weights,
                score_weights,
            ],
            ignore_index=True,
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

    diagnostics = compute_portfolio_diagnostics(
        weights,
        weight_tolerance=(construction_config.weight_tolerance),
    )

    checks = validate_baseline_portfolios(
        weights,
        config=construction_config,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    risk_summary = _evaluate_risk(
        weights,
        risk_estimates,
        covariance,
        config=risk_config,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DIAGNOSTICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weights.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    diagnostics.to_csv(
        DIAGNOSTICS_PATH,
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

    _write_report(
        diagnostics,
        risk_summary,
        checks,
    )

    if failed_checks:
        raise ValueError(
            f"Baseline portfolio validation failed with {failed_checks} failed checks."
        )

    logger.info("Baseline portfolio construction completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("Baseline portfolio construction")
    print("------------------------------------------------")

    print(f"dates: {weights['as_of_date'].nunique()}")

    print(f"methods: {weights['method'].nunique()}")

    print(f"target_weight_rows: {len(weights)}")

    print()

    print("Construction comparison:")

    comparison = diagnostics.groupby("method").agg(
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

    print(comparison.to_string())

    print()
    print("Predicted risk:")

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

    print(risk_comparison.to_string())

    print()
    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()
    print(f"Target weights: {OUTPUT_PATH}")

    print(f"Diagnostics: {DIAGNOSTICS_PATH}")

    print(f"Risk summary: {RISK_SUMMARY_PATH}")

    print(f"Checks: {CHECKS_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
