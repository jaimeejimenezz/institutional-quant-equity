"""Build and compare risk-aware optimized target portfolios."""

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
    PortfolioOptimizerConfig,
    build_alpha_risk_turnover_portfolios,
    compute_portfolio_diagnostics,
    validate_baseline_portfolios,
    validate_optimizer_diagnostics,
)
from quant_equity.risk import (
    PortfolioRiskConfig,
    calculate_portfolio_risk,
)

SIGNAL_PATH = PROCESSED_DATA_DIR / "final_alpha_signal.parquet"

COVARIANCE_PATH = PROCESSED_DATA_DIR / "covariance_matrices.parquet"

RISK_ESTIMATES_PATH = PROCESSED_DATA_DIR / "risk_estimates.parquet"

BASELINE_WEIGHTS_PATH = PROCESSED_DATA_DIR / "target_weights_baseline_methods.parquet"

OPTIMIZED_WEIGHTS_PATH = PROCESSED_DATA_DIR / "target_weights_optimized.parquet"

COMBINED_WEIGHTS_PATH = PROCESSED_DATA_DIR / "target_weights_core_methods.parquet"

OPTIMIZER_DIAGNOSTICS_PATH = REPORTS_DIR / "tables" / "portfolio_optimizer_diagnostics.csv"

CONSTRUCTION_DIAGNOSTICS_PATH = REPORTS_DIR / "tables" / "core_portfolio_diagnostics.csv"

RISK_SUMMARY_PATH = REPORTS_DIR / "tables" / "core_portfolio_risk_summary.csv"

CHECKS_PATH = REPORTS_DIR / "tables" / "core_portfolio_checks.csv"

REPORT_PATH = REPORTS_DIR / "portfolio" / "portfolio_optimizer_comparison.md"


def evaluate_risk(
    weights: pd.DataFrame,
    risk_estimates: pd.DataFrame,
    covariance: pd.DataFrame,
    *,
    config: PortfolioRiskConfig,
) -> pd.DataFrame:
    """Evaluate risk independently for every construction method."""
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


def write_report(
    diagnostics: pd.DataFrame,
    risk_summary: pd.DataFrame,
    optimizer_diagnostics: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    """Write comparison of core portfolio-construction methods."""
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    construction_comparison = diagnostics.groupby(
        "method",
        as_index=False,
    ).agg(
        mean_positions=(
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
        maximum_sector_weight=(
            "maximum_sector_weight",
            "max",
        ),
    )

    optimizer_summary = optimizer_diagnostics[
        [
            "predicted_alpha_proxy",
            "predicted_volatility",
            "one_way_turnover",
            "positions",
            "maximum_weight",
            "maximum_sector_weight",
        ]
    ].describe()

    lines = [
        "# Portfolio Optimizer Comparison",
        "",
        "## Construction methods",
        "",
        "- Sector-controlled top-N equal weight",
        "- Constrained score-weighted",
        "- Alpha-risk-turnover optimizer",
        "",
        "## Construction diagnostics",
        "",
        "```text",
        construction_comparison.to_string(index=False),
        "```",
        "",
        "## Predicted risk",
        "",
        "```text",
        risk_comparison.to_string(index=False),
        "```",
        "",
        "## Optimizer diagnostics",
        "",
        "```text",
        optimizer_summary.to_string(),
        "```",
        "",
        "## Constraint checks",
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
    """Build the alpha-risk-turnover portfolio and compare methods."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    required_paths = (
        SIGNAL_PATH,
        COVARIANCE_PATH,
        RISK_ESTIMATES_PATH,
        BASELINE_WEIGHTS_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    project_config = load_config()

    optimizer_config = PortfolioOptimizerConfig.from_mapping(
        project_config.get(
            "portfolio_optimizer",
            {},
        )
    )

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

    covariance = pd.read_parquet(COVARIANCE_PATH)

    risk_estimates = pd.read_parquet(RISK_ESTIMATES_PATH)

    baseline_weights = pd.read_parquet(BASELINE_WEIGHTS_PATH)

    (
        optimized_weights,
        optimizer_diagnostics,
    ) = build_alpha_risk_turnover_portfolios(
        final_signal,
        covariance,
        risk_estimates,
        config=optimizer_config,
    )

    combined_weights = (
        pd.concat(
            [
                baseline_weights,
                optimized_weights,
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

    checks = validate_baseline_portfolios(
        combined_weights,
        config=construction_config,
    )

    optimizer_checks = validate_optimizer_diagnostics(
        optimizer_diagnostics,
        config=optimizer_config,
    )

    checks = pd.concat(
        [
            checks,
            optimizer_checks,
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

    OPTIMIZED_WEIGHTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OPTIMIZER_DIAGNOSTICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    optimized_weights.to_parquet(
        OPTIMIZED_WEIGHTS_PATH,
        index=False,
    )

    combined_weights.to_parquet(
        COMBINED_WEIGHTS_PATH,
        index=False,
    )

    optimizer_diagnostics.to_csv(
        OPTIMIZER_DIAGNOSTICS_PATH,
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

    write_report(
        construction_diagnostics,
        risk_summary,
        optimizer_diagnostics,
        checks,
    )

    if failed_checks:
        raise ValueError(
            f"Portfolio construction validation failed with {failed_checks} failed checks."
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

    logger.info("Risk-aware portfolio optimization completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("Risk-aware portfolio optimization")
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
    print("Optimizer objective diagnostics:")

    print(
        optimizer_diagnostics[
            [
                "predicted_alpha_proxy",
                "predicted_volatility",
                "one_way_turnover",
            ]
        ]
        .mean()
        .to_string()
    )

    print()
    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()
    print(f"Optimized weights: {OPTIMIZED_WEIGHTS_PATH}")

    print(f"Combined weights: {COMBINED_WEIGHTS_PATH}")

    print(f"Optimizer diagnostics: {OPTIMIZER_DIAGNOSTICS_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
