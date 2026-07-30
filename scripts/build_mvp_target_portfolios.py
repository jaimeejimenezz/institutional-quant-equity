"""Build Step 8A MVP target portfolios."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
    PortfolioConstructionConfig,
    build_mvp_target_portfolios,
)

PREDICTIONS_PATH = PROCESSED_DATA_DIR / "predictions_linear_oos_evaluated.parquet"

TARGET_WEIGHTS_PATH = PROCESSED_DATA_DIR / "mvp_target_weights.parquet"

TABLES_DIR = REPORTS_DIR / "tables"

PORTFOLIO_SUMMARY_PATH = TABLES_DIR / "mvp_target_portfolio_summary.csv"

CONSTRAINT_CHECKS_PATH = TABLES_DIR / "mvp_target_portfolio_constraints.csv"

REPORT_PATH = REPORTS_DIR / "portfolio" / "mvp_target_portfolios_report.md"


def _write_parquet_atomically(
    data: pd.DataFrame,
    path: Path,
) -> Path:
    """Write a Parquet dataset atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(".tmp.parquet")

    temporary_path.unlink(missing_ok=True)

    data.to_parquet(
        temporary_path,
        index=False,
    )

    temporary_path.replace(path)

    return path


def _write_csv(
    data: pd.DataFrame,
    path: Path,
) -> Path:
    """Write a CSV table."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        path,
        index=False,
    )

    return path


def _format_value(
    value: Any,
) -> str:
    """Format a value for Markdown."""
    if value is None or pd.isna(value):
        return ""

    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()

    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value).replace(
        "|",
        "\\|",
    )


def _to_markdown(
    data: pd.DataFrame,
) -> str:
    """Convert a dataframe to a Markdown table."""
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
    portfolio_summary: pd.DataFrame,
    constraint_checks: pd.DataFrame,
    *,
    config: PortfolioConstructionConfig,
) -> str:
    """Build the Step 8A Markdown report."""
    latest_checks = (
        constraint_checks.sort_values("as_of_date")
        .groupby(
            "strategy_name",
            as_index=False,
        )
        .tail(1)
        .loc[
            :,
            [
                "strategy_name",
                "as_of_date",
                "holdings",
                "maximum_weight",
                "maximum_sector_weight",
                "constraints_pass",
            ],
        ]
        .sort_values("strategy_name")
    )

    return "\n".join(
        [
            "# MVP Target Portfolios — Step 8A",
            "",
            "## Objective",
            "",
            (
                "Transform the out-of-sample model predictions "
                "into monthly long-only target portfolios before "
                "running the daily execution backtest."
            ),
            "",
            "## Frozen portfolio constraints",
            "",
            f"- Top-N holdings: `{config.top_n}`",
            (f"- Score-weighted candidate count: `{config.score_weighted_candidate_count}`"),
            (f"- Maximum company weight: `{config.max_weight:.2%}`"),
            (f"- Maximum sector weight: `{config.max_sector_weight:.2%}`"),
            (f"- Primary model: `{config.primary_model_name}`"),
            (f"- Challenger model: `{config.challenger_model_name}`"),
            "",
            "## Strategy summary",
            "",
            _to_markdown(portfolio_summary),
            "",
            "## Latest target portfolios",
            "",
            _to_markdown(latest_checks),
            "",
            "## Interpretation",
            "",
            (
                "These are target weights, not realized daily "
                "positions. The following step must map every "
                "signal date to the next trading session, create "
                "orders, model shares and cash, apply transaction "
                "costs, and allow weights to drift between "
                "rebalances."
            ),
            "",
        ]
    )


def main() -> None:
    """Run Step 8A portfolio construction."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    project_config = load_config()

    construction_config = PortfolioConstructionConfig.from_mapping(
        project_config.get(
            "portfolio_construction",
            {},
        )
    )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Evaluated prediction dataset not found: {PREDICTIONS_PATH}")

    predictions = pd.read_parquet(PREDICTIONS_PATH)

    outputs = build_mvp_target_portfolios(
        predictions,
        config=construction_config,
    )

    _write_parquet_atomically(
        outputs.target_weights,
        TARGET_WEIGHTS_PATH,
    )

    _write_csv(
        outputs.portfolio_summary,
        PORTFOLIO_SUMMARY_PATH,
    )

    _write_csv(
        outputs.constraint_checks,
        CONSTRAINT_CHECKS_PATH,
    )

    report = _build_report(
        outputs.portfolio_summary,
        outputs.constraint_checks,
        config=construction_config,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    logger.info("MVP target portfolio construction completed.")

    logger.info(
        "Target-weight rows: %s",
        len(outputs.target_weights),
    )

    logger.info(
        "Signal dates: %s",
        outputs.target_weights["as_of_date"].nunique(),
    )

    logger.info(
        "Strategies: %s",
        outputs.target_weights["strategy_name"].nunique(),
    )

    failed_checks = int((~outputs.constraint_checks["constraints_pass"]).sum())

    print()
    print("Institutional Quant Equity Research Platform")
    print("MVP target portfolio construction - Step 8A")
    print("-" * 60)
    print(f"Target-weight rows: {len(outputs.target_weights)}")
    print(f"Signal dates: {outputs.target_weights['as_of_date'].nunique()}")
    print(f"Strategies: {outputs.target_weights['strategy_name'].nunique()}")
    print(f"Constraint checks: {len(outputs.constraint_checks)}")
    print(f"Failed constraint checks: {failed_checks}")
    print()
    print(outputs.portfolio_summary.to_string(index=False))
    print()
    print(f"Target weights: {TARGET_WEIGHTS_PATH}")
    print(f"Summary: {PORTFOLIO_SUMMARY_PATH}")
    print(f"Constraint checks: {CONSTRAINT_CHECKS_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
