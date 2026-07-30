"""Run the daily MVP portfolio backtest."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from quant_equity.backtest import (
    MVPBacktestConfig,
    run_mvp_backtest,
)
from quant_equity.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    load_config,
)
from quant_equity.logging_config import (
    configure_logging,
)

TARGET_WEIGHTS_PATH = PROCESSED_DATA_DIR / "mvp_target_weights.parquet"

MARKET_DATA_PATH = PROCESSED_DATA_DIR / "market_daily.parquet"

DAILY_BACKTEST_PATH = PROCESSED_DATA_DIR / "backtest_mvp_daily.parquet"

POSITIONS_PATH = PROCESSED_DATA_DIR / "positions_mvp.parquet"

TRADES_PATH = PROCESSED_DATA_DIR / "trades_mvp.parquet"

REBALANCE_PATH = PROCESSED_DATA_DIR / "rebalances_mvp.parquet"

TABLES_DIR = REPORTS_DIR / "tables"

EXECUTION_SCHEDULE_PATH = TABLES_DIR / "mvp_execution_schedule.csv"

EXECUTION_SUMMARY_PATH = TABLES_DIR / "mvp_execution_summary.csv"

REBALANCE_SUMMARY_PATH = TABLES_DIR / "mvp_rebalance_summary.csv"

REPORT_PATH = REPORTS_DIR / "backtests" / "mvp_execution_report.md"


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
    execution_summary: pd.DataFrame,
    execution_schedule: pd.DataFrame,
    *,
    config: MVPBacktestConfig,
) -> str:
    """Build the execution report."""
    schedule_summary = pd.DataFrame(
        [
            {
                "signal_dates": len(execution_schedule),
                "first_signal_date": (execution_schedule["signal_date"].min()),
                "first_execution_date": (execution_schedule["execution_date"].min()),
                "last_signal_date": (execution_schedule["signal_date"].max()),
                "final_backtest_date": (execution_schedule["holding_end_date"].iloc[-1]),
            }
        ]
    )

    return "\n".join(
        [
            "# MVP Daily Execution Backtest — Step 8B",
            "",
            "## Execution convention",
            "",
            ("- Signals are observed after the close of `as_of_date`."),
            (
                "- Trades are executed at the adjusted opening "
                "price of the following market session."
            ),
            ("- Positions are valued daily using adjusted closing prices."),
            ("- Fractional shares are permitted in the MVP."),
            ("- Portfolio weights drift naturally between monthly rebalances."),
            (f"- Transaction cost: `{config.transaction_cost_bps:.2f}` bps per dollar traded."),
            "",
            "## Calendar",
            "",
            _to_markdown(schedule_summary),
            "",
            "## Execution summary",
            "",
            _to_markdown(execution_summary),
            "",
            "## Important interpretation",
            "",
            (
                "The preliminary total return shown here is an "
                "accounting validation, not the final model "
                "selection criterion. Risk-adjusted performance, "
                "drawdowns, benchmarks and cost sensitivity are "
                "evaluated in Step 8C."
            ),
            "",
        ]
    )


def main() -> None:
    """Run Step 8B."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    project_config = load_config()

    backtest_config = MVPBacktestConfig.from_mapping(
        project_config.get(
            "mvp_backtest",
            {},
        )
    )

    if not TARGET_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"MVP target weights not found: {TARGET_WEIGHTS_PATH}")

    if not MARKET_DATA_PATH.exists():
        raise FileNotFoundError(f"Daily market data not found: {MARKET_DATA_PATH}")

    target_weights = pd.read_parquet(TARGET_WEIGHTS_PATH)

    market_data = pd.read_parquet(MARKET_DATA_PATH)

    outputs = run_mvp_backtest(
        target_weights,
        market_data,
        config=backtest_config,
    )

    _write_parquet_atomically(
        outputs.daily_performance,
        DAILY_BACKTEST_PATH,
    )

    _write_parquet_atomically(
        outputs.daily_positions,
        POSITIONS_PATH,
    )

    _write_parquet_atomically(
        outputs.trades,
        TRADES_PATH,
    )

    _write_parquet_atomically(
        outputs.rebalance_summary,
        REBALANCE_PATH,
    )

    _write_csv(
        outputs.execution_schedule,
        EXECUTION_SCHEDULE_PATH,
    )

    _write_csv(
        outputs.execution_summary,
        EXECUTION_SUMMARY_PATH,
    )

    _write_csv(
        outputs.rebalance_summary,
        REBALANCE_SUMMARY_PATH,
    )

    report = _build_report(
        outputs.execution_summary,
        outputs.execution_schedule,
        config=backtest_config,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    logger.info("MVP daily backtest completed.")

    logger.info(
        "Daily rows: %s",
        len(outputs.daily_performance),
    )

    logger.info(
        "Position rows: %s",
        len(outputs.daily_positions),
    )

    logger.info(
        "Trade rows: %s",
        len(outputs.trades),
    )

    logger.info(
        "Rebalances: %s",
        len(outputs.rebalance_summary),
    )

    print()
    print("Institutional Quant Equity Research Platform")
    print("MVP daily execution backtest - Step 8B")
    print("-" * 60)
    print(f"Strategies: {outputs.daily_performance['strategy_name'].nunique()}")
    print(f"Signal dates: {len(outputs.execution_schedule)}")
    print(f"Rebalances executed: {len(outputs.rebalance_summary)}")
    print(f"Daily performance rows: {len(outputs.daily_performance)}")
    print(f"Daily position rows: {len(outputs.daily_positions)}")
    print(f"Trade rows: {len(outputs.trades)}")
    print()
    print(outputs.execution_summary.to_string(index=False))
    print()
    print(f"Daily backtest: {DAILY_BACKTEST_PATH}")
    print(f"Positions: {POSITIONS_PATH}")
    print(f"Trades: {TRADES_PATH}")
    print(f"Rebalances: {REBALANCE_PATH}")
    print(f"Execution summary: {EXECUTION_SUMMARY_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
