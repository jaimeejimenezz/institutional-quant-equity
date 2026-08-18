from __future__ import annotations

import pytest

from quant_equity.reporting.dashboard_data import read_dashboard_source
from quant_equity.reporting.dashboard_metrics import (
    build_drawdown_series,
    build_performance_index,
    latest_portfolio_weights,
    sector_exposure,
    strategy_summary,
)

STRATEGY = "score_weighted"


def test_overview_summary_uses_canonical_strategy_row() -> None:
    summary_table = read_dashboard_source("performance_summary")
    summary = strategy_summary(summary_table, STRATEGY)

    assert summary.strategy_name == STRATEGY
    assert summary.net_cagr > -1.0
    assert summary.net_sharpe_ratio == pytest.approx(
        float(
            summary_table.loc[
                summary_table["strategy_name"] == STRATEGY,
                "net_sharpe_ratio",
            ].iloc[0]
        )
    )


def test_performance_index_starts_at_one_hundred() -> None:
    net_daily = read_dashboard_source("performance_net_daily")
    benchmark = read_dashboard_source("benchmark_spy")
    indexed = build_performance_index(net_daily, benchmark, STRATEGY)

    starts = indexed.sort_values("date").groupby("series")["index_value"].first()
    assert not starts.empty
    assert all(value == pytest.approx(100.0) for value in starts)


def test_drawdowns_are_non_positive() -> None:
    net_daily = read_dashboard_source("performance_net_daily")
    benchmark = read_dashboard_source("benchmark_spy")
    indexed = build_performance_index(net_daily, benchmark, STRATEGY)
    drawdowns = build_drawdown_series(indexed)

    assert float(drawdowns["drawdown"].max()) <= 1e-12
    assert drawdowns["drawdown"].min() <= 0.0


def test_latest_portfolio_and_sector_exposure_are_fully_invested() -> None:
    target_weights = read_dashboard_source("target_weights")
    portfolio = latest_portfolio_weights(target_weights, STRATEGY)
    sectors = sector_exposure(portfolio)

    assert portfolio["weight"].sum() == pytest.approx(1.0, abs=1e-6)
    assert sectors["sector_weight"].sum() == pytest.approx(1.0, abs=1e-6)
    assert (portfolio["weight"] > 0.0).all()
