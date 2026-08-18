from __future__ import annotations

import pandas as pd
import pytest

from quant_equity.reporting.dashboard_metrics import (
    capacity_curve,
    execution_cost_breakdown,
    execution_dates,
    execution_method_comparison,
    execution_summary_row,
    execution_trade_snapshot,
    rebalance_execution_history,
    transaction_cost_sensitivity_curve,
)


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [
                "2026-04-30",
                "2026-04-30",
                "2026-05-29",
                "2026-05-29",
            ],
            "execution_date": [
                "2026-05-01",
                "2026-05-01",
                "2026-06-01",
                "2026-06-01",
            ],
            "strategy_name": ["selected"] * 4,
            "ticker": ["AAA", "BBB", "AAA", "CCC"],
            "side": ["BUY", "SELL", "BUY", "SELL"],
            "trade_notional": [100_000.0, -50_000.0, 120_000.0, -80_000.0],
            "absolute_trade_notional": [
                100_000.0,
                50_000.0,
                120_000.0,
                80_000.0,
            ],
            "total_execution_cost": [50.0, 25.0, 72.0, 48.0],
            "effective_cost_bps": [5.0, 5.0, 6.0, 6.0],
            "order_adv_fraction": [0.01, 0.02, 0.015, 0.025],
        }
    )


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy_name": ["selected", "other"],
            "start_date": ["2020-01-31", "2020-01-31"],
            "end_date": ["2026-06-30", "2026-06-30"],
            "rebalances": [77, 77],
            "final_portfolio_value": [4_000_000.0, 3_500_000.0],
            "total_transaction_cost": [40_000.0, 35_000.0],
            "mean_one_way_turnover": [0.20, 0.15],
        }
    )


def _costs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy_name": ["selected", "other"],
            "commission_cost": [10_000.0, 8_000.0],
            "spread_cost": [12_000.0, 10_000.0],
            "slippage_cost": [9_000.0, 8_000.0],
            "market_impact_cost": [9_000.0, 9_000.0],
            "total_execution_cost": [40_000.0, 35_000.0],
            "effective_cost_bps": [5.2, 5.0],
        }
    )


def test_execution_trade_snapshot_and_dates() -> None:
    dates = execution_dates(_trades(), "selected")
    assert dates == (
        pd.Timestamp("2026-05-01"),
        pd.Timestamp("2026-06-01"),
    )

    snapshot = execution_trade_snapshot(
        _trades(),
        "selected",
        pd.Timestamp("2026-06-01"),
    )
    assert snapshot["ticker"].tolist() == ["AAA", "CCC"]
    assert snapshot["total_execution_cost"].sum() == pytest.approx(120.0)


def test_rebalance_execution_history_uses_gross_traded_notional() -> None:
    history = rebalance_execution_history(_trades(), "selected")
    assert len(history) == 2

    first = history.iloc[0]
    assert float(first["traded_notional"]) == pytest.approx(150_000.0)
    assert float(first["effective_cost_bps"]) == pytest.approx(5.0)

    latest = history.iloc[-1]
    assert float(latest["execution_cost"]) == pytest.approx(120.0)
    assert float(latest["traded_notional"]) == pytest.approx(200_000.0)
    assert float(latest["effective_cost_bps"]) == pytest.approx(6.0)


def test_execution_cost_breakdown_sums_components() -> None:
    breakdown = execution_cost_breakdown(_costs(), "selected")
    assert breakdown["cost"].sum() == pytest.approx(40_000.0)
    assert breakdown["share"].sum() == pytest.approx(1.0)


def test_execution_summary_and_method_comparison() -> None:
    row = execution_summary_row(_summary(), "selected")
    assert int(row["rebalances"]) == 77

    comparison = execution_method_comparison(
        _summary(),
        _costs(),
        "selected",
    )
    selected = comparison.loc[comparison["role"] == "selected"]
    assert selected["strategy_name"].tolist() == ["selected"]


def test_capacity_curve_sorts_capital() -> None:
    capacity = pd.DataFrame(
        {
            "capital": [10_000_000.0, 1_000_000.0],
            "strategy_name": ["selected", "selected"],
            "net_cagr": [0.20, 0.22],
            "net_sharpe_ratio": [1.0, 1.1],
            "effective_cost_bps": [8.0, 5.0],
            "maximum_order_adv_fraction": [0.05, 0.01],
        }
    )
    curve = capacity_curve(capacity, "selected")
    assert curve["capital"].tolist() == [1_000_000.0, 10_000_000.0]


def test_transaction_cost_sensitivity_is_strategy_scoped() -> None:
    sensitivity = pd.DataFrame(
        {
            "scenario": ["base", "high"],
            "strategy_name": ["selected", "selected"],
            "cagr": [0.22, 0.20],
            "sharpe_ratio": [1.1, 1.0],
            "maximum_drawdown": [-0.30, -0.31],
            "total_transaction_cost": [40_000.0, 60_000.0],
        }
    )
    curve = transaction_cost_sensitivity_curve(
        sensitivity,
        "selected",
    )
    assert curve["scenario"].tolist() == ["base", "high"]
