"""Tests for the daily MVP execution engine."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_equity.backtest import (
    MVPBacktestConfig,
    MVPBacktestError,
    build_execution_schedule,
    run_mvp_backtest,
)


def make_config(
    *,
    transaction_cost_bps: float = 0.0,
) -> MVPBacktestConfig:
    """Create a compact backtest configuration."""
    return MVPBacktestConfig(
        initial_capital=1_000.0,
        transaction_cost_bps=(transaction_cost_bps),
        final_holding_sessions=2,
        minimum_trade_notional=0.0001,
        cash_tolerance=0.01,
        weight_tolerance=1.0e-8,
        share_tolerance=1.0e-12,
        bisection_tolerance=1.0e-10,
        bisection_max_iterations=200,
    )


def make_market_data() -> pd.DataFrame:
    """Create a synthetic five-session market."""
    dates = pd.bdate_range(
        "2024-01-02",
        periods=6,
    )

    rows: list[dict[str, object]] = []

    for date in dates:
        for ticker in (
            "A",
            "B",
            "C",
            "D",
        ):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": 100.0,
                    "close": 100.0,
                    "adjusted_close": 100.0,
                }
            )

    return pd.DataFrame(rows)


def make_target_weights() -> pd.DataFrame:
    """Create two strategies and two signal dates."""
    signal_dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-04",
        ]
    )

    sectors = {
        "A": "Sector 1",
        "B": "Sector 1",
        "C": "Sector 2",
        "D": "Sector 2",
    }

    rows: list[dict[str, object]] = []

    alpha_weights = (
        {
            "A": 0.50,
            "B": 0.50,
            "C": 0.00,
            "D": 0.00,
        },
        {
            "A": 0.00,
            "B": 0.00,
            "C": 0.50,
            "D": 0.50,
        },
    )

    benchmark_weights = {
        "A": 0.25,
        "B": 0.25,
        "C": 0.25,
        "D": 0.25,
    }

    for date_index, signal_date in enumerate(signal_dates):
        for ticker, weight in alpha_weights[date_index].items():
            rows.append(
                {
                    "as_of_date": signal_date,
                    "strategy_name": "alpha",
                    "ticker": ticker,
                    "sector": sectors[ticker],
                    "target_weight": weight,
                }
            )

        for ticker, weight in benchmark_weights.items():
            rows.append(
                {
                    "as_of_date": signal_date,
                    "strategy_name": "benchmark",
                    "ticker": ticker,
                    "sector": sectors[ticker],
                    "target_weight": weight,
                }
            )

    return pd.DataFrame(rows)


def test_execution_schedule_uses_next_session() -> None:
    """Signals should be executed in the following session."""
    market_dates = pd.bdate_range(
        "2024-01-02",
        periods=6,
    )

    schedule = build_execution_schedule(
        pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-04",
            ]
        ),
        market_dates,
        final_holding_sessions=2,
    )

    assert schedule["execution_date"].tolist() == [
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-05"),
    ]

    assert (schedule["execution_date"] > schedule["signal_date"]).all()


def test_final_holding_period_uses_requested_sessions() -> None:
    """The final signal should be held for two sessions."""
    market_dates = pd.bdate_range(
        "2024-01-02",
        periods=6,
    )

    schedule = build_execution_schedule(
        pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-04",
            ]
        ),
        market_dates,
        final_holding_sessions=2,
    )

    assert schedule["holding_end_date"].iloc[-1] == pd.Timestamp("2024-01-08")


def test_constant_prices_preserve_capital_without_costs() -> None:
    """A constant market should preserve capital with zero costs."""
    outputs = run_mvp_backtest(
        make_target_weights(),
        make_market_data(),
        config=make_config(transaction_cost_bps=0.0),
    )

    final_values = (
        outputs.daily_performance.sort_values("date")
        .groupby("strategy_name")["portfolio_value"]
        .last()
    )

    assert (final_values.sub(1_000.0).abs() <= 1.0e-8).all()

    assert (outputs.daily_performance["daily_return"].abs() <= 1.0e-10).all()


def test_transaction_costs_reduce_portfolio_value() -> None:
    """Positive transaction costs should reduce final NAV."""
    outputs = run_mvp_backtest(
        make_target_weights(),
        make_market_data(),
        config=make_config(transaction_cost_bps=10.0),
    )

    final_values = (
        outputs.daily_performance.sort_values("date")
        .groupby("strategy_name")["portfolio_value"]
        .last()
    )

    assert final_values.lt(1_000.0).all()

    assert outputs.rebalance_summary["transaction_cost"].gt(0.0).any()


def test_weights_drift_between_rebalances() -> None:
    """Market movements should move actual weights away from targets."""
    market_data = make_market_data()

    mask = market_data["date"].eq(pd.Timestamp("2024-01-04")) & market_data["ticker"].eq("A")

    market_data.loc[
        mask,
        "close",
    ] = 120.0

    market_data.loc[
        mask,
        "adjusted_close",
    ] = 120.0

    outputs = run_mvp_backtest(
        make_target_weights(),
        market_data,
        config=make_config(),
    )

    position = outputs.daily_positions.loc[
        outputs.daily_positions["date"].eq(pd.Timestamp("2024-01-04"))
        & outputs.daily_positions["strategy_name"].eq("alpha")
        & outputs.daily_positions["ticker"].eq("A")
    ].iloc[0]

    assert position["actual_weight"] > 0.50

    assert position["weight_drift"] > 0.0


def test_rebalance_trades_include_buys_and_sells() -> None:
    """Changing the selected companies should create both sides."""
    outputs = run_mvp_backtest(
        make_target_weights(),
        make_market_data(),
        config=make_config(),
    )

    second_alpha_rebalance = outputs.trades.loc[
        outputs.trades["strategy_name"].eq("alpha")
        & outputs.trades["execution_date"].eq(pd.Timestamp("2024-01-05"))
    ]

    assert set(second_alpha_rebalance["side"]) == {
        "BUY",
        "SELL",
    }


def test_missing_execution_price_is_rejected() -> None:
    """A selected company must have an execution price."""
    market_data = make_market_data()

    mask = market_data["date"].eq(pd.Timestamp("2024-01-03")) & market_data["ticker"].eq("A")

    market_data.loc[
        mask,
        "open",
    ] = pd.NA

    with pytest.raises(
        MVPBacktestError,
        match="Missing execution price",
    ):
        run_mvp_backtest(
            make_target_weights(),
            market_data,
            config=make_config(),
        )


def test_all_expected_rebalances_are_executed() -> None:
    """Every strategy and signal should create one rebalance."""
    outputs = run_mvp_backtest(
        make_target_weights(),
        make_market_data(),
        config=make_config(),
    )

    assert len(outputs.rebalance_summary) == 4

    assert (outputs.rebalance_summary.groupby("strategy_name").size() == 2).all()


def test_cash_is_negligible_after_rebalancing() -> None:
    """Self-financing execution should leave negligible cash."""
    outputs = run_mvp_backtest(
        make_target_weights(),
        make_market_data(),
        config=make_config(transaction_cost_bps=10.0),
    )

    rebalance_days = outputs.daily_performance.loc[outputs.daily_performance["is_rebalance"]]

    assert rebalance_days["cash"].abs().le(0.01).all()
