"""Tests for MVP financial performance evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_equity.backtest import (
    PerformanceEvaluationConfig,
    PerformanceEvaluationError,
    build_buy_and_hold_benchmark,
    evaluate_performance,
)


def make_config() -> PerformanceEvaluationConfig:
    """Create test performance configuration."""
    return PerformanceEvaluationConfig(
        benchmark_name="spy_buy_and_hold",
        benchmark_ticker="SPY",
        annualization_periods=252,
        risk_free_rate=0.0,
        cost_scenarios_bps=(
            0.0,
            10.0,
        ),
        numerical_tolerance=1.0e-12,
    )


def make_daily(
    strategy_name: str,
    returns: list[float],
) -> pd.DataFrame:
    """Create daily portfolio values from returns."""
    dates = pd.bdate_range(
        "2024-01-02",
        periods=len(returns),
    )

    equity = (
        1.0
        + pd.Series(
            returns,
            dtype=float,
        )
    ).cumprod() * 1_000.0

    return pd.DataFrame(
        {
            "date": dates,
            "strategy_name": (strategy_name),
            "portfolio_value": equity,
            "daily_return": returns,
            "transaction_cost": 0.0,
            "traded_notional": 0.0,
            "two_way_turnover": 0.0,
            "one_way_turnover": 0.0,
        }
    )


def make_benchmark_market() -> pd.DataFrame:
    """Create synthetic SPY market data."""
    dates = pd.bdate_range(
        "2024-01-02",
        periods=4,
    )

    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "SPY",
            "open": [
                100.0,
                101.0,
                102.0,
                103.0,
            ],
            "close": [
                101.0,
                102.0,
                103.0,
                104.0,
            ],
            "adjusted_close": [
                101.0,
                102.0,
                103.0,
                104.0,
            ],
        }
    )


def test_buy_and_hold_benchmark_uses_next_period_prices() -> None:
    """The benchmark should invest at the first adjusted open."""
    benchmark = build_buy_and_hold_benchmark(
        make_benchmark_market(),
        strategy_name="spy_buy_and_hold",
        ticker="SPY",
        start_date=pd.Timestamp("2024-01-02"),
        end_date=pd.Timestamp("2024-01-05"),
        initial_capital=1_000.0,
        transaction_cost_bps=0.0,
    )

    expected_final_value = 1_000.0 / 100.0 * 104.0

    assert benchmark["portfolio_value"].iloc[-1] == pytest.approx(expected_final_value)


def test_benchmark_transaction_cost_reduces_value() -> None:
    """An initial benchmark purchase should pay transaction costs."""
    no_cost = build_buy_and_hold_benchmark(
        make_benchmark_market(),
        strategy_name="spy_buy_and_hold",
        ticker="SPY",
        start_date=pd.Timestamp("2024-01-02"),
        end_date=pd.Timestamp("2024-01-05"),
        initial_capital=1_000.0,
        transaction_cost_bps=0.0,
    )

    with_cost = build_buy_and_hold_benchmark(
        make_benchmark_market(),
        strategy_name="spy_buy_and_hold",
        ticker="SPY",
        start_date=pd.Timestamp("2024-01-02"),
        end_date=pd.Timestamp("2024-01-05"),
        initial_capital=1_000.0,
        transaction_cost_bps=10.0,
    )

    assert with_cost["portfolio_value"].iloc[-1] < no_cost["portfolio_value"].iloc[-1]

    assert with_cost["transaction_cost"].sum() > 0.0


def test_flat_strategy_has_zero_total_return() -> None:
    """A flat portfolio should have zero return."""
    strategy = make_daily(
        "flat_strategy",
        [
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )

    benchmark = make_daily(
        "spy_buy_and_hold",
        [
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )

    outputs = evaluate_performance(
        strategy,
        benchmark,
        initial_capital=1_000.0,
        config=make_config(),
    )

    flat_summary = outputs.performance_summary.loc[
        outputs.performance_summary["strategy_name"].eq("flat_strategy")
    ].iloc[0]

    assert flat_summary["total_return"] == pytest.approx(0.0)

    assert flat_summary["maximum_drawdown"] == pytest.approx(0.0)


def test_maximum_drawdown_is_calculated() -> None:
    """A loss after a peak should produce the expected drawdown."""
    strategy = make_daily(
        "strategy",
        [
            0.10,
            -0.20,
            0.00,
            0.00,
        ],
    )

    benchmark = make_daily(
        "spy_buy_and_hold",
        [
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )

    outputs = evaluate_performance(
        strategy,
        benchmark,
        initial_capital=1_000.0,
        config=make_config(),
    )

    strategy_summary = outputs.performance_summary.loc[
        outputs.performance_summary["strategy_name"].eq("strategy")
    ].iloc[0]

    assert strategy_summary["maximum_drawdown"] == pytest.approx(-0.20)


def test_strategy_dates_must_match_benchmark() -> None:
    """Strategies and benchmark must use identical dates."""
    strategy = make_daily(
        "strategy",
        [
            0.01,
            0.01,
            0.01,
        ],
    )

    benchmark = make_daily(
        "spy_buy_and_hold",
        [
            0.01,
            0.01,
            0.01,
            0.01,
        ],
    )

    with pytest.raises(
        PerformanceEvaluationError,
        match="identical trading dates",
    ):
        evaluate_performance(
            strategy,
            benchmark,
            initial_capital=1_000.0,
            config=make_config(),
        )


def test_monthly_and_yearly_outputs_are_created() -> None:
    """Evaluation should create temporal summaries."""
    strategy = make_daily(
        "strategy",
        [
            0.01,
            -0.01,
            0.02,
            0.00,
        ],
    )

    benchmark = make_daily(
        "spy_buy_and_hold",
        [
            0.00,
            0.01,
            0.00,
            0.01,
        ],
    )

    outputs = evaluate_performance(
        strategy,
        benchmark,
        initial_capital=1_000.0,
        config=make_config(),
    )

    assert not outputs.yearly_summary.empty
    assert not outputs.monthly_returns.empty
    assert not outputs.drawdowns.empty
