"""Tests for advanced execution costs inside the daily backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_equity.backtest import (
    ExecutionCostConfig,
    MVPBacktestConfig,
    MVPBacktestError,
    estimate_trade_execution_costs,
    run_mvp_backtest,
)


def make_backtest_config(
    *,
    initial_capital: float = 1_000_000.0,
    transaction_cost_bps: float = 0.0,
) -> MVPBacktestConfig:
    """Create compact execution settings."""
    return MVPBacktestConfig(
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        final_holding_sessions=2,
        minimum_trade_notional=0.0001,
        cash_tolerance=0.05,
        weight_tolerance=1e-8,
        share_tolerance=1e-12,
        bisection_tolerance=1e-8,
        bisection_max_iterations=200,
    )


def make_execution_config(
    *,
    market_impact_coefficient: float = 0.10,
) -> ExecutionCostConfig:
    """Create deterministic advanced execution settings."""
    return ExecutionCostConfig(
        commission_bps=0.5,
        half_spread_bps=2.0,
        slippage_bps=2.5,
        market_impact_coefficient=market_impact_coefficient,
        annualization_factor=252,
    )


def make_market_data() -> pd.DataFrame:
    """Create constant synthetic market prices."""
    dates = pd.bdate_range(
        "2024-01-02",
        periods=6,
    )

    rows = []

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
    """Create two target portfolios."""
    sectors = {
        "A": "Sector 1",
        "B": "Sector 1",
        "C": "Sector 2",
        "D": "Sector 2",
    }

    allocations = {
        pd.Timestamp("2024-01-02"): {
            "A": 0.50,
            "B": 0.50,
            "C": 0.00,
            "D": 0.00,
        },
        pd.Timestamp("2024-01-04"): {
            "A": 0.00,
            "B": 0.00,
            "C": 0.50,
            "D": 0.50,
        },
    }

    rows = []

    for signal_date, weights in allocations.items():
        for ticker, weight in weights.items():
            rows.append(
                {
                    "as_of_date": signal_date,
                    "strategy_name": "advanced",
                    "ticker": ticker,
                    "sector": sectors[ticker],
                    "target_weight": weight,
                }
            )

    return pd.DataFrame(rows)


def make_risk_estimates() -> pd.DataFrame:
    """Create point-in-time volatility and liquidity estimates."""
    rows = []

    for signal_date in (
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-04"),
    ):
        for ticker in (
            "A",
            "B",
            "C",
            "D",
        ):
            rows.append(
                {
                    "as_of_date": signal_date,
                    "ticker": ticker,
                    "annualized_volatility": 0.20,
                    "average_dollar_volume": 100_000_000.0,
                }
            )

    return pd.DataFrame(rows)


def test_zero_impact_advanced_cost_matches_linear_five_bps() -> None:
    """Advanced linear components must reproduce a five-bps legacy model."""
    legacy = run_mvp_backtest(
        make_target_weights(),
        make_market_data(),
        config=make_backtest_config(transaction_cost_bps=5.0),
    )

    advanced = run_mvp_backtest(
        make_target_weights(),
        make_market_data(),
        config=make_backtest_config(transaction_cost_bps=0.0),
        risk_estimates=make_risk_estimates(),
        execution_cost_config=make_execution_config(market_impact_coefficient=0.0),
    )

    assert np.allclose(
        legacy.daily_performance["portfolio_value"],
        advanced.daily_performance["portfolio_value"],
        rtol=0.0,
        atol=1e-6,
    )

    assert np.allclose(
        legacy.rebalance_summary["transaction_cost"],
        advanced.rebalance_summary["transaction_cost"],
        rtol=0.0,
        atol=1e-6,
    )


def test_advanced_execution_remains_self_financing() -> None:
    """Advanced costs must leave only numerical residual cash."""
    outputs = run_mvp_backtest(
        make_target_weights(),
        make_market_data(),
        config=make_backtest_config(),
        risk_estimates=make_risk_estimates(),
        execution_cost_config=make_execution_config(),
    )

    assert outputs.rebalance_summary["cash_after"].abs().max() <= 0.05


def test_market_impact_reduces_final_portfolio_value() -> None:
    """Positive market impact must reduce ending NAV."""
    without_impact = run_mvp_backtest(
        make_target_weights(),
        make_market_data(),
        config=make_backtest_config(),
        risk_estimates=make_risk_estimates(),
        execution_cost_config=make_execution_config(market_impact_coefficient=0.0),
    )

    with_impact = run_mvp_backtest(
        make_target_weights(),
        make_market_data(),
        config=make_backtest_config(),
        risk_estimates=make_risk_estimates(),
        execution_cost_config=make_execution_config(market_impact_coefficient=0.10),
    )

    final_without = float(without_impact.daily_performance["portfolio_value"].iloc[-1])

    final_with = float(with_impact.daily_performance["portfolio_value"].iloc[-1])

    assert final_with < final_without


def test_larger_portfolio_has_higher_cost_fraction() -> None:
    """Market impact must worsen as portfolio size increases."""
    small = run_mvp_backtest(
        make_target_weights(),
        make_market_data(),
        config=make_backtest_config(initial_capital=100_000.0),
        risk_estimates=make_risk_estimates(),
        execution_cost_config=make_execution_config(),
    )

    large = run_mvp_backtest(
        make_target_weights(),
        make_market_data(),
        config=make_backtest_config(initial_capital=10_000_000.0),
        risk_estimates=make_risk_estimates(),
        execution_cost_config=make_execution_config(),
    )

    small_fraction = float(
        small.execution_summary.loc[
            0,
            "total_transaction_cost",
        ]
        / 100_000.0
    )

    large_fraction = float(
        large.execution_summary.loc[
            0,
            "total_transaction_cost",
        ]
        / 10_000_000.0
    )

    assert large_fraction > small_fraction


def test_missing_execution_risk_is_rejected() -> None:
    """Every required signal-date ticker must have execution inputs."""
    risk = make_risk_estimates()

    risk = risk.loc[~(risk["as_of_date"].eq(pd.Timestamp("2024-01-02")) & risk["ticker"].eq("A"))]

    with pytest.raises(MVPBacktestError):
        run_mvp_backtest(
            make_target_weights(),
            make_market_data(),
            config=make_backtest_config(),
            risk_estimates=risk,
            execution_cost_config=make_execution_config(),
        )


def test_trade_costs_match_standalone_cost_model() -> None:
    """Backtest trade costs must equal the standalone execution model."""
    risk = make_risk_estimates()

    execution_config = make_execution_config()

    outputs = run_mvp_backtest(
        make_target_weights(),
        make_market_data(),
        config=make_backtest_config(),
        risk_estimates=risk,
        execution_cost_config=execution_config,
    )

    enriched = estimate_trade_execution_costs(
        outputs.trades,
        risk,
        config=execution_config,
    )

    assert np.allclose(
        outputs.trades["transaction_cost"],
        enriched["total_execution_cost"],
        rtol=0.0,
        atol=1e-6,
    )


def test_future_risk_estimates_do_not_change_past_execution() -> None:
    """Risk information after signal dates must not affect the backtest."""
    first_risk = make_risk_estimates()

    future_rows = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp("2024-01-05")] * 4,
            "ticker": [
                "A",
                "B",
                "C",
                "D",
            ],
            "annualized_volatility": [
                10.0,
                10.0,
                10.0,
                10.0,
            ],
            "average_dollar_volume": [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        }
    )

    second_risk = pd.concat(
        [
            first_risk,
            future_rows,
        ],
        ignore_index=True,
    )

    first = run_mvp_backtest(
        make_target_weights(),
        make_market_data(),
        config=make_backtest_config(),
        risk_estimates=first_risk,
        execution_cost_config=make_execution_config(),
    )

    second = run_mvp_backtest(
        make_target_weights(),
        make_market_data(),
        config=make_backtest_config(),
        risk_estimates=second_risk,
        execution_cost_config=make_execution_config(),
    )

    assert np.allclose(
        first.daily_performance["portfolio_value"],
        second.daily_performance["portfolio_value"],
    )
