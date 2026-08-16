"""Tests for transaction execution costs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_equity.backtest import (
    ExecutionCostConfig,
    ExecutionCostError,
    estimate_trade_execution_cost,
    estimate_trade_execution_costs,
)


def make_config(
    *,
    market_impact_coefficient: float = 0.10,
) -> ExecutionCostConfig:
    """Create deterministic execution-cost settings."""
    return ExecutionCostConfig(
        commission_bps=0.5,
        half_spread_bps=2.0,
        slippage_bps=2.5,
        market_impact_coefficient=(market_impact_coefficient),
        annualization_factor=252,
    )


def test_linear_execution_cost_matches_manual_calculation() -> None:
    """Linear cost components must match their basis-point definitions."""
    result = estimate_trade_execution_cost(
        100_000.0,
        0.20,
        100_000_000.0,
        config=make_config(market_impact_coefficient=0.0),
    )

    assert result["commission_cost"] == pytest.approx(5.0)

    assert result["spread_cost"] == pytest.approx(20.0)

    assert result["slippage_cost"] == pytest.approx(25.0)

    assert result["total_execution_cost"] == pytest.approx(50.0)

    assert result["effective_cost_bps"] == pytest.approx(5.0)


def test_zero_trade_has_zero_execution_cost() -> None:
    """A zero-notional trade must generate no cost."""
    result = estimate_trade_execution_cost(
        0.0,
        0.20,
        100_000_000.0,
        config=make_config(),
    )

    assert result["total_execution_cost"] == 0.0

    assert result["market_impact_cost"] == 0.0

    assert result["effective_cost_bps"] == 0.0


def test_market_impact_matches_manual_formula() -> None:
    """Market impact must follow volatility and square-root participation."""
    notional = 1_000_000.0
    annualized_volatility = 0.252
    average_dollar_volume = 100_000_000.0

    config = make_config(market_impact_coefficient=0.10)

    result = estimate_trade_execution_cost(
        notional,
        annualized_volatility,
        average_dollar_volume,
        config=config,
    )

    expected_daily_volatility = annualized_volatility / np.sqrt(252.0)

    expected_participation = notional / average_dollar_volume

    expected_impact_rate = 0.10 * expected_daily_volatility * np.sqrt(expected_participation)

    expected_impact_cost = notional * expected_impact_rate

    assert result["order_adv_fraction"] == pytest.approx(expected_participation)

    assert result["market_impact_cost"] == pytest.approx(expected_impact_cost)


def test_market_impact_increases_with_order_size() -> None:
    """Larger orders must produce higher market-impact costs."""
    config = make_config()

    small = estimate_trade_execution_cost(
        100_000.0,
        0.25,
        100_000_000.0,
        config=config,
    )

    large = estimate_trade_execution_cost(
        5_000_000.0,
        0.25,
        100_000_000.0,
        config=config,
    )

    assert large["market_impact_bps"] > small["market_impact_bps"]


def test_market_impact_decreases_with_liquidity() -> None:
    """More liquid securities must have lower estimated market impact."""
    config = make_config()

    less_liquid = estimate_trade_execution_cost(
        1_000_000.0,
        0.25,
        20_000_000.0,
        config=config,
    )

    more_liquid = estimate_trade_execution_cost(
        1_000_000.0,
        0.25,
        500_000_000.0,
        config=config,
    )

    assert less_liquid["market_impact_bps"] > more_liquid["market_impact_bps"]


def test_invalid_execution_inputs_are_rejected() -> None:
    """Execution cost estimation must reject invalid values."""
    with pytest.raises(ExecutionCostError):
        estimate_trade_execution_cost(
            -1.0,
            0.20,
            100_000_000.0,
            config=make_config(),
        )

    with pytest.raises(ExecutionCostError):
        estimate_trade_execution_cost(
            1_000.0,
            0.20,
            0.0,
            config=make_config(),
        )


def test_trade_table_uses_point_in_time_risk_estimates() -> None:
    """Trade costs must join risk estimates by signal date and ticker."""
    signal_date = pd.Timestamp("2024-01-31")

    trades = pd.DataFrame(
        {
            "signal_date": [
                signal_date,
            ],
            "execution_date": [
                pd.Timestamp("2024-02-01"),
            ],
            "ticker": [
                "AAA",
            ],
            "absolute_trade_notional": [
                100_000.0,
            ],
        }
    )

    risk_estimates = pd.DataFrame(
        {
            "as_of_date": [
                signal_date,
            ],
            "ticker": [
                "AAA",
            ],
            "annualized_volatility": [
                0.20,
            ],
            "average_dollar_volume": [
                100_000_000.0,
            ],
        }
    )

    result = estimate_trade_execution_costs(
        trades,
        risk_estimates,
        config=make_config(),
    )

    assert len(result) == 1

    assert result.loc[
        0,
        "annualized_volatility",
    ] == pytest.approx(0.20)

    assert result.loc[
        0,
        "average_dollar_volume",
    ] == pytest.approx(100_000_000.0)

    assert (
        result.loc[
            0,
            "total_execution_cost",
        ]
        > 0.0
    )
