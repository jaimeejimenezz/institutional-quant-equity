"""Tests for Conditional Value-at-Risk portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_equity.portfolio import (
    CvarPortfolioError,
    CvarRiskConfig,
    PortfolioOptimizerConfig,
    build_cvar_portfolios,
    validate_cvar_diagnostics,
)

FIRST_DATE = pd.Timestamp("2024-01-31")

SECOND_DATE = pd.Timestamp("2024-02-29")


def make_portfolio_config(
    *,
    turnover_penalty: float = 0.01,
) -> PortfolioOptimizerConfig:
    """Create feasible portfolio constraints for tests."""
    return PortfolioOptimizerConfig(
        candidate_count=4,
        annualized_alpha_scale=0.10,
        risk_aversion=0.50,
        turnover_penalty=turnover_penalty,
        max_security_weight=0.40,
        max_sector_weight=0.60,
        weight_tolerance=1e-8,
        solver_tolerance=1e-9,
        constraint_margin=1e-6,
        minimum_portfolio_beta=0.50,
        maximum_portfolio_beta=1.50,
        reference_portfolio_value=100.0,
        max_position_adv_fraction=0.50,
    )


def make_cvar_config() -> CvarRiskConfig:
    """Create compact CVaR settings for tests."""
    return CvarRiskConfig(
        confidence_level=0.95,
        horizon_days=5,
        scenario_lookback=60,
        minimum_scenarios=30,
        cvar_penalty=0.05,
    )


def make_signal(
    *,
    two_dates: bool = False,
) -> pd.DataFrame:
    """Create synthetic ranked alpha signals."""
    first = pd.DataFrame(
        {
            "as_of_date": [
                FIRST_DATE,
            ]
            * 5,
            "ticker": [
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "EEE",
            ],
            "sector": [
                "Technology",
                "Technology",
                "Financials",
                "Healthcare",
                "Industrials",
            ],
            "rank": [
                1,
                2,
                3,
                4,
                5,
            ],
            "percentile_score": [
                1.00,
                0.90,
                0.80,
                0.70,
                0.60,
            ],
        }
    )

    if not two_dates:
        return first

    second = first.copy()

    second["as_of_date"] = SECOND_DATE

    second["rank"] = [
        4,
        3,
        2,
        1,
        5,
    ]

    second["percentile_score"] = [
        0.70,
        0.80,
        0.90,
        1.00,
        0.60,
    ]

    return pd.concat(
        [
            first,
            second,
        ],
        ignore_index=True,
    )


def make_market_data() -> pd.DataFrame:
    """Create deterministic daily prices with both gains and losses."""
    dates = pd.bdate_range(
        "2023-05-01",
        SECOND_DATE,
    )

    tickers = [
        "AAA",
        "BBB",
        "CCC",
        "DDD",
        "EEE",
    ]

    rows = []

    for ticker_index, ticker in enumerate(tickers):
        phase = ticker_index * 0.7

        daily_returns = 0.0003 + 0.008 * np.sin(np.arange(len(dates)) * 0.21 + phase)

        prices = 100.0 * np.cumprod(1.0 + daily_returns)

        for date, price in zip(
            dates,
            prices,
            strict=True,
        ):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adjusted_close": price,
                }
            )

    return pd.DataFrame(rows)


def make_risk_estimates(
    *,
    two_dates: bool = False,
) -> pd.DataFrame:
    """Create beta and liquidity estimates."""
    dates = (
        [
            FIRST_DATE,
            SECOND_DATE,
        ]
        if two_dates
        else [
            FIRST_DATE,
        ]
    )

    rows = []

    for date in dates:
        for ticker, beta in zip(
            [
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "EEE",
            ],
            [
                1.10,
                1.05,
                1.00,
                0.95,
                0.90,
            ],
            strict=True,
        ):
            rows.append(
                {
                    "as_of_date": date,
                    "ticker": ticker,
                    "beta_vs_spy": beta,
                    "average_dollar_volume": 1_000_000.0,
                }
            )

    return pd.DataFrame(rows)


def test_cvar_portfolio_is_fully_invested() -> None:
    """CVaR portfolio must allocate all capital."""
    weights, diagnostics = build_cvar_portfolios(
        make_signal(),
        make_market_data(),
        make_risk_estimates(),
        portfolio_config=make_portfolio_config(),
        cvar_config=make_cvar_config(),
    )

    assert weights["weight"].sum() == pytest.approx(
        1.0,
        abs=1e-7,
    )

    assert np.isfinite(
        diagnostics.loc[
            0,
            "cvar_loss",
        ]
    )


def test_cvar_portfolio_respects_constraints() -> None:
    """CVaR weights must satisfy security and sector constraints."""
    weights, diagnostics = build_cvar_portfolios(
        make_signal(),
        make_market_data(),
        make_risk_estimates(),
        portfolio_config=make_portfolio_config(),
        cvar_config=make_cvar_config(),
    )

    assert weights["weight"].max() <= (0.40 + 1e-8)

    sector_weights = weights.groupby("sector")["weight"].sum()

    assert sector_weights.max() <= (0.60 + 1e-8)

    assert (
        diagnostics.loc[
            0,
            "portfolio_beta_vs_spy",
        ]
        <= 1.50 + 1e-8
    )


def test_cvar_is_not_below_var() -> None:
    """Expected tail loss must not be smaller than its VaR threshold."""
    _, diagnostics = build_cvar_portfolios(
        make_signal(),
        make_market_data(),
        make_risk_estimates(),
        portfolio_config=make_portfolio_config(),
        cvar_config=make_cvar_config(),
    )

    assert diagnostics.loc[
        0,
        "cvar_loss",
    ] >= (
        diagnostics.loc[
            0,
            "var_loss",
        ]
        - 1e-8
    )


def test_future_market_data_does_not_change_past_weights() -> None:
    """Prices after the rebalance date must not influence prior weights."""
    first_market = make_market_data()

    second_market = first_market.copy()

    second_market.loc[
        second_market["date"].gt(FIRST_DATE),
        "adjusted_close",
    ] *= 10.0

    first_weights, _ = build_cvar_portfolios(
        make_signal(),
        first_market,
        make_risk_estimates(),
        portfolio_config=make_portfolio_config(),
        cvar_config=make_cvar_config(),
    )

    second_weights, _ = build_cvar_portfolios(
        make_signal(),
        second_market,
        make_risk_estimates(),
        portfolio_config=make_portfolio_config(),
        cvar_config=make_cvar_config(),
    )

    assert np.allclose(
        first_weights["weight"],
        second_weights["weight"],
    )


def test_insufficient_scenario_history_is_rejected() -> None:
    """CVaR construction must reject insufficient historical scenarios."""
    market = make_market_data()

    recent_dates = sorted(market["date"].unique())[-15:]

    market = market.loc[market["date"].isin(recent_dates)]

    with pytest.raises(CvarPortfolioError):
        build_cvar_portfolios(
            make_signal(),
            market,
            make_risk_estimates(),
            portfolio_config=make_portfolio_config(),
            cvar_config=make_cvar_config(),
        )


def test_cvar_readiness_checks_pass() -> None:
    """Valid CVaR diagnostics must satisfy all specialized checks."""
    _, diagnostics = build_cvar_portfolios(
        make_signal(),
        make_market_data(),
        make_risk_estimates(),
        portfolio_config=make_portfolio_config(),
        cvar_config=make_cvar_config(),
    )

    checks = validate_cvar_diagnostics(
        diagnostics,
        portfolio_config=make_portfolio_config(),
        cvar_config=make_cvar_config(),
    )

    assert checks["status"].eq("PASS").all()


def test_cvar_records_successful_solver() -> None:
    """CVaR diagnostics must record the solver used."""
    _, diagnostics = build_cvar_portfolios(
        make_signal(),
        make_market_data(),
        make_risk_estimates(),
        portfolio_config=make_portfolio_config(),
        cvar_config=make_cvar_config(),
    )

    assert diagnostics.loc[
        0,
        "solver",
    ] in {
        "CLARABEL",
        "SCS",
    }

    assert diagnostics.loc[
        0,
        "solver_status",
    ] in {
        "optimal",
        "optimal_inaccurate",
    }


def test_cvar_weights_are_normalized_to_one() -> None:
    """Stored CVaR weights must sum to one after solver cleanup."""
    weights, _ = build_cvar_portfolios(
        make_signal(),
        make_market_data(),
        make_risk_estimates(),
        portfolio_config=make_portfolio_config(),
        cvar_config=make_cvar_config(),
    )

    assert weights["weight"].sum() == pytest.approx(
        1.0,
        abs=1e-10,
    )
