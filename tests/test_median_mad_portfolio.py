"""Tests for median-MAD portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_equity.portfolio import (
    MedianMadConfig,
    MedianMadPortfolioError,
    PortfolioOptimizerConfig,
    build_median_mad_portfolios,
    validate_median_mad_diagnostics,
)
from quant_equity.portfolio.median_mad import (
    _portfolio_statistics,
    _project_bounded_simplex,
)

FIRST_DATE = pd.Timestamp("2024-01-31")

SECOND_DATE = pd.Timestamp("2024-02-29")


def make_portfolio_config() -> PortfolioOptimizerConfig:
    """Create feasible portfolio constraints for tests."""
    return PortfolioOptimizerConfig(
        candidate_count=4,
        annualized_alpha_scale=0.10,
        risk_aversion=0.50,
        turnover_penalty=0.01,
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


def make_median_mad_config() -> MedianMadConfig:
    """Create compact optimization settings for tests."""
    return MedianMadConfig(
        lookback_days=60,
        minimum_observations=40,
        mad_limit=0.02,
        mad_violation_penalty=10.0,
        turnover_penalty=0.001,
        seed=42,
        max_iterations=4,
        population_size=4,
        tolerance=1e-6,
        mutation_min=0.5,
        mutation_max=1.0,
        recombination=0.7,
        polish=False,
    )


def make_signal() -> pd.DataFrame:
    """Create synthetic ranked alpha signals."""
    return pd.DataFrame(
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


def make_market_data() -> pd.DataFrame:
    """Create deterministic daily prices."""
    dates = pd.bdate_range(
        "2023-08-01",
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
        phase = ticker_index * 0.5

        daily_returns = 0.0003 + 0.006 * np.sin(np.arange(len(dates)) * 0.19 + phase)

        prices = 100.0 * np.cumprod(1.0 + daily_returns)

        for (
            date,
            price,
        ) in zip(
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


def make_risk_estimates() -> pd.DataFrame:
    """Create beta and liquidity estimates."""
    rows = []

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
                "as_of_date": FIRST_DATE,
                "ticker": ticker,
                "beta_vs_spy": beta,
                "average_dollar_volume": 1_000_000.0,
            }
        )

    return pd.DataFrame(rows)


def test_median_and_mad_match_manual_calculation() -> None:
    """Median and MAD must match their direct definitions."""
    returns = np.asarray(
        [
            [0.01, 0.00],
            [0.02, 0.01],
            [-0.01, 0.00],
            [0.00, 0.02],
        ]
    )

    weights = np.asarray(
        [
            0.50,
            0.50,
        ]
    )

    portfolio_returns = returns @ weights

    expected_median = float(np.median(portfolio_returns))

    expected_mad = float(np.mean(np.abs(portfolio_returns - expected_median)))

    median, mad = _portfolio_statistics(
        returns,
        weights,
    )

    assert median == pytest.approx(expected_median)

    assert mad == pytest.approx(expected_mad)


def test_bounded_simplex_projection_respects_limits() -> None:
    """Projection must remain fully invested and respect individual caps."""
    projected = _project_bounded_simplex(
        np.asarray(
            [
                10.0,
                2.0,
                1.0,
                0.0,
            ]
        ),
        np.asarray(
            [
                0.40,
                0.30,
                0.20,
                0.20,
            ]
        ),
    )

    assert projected.sum() == pytest.approx(1.0)

    assert (projected >= 0.0).all()

    assert (
        projected
        <= np.asarray(
            [
                0.40,
                0.30,
                0.20,
                0.20,
            ]
        )
        + 1e-9
    ).all()


def test_median_mad_portfolio_respects_constraints() -> None:
    """Constructed weights must satisfy portfolio constraints."""
    weights, diagnostics = build_median_mad_portfolios(
        make_signal(),
        make_market_data(),
        make_risk_estimates(),
        portfolio_config=make_portfolio_config(),
        median_mad_config=make_median_mad_config(),
    )

    assert weights["weight"].sum() == pytest.approx(
        1.0,
        abs=1e-6,
    )

    assert weights["weight"].max() <= (0.40 + 1e-6)

    sector_weights = weights.groupby("sector")["weight"].sum()

    assert sector_weights.max() <= (0.60 + 1e-6)

    assert diagnostics.loc[
        0,
        "portfolio_beta_vs_spy",
    ] <= (1.50 + 1e-6)


def test_future_market_data_does_not_change_past_weights() -> None:
    """Future prices must not influence a previous rebalance."""
    first_market = make_market_data()

    modified_market = first_market.copy()

    modified_market.loc[
        modified_market["date"].gt(FIRST_DATE),
        "adjusted_close",
    ] *= 10.0

    first_weights, _ = build_median_mad_portfolios(
        make_signal(),
        first_market,
        make_risk_estimates(),
        portfolio_config=make_portfolio_config(),
        median_mad_config=make_median_mad_config(),
    )

    second_weights, _ = build_median_mad_portfolios(
        make_signal(),
        modified_market,
        make_risk_estimates(),
        portfolio_config=make_portfolio_config(),
        median_mad_config=make_median_mad_config(),
    )

    assert np.allclose(
        first_weights["weight"],
        second_weights["weight"],
    )


def test_insufficient_history_is_rejected() -> None:
    """Construction must reject insufficient historical data."""
    market = make_market_data()

    recent_dates = sorted(market["date"].unique())[-20:]

    market = market.loc[market["date"].isin(recent_dates)]

    with pytest.raises(MedianMadPortfolioError):
        build_median_mad_portfolios(
            make_signal(),
            market,
            make_risk_estimates(),
            portfolio_config=make_portfolio_config(),
            median_mad_config=make_median_mad_config(),
        )


def test_median_mad_is_reproducible() -> None:
    """The configured random seed must make weights reproducible."""
    first_weights, _ = build_median_mad_portfolios(
        make_signal(),
        make_market_data(),
        make_risk_estimates(),
        portfolio_config=make_portfolio_config(),
        median_mad_config=make_median_mad_config(),
    )

    second_weights, _ = build_median_mad_portfolios(
        make_signal(),
        make_market_data(),
        make_risk_estimates(),
        portfolio_config=make_portfolio_config(),
        median_mad_config=make_median_mad_config(),
    )

    assert np.allclose(
        first_weights["weight"],
        second_weights["weight"],
    )


def test_median_mad_readiness_checks_pass() -> None:
    """Valid diagnostics must satisfy all hard readiness checks."""
    _, diagnostics = build_median_mad_portfolios(
        make_signal(),
        make_market_data(),
        make_risk_estimates(),
        portfolio_config=make_portfolio_config(),
        median_mad_config=make_median_mad_config(),
    )

    checks = validate_median_mad_diagnostics(
        diagnostics,
        portfolio_config=make_portfolio_config(),
        median_mad_config=make_median_mad_config(),
    )

    assert checks["status"].eq("PASS").all()
