"""Tests for risk-aware portfolio optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_equity.portfolio import (
    PortfolioOptimizationError,
    PortfolioOptimizerConfig,
    build_alpha_risk_turnover_portfolios,
)

FIRST_DATE = pd.Timestamp("2024-01-31")

SECOND_DATE = pd.Timestamp("2024-02-29")


def make_config(
    *,
    risk_aversion: float = 0.50,
    turnover_penalty: float = 0.01,
) -> PortfolioOptimizerConfig:
    """Create a compact feasible optimizer configuration."""
    return PortfolioOptimizerConfig(
        candidate_count=4,
        annualized_alpha_scale=0.10,
        risk_aversion=risk_aversion,
        turnover_penalty=turnover_penalty,
        max_security_weight=0.40,
        max_sector_weight=0.60,
        weight_tolerance=1e-8,
    )


def make_signal(
    *,
    two_dates: bool = False,
) -> pd.DataFrame:
    """Create synthetic alpha rankings."""
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


def make_covariance(
    *,
    two_dates: bool = False,
) -> pd.DataFrame:
    """Create positive-semidefinite covariance matrices."""
    tickers = [
        "AAA",
        "BBB",
        "CCC",
        "DDD",
        "EEE",
    ]

    rows = []

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

    for date in dates:
        for first in tickers:
            for second in tickers:
                covariance = 0.040 if first == second else 0.010

                rows.append(
                    {
                        "as_of_date": date,
                        "ticker_i": first,
                        "ticker_j": second,
                        "covariance": covariance,
                    }
                )

    return pd.DataFrame(rows)


def test_optimizer_is_fully_invested() -> None:
    """Optimized weights must sum to one."""
    weights, _ = build_alpha_risk_turnover_portfolios(
        make_signal(),
        make_covariance(),
        config=make_config(),
    )

    assert weights["weight"].sum() == pytest.approx(
        1.0,
        abs=1e-7,
    )


def test_optimizer_respects_security_and_sector_caps() -> None:
    """Optimized portfolios must respect stored constraints."""
    weights, _ = build_alpha_risk_turnover_portfolios(
        make_signal(),
        make_covariance(),
        config=make_config(),
    )

    assert weights["weight"].max() <= (0.40 + 1e-8)

    sector_weights = weights.groupby("sector")["weight"].sum()

    assert sector_weights.max() <= (0.60 + 1e-8)


def test_optimizer_prefers_stronger_alpha_when_risk_is_equal() -> None:
    """Higher alpha should receive no less weight under equal risk."""
    weights, _ = build_alpha_risk_turnover_portfolios(
        make_signal(),
        make_covariance(),
        config=make_config(
            risk_aversion=0.10,
            turnover_penalty=0.0,
        ),
    )

    ordered = weights.sort_values(
        "percentile_score",
        ascending=False,
    )

    assert ordered.iloc[0]["weight"] >= ordered.iloc[-1]["weight"]


def test_turnover_penalty_reduces_portfolio_changes() -> None:
    """A turnover penalty should keep consecutive portfolios closer."""
    signal = make_signal(two_dates=True)

    covariance = make_covariance(two_dates=True)

    _, no_penalty = build_alpha_risk_turnover_portfolios(
        signal,
        covariance,
        config=make_config(
            risk_aversion=0.10,
            turnover_penalty=0.0,
        ),
    )

    _, with_penalty = build_alpha_risk_turnover_portfolios(
        signal,
        covariance,
        config=make_config(
            risk_aversion=0.10,
            turnover_penalty=0.10,
        ),
    )

    no_penalty_turnover = no_penalty.loc[
        no_penalty["as_of_date"].eq(SECOND_DATE),
        "one_way_turnover",
    ].iloc[0]

    penalized_turnover = with_penalty.loc[
        with_penalty["as_of_date"].eq(SECOND_DATE),
        "one_way_turnover",
    ].iloc[0]

    assert penalized_turnover <= no_penalty_turnover + 1e-7


def test_future_columns_do_not_change_weights() -> None:
    """Unused future outcomes must not influence optimization."""
    first_signal = make_signal()

    second_signal = first_signal.copy()

    first_signal["future_return"] = np.arange(
        len(first_signal),
        dtype=float,
    )

    second_signal["future_return"] = np.arange(
        len(second_signal),
        dtype=float,
    )[::-1]

    first_weights, _ = build_alpha_risk_turnover_portfolios(
        first_signal,
        make_covariance(),
        config=make_config(),
    )

    second_weights, _ = build_alpha_risk_turnover_portfolios(
        second_signal,
        make_covariance(),
        config=make_config(),
    )

    assert np.allclose(
        first_weights["weight"],
        second_weights["weight"],
    )


def test_incomplete_covariance_is_rejected() -> None:
    """Optimization must fail when required covariance coverage is incomplete."""
    covariance = make_covariance()

    covariance = covariance.loc[
        ~(covariance["ticker_i"].eq("AAA") & covariance["ticker_j"].eq("AAA"))
    ].copy()

    with pytest.raises(PortfolioOptimizationError):
        build_alpha_risk_turnover_portfolios(
            make_signal(),
            covariance,
            config=make_config(),
        )


def test_optimizer_accepts_project_covariance_schema() -> None:
    """Optimizer must accept the covariance schema stored by the risk model."""
    covariance = make_covariance().rename(
        columns={
            "ticker_i": "ticker_a",
            "ticker_j": "ticker_b",
            "covariance": "annualized_covariance",
        }
    )

    covariance["correlation"] = np.nan

    weights, _ = build_alpha_risk_turnover_portfolios(
        make_signal(),
        covariance,
        config=make_config(),
    )

    assert weights["weight"].sum() == pytest.approx(
        1.0,
        abs=1e-7,
    )
