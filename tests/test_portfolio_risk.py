"""Tests for portfolio-level risk analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_equity.risk import (
    PortfolioRiskConfig,
    PortfolioRiskError,
    build_top_n_equal_weights,
    calculate_portfolio_risk,
    validate_portfolio_risk,
)

TEST_DATE = pd.Timestamp("2024-01-31")


def make_portfolio_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create a small portfolio with known risk properties."""
    weights = pd.DataFrame(
        {
            "as_of_date": [
                TEST_DATE,
                TEST_DATE,
            ],
            "ticker": [
                "AAA",
                "BBB",
            ],
            "weight": [
                0.5,
                0.5,
            ],
        }
    )

    estimates = pd.DataFrame(
        {
            "as_of_date": [
                TEST_DATE,
                TEST_DATE,
            ],
            "ticker": [
                "AAA",
                "BBB",
            ],
            "sector": [
                "Technology",
                "Financials",
            ],
            "annualized_volatility": [
                0.20,
                0.30,
            ],
            "beta_vs_spy": [
                1.20,
                0.80,
            ],
            "average_dollar_volume": [
                100_000_000.0,
                50_000_000.0,
            ],
        }
    )

    covariance = pd.DataFrame(
        {
            "as_of_date": [
                TEST_DATE,
                TEST_DATE,
                TEST_DATE,
                TEST_DATE,
            ],
            "ticker_a": [
                "AAA",
                "AAA",
                "BBB",
                "BBB",
            ],
            "ticker_b": [
                "AAA",
                "BBB",
                "AAA",
                "BBB",
            ],
            "annualized_covariance": [
                0.04,
                0.01,
                0.01,
                0.09,
            ],
        }
    )

    return (
        weights,
        estimates,
        covariance,
    )


def test_portfolio_volatility_matches_manual_calculation() -> None:
    """Matrix portfolio volatility must match the manual formula."""
    (
        weights,
        estimates,
        covariance,
    ) = make_portfolio_inputs()

    summary, _, _ = calculate_portfolio_risk(
        weights,
        estimates,
        covariance,
    )

    expected_variance = 0.25 * 0.04 + 0.25 * 0.09 + 2.0 * 0.25 * 0.01

    expected_volatility = np.sqrt(expected_variance)

    assert summary.loc[
        0,
        "predicted_volatility",
    ] == pytest.approx(expected_volatility)


def test_portfolio_beta_is_weighted_asset_beta() -> None:
    """Portfolio beta must equal the weighted security betas."""
    (
        weights,
        estimates,
        covariance,
    ) = make_portfolio_inputs()

    summary, _, _ = calculate_portfolio_risk(
        weights,
        estimates,
        covariance,
    )

    assert summary.loc[
        0,
        "portfolio_beta_vs_spy",
    ] == pytest.approx(1.0)


def test_risk_contributions_reconstruct_portfolio_volatility() -> None:
    """Euler component contributions must sum to total volatility."""
    (
        weights,
        estimates,
        covariance,
    ) = make_portfolio_inputs()

    summary, contributions, _ = calculate_portfolio_risk(
        weights,
        estimates,
        covariance,
    )

    assert contributions["component_risk"].sum() == pytest.approx(
        summary.loc[
            0,
            "predicted_volatility",
        ]
    )

    assert contributions["risk_contribution_share"].sum() == pytest.approx(1.0)


def test_sector_exposures_reconstruct_portfolio() -> None:
    """Sector weights must sum to the fully invested portfolio."""
    (
        weights,
        estimates,
        covariance,
    ) = make_portfolio_inputs()

    _, _, sectors = calculate_portfolio_risk(
        weights,
        estimates,
        covariance,
    )

    assert sectors["portfolio_weight"].sum() == pytest.approx(1.0)


def test_equal_weight_concentration_has_two_effective_positions() -> None:
    """A two-security equal-weight portfolio has effective N equal to two."""
    (
        weights,
        estimates,
        covariance,
    ) = make_portfolio_inputs()

    summary, _, _ = calculate_portfolio_risk(
        weights,
        estimates,
        covariance,
    )

    assert summary.loc[
        0,
        "concentration_hhi",
    ] == pytest.approx(0.5)

    assert summary.loc[
        0,
        "effective_positions",
    ] == pytest.approx(2.0)


def test_invalid_weight_sum_is_rejected() -> None:
    """A partially invested portfolio must fail validation."""
    (
        weights,
        estimates,
        covariance,
    ) = make_portfolio_inputs()

    weights.loc[
        :,
        "weight",
    ] = [
        0.4,
        0.4,
    ]

    with pytest.raises(PortfolioRiskError):
        calculate_portfolio_risk(
            weights,
            estimates,
            covariance,
        )


def test_portfolio_risk_readiness_checks_pass() -> None:
    """Valid portfolio-risk artifacts must pass readiness checks."""
    (
        weights,
        estimates,
        covariance,
    ) = make_portfolio_inputs()

    (
        summary,
        contributions,
        sectors,
    ) = calculate_portfolio_risk(
        weights,
        estimates,
        covariance,
        config=PortfolioRiskConfig(
            portfolio_value=1_000_000.0,
            max_daily_adv_participation=0.10,
        ),
    )

    checks = validate_portfolio_risk(
        summary,
        contributions,
        sectors,
    )

    assert checks["status"].eq("PASS").all()


def test_top_n_reference_weights_are_equal_and_fully_invested() -> None:
    """Reference top-N portfolios must use transparent equal weights."""
    signal = pd.DataFrame(
        {
            "as_of_date": [
                TEST_DATE,
            ]
            * 4,
            "ticker": [
                "AAA",
                "BBB",
                "CCC",
                "DDD",
            ],
            "rank": [
                1,
                2,
                3,
                4,
            ],
        }
    )

    weights = build_top_n_equal_weights(
        signal,
        top_n=2,
    )

    assert len(weights) == 2

    assert weights["weight"].eq(0.5).all()

    assert weights["weight"].sum() == pytest.approx(1.0)
