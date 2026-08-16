"""Tests for constrained baseline portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_equity.portfolio import (
    BaselinePortfolioConfig,
    BaselinePortfolioConstructionError,
    build_equal_weight_portfolios,
    build_score_weighted_portfolios,
    compute_portfolio_diagnostics,
    validate_baseline_portfolios,
)

TEST_DATE = pd.Timestamp("2024-01-31")


def make_signal() -> pd.DataFrame:
    """Create a ranked cross-section with sector concentration."""
    return pd.DataFrame(
        {
            "as_of_date": [
                TEST_DATE,
            ]
            * 6,
            "ticker": [
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "EEE",
                "FFF",
            ],
            "sector": [
                "Technology",
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
                6,
            ],
            "percentile_score": [
                1.00,
                0.90,
                0.80,
                0.70,
                0.60,
                0.50,
            ],
        }
    )


def make_config() -> BaselinePortfolioConfig:
    """Create compact feasible constraints for tests."""
    return BaselinePortfolioConfig(
        candidate_count=4,
        equal_weight_positions=4,
        max_security_weight=0.30,
        max_sector_weight=0.50,
        minimum_positions=3,
        weight_tolerance=1e-8,
    )


def test_equal_weight_portfolio_is_fully_invested() -> None:
    """Equal-weight construction must allocate all capital."""
    weights = build_equal_weight_portfolios(
        make_signal(),
        config=make_config(),
    )

    assert len(weights) == 4

    assert weights["weight"].eq(0.25).all()

    assert weights["weight"].sum() == pytest.approx(1.0)


def test_equal_weight_selection_respects_sector_limit() -> None:
    """Sector-aware selection must skip excessive sector concentration."""
    weights = build_equal_weight_portfolios(
        make_signal(),
        config=make_config(),
    )

    technology_weight = weights.loc[
        weights["sector"].eq("Technology"),
        "weight",
    ].sum()

    assert technology_weight <= (0.50 + 1e-12)

    assert "DDD" in set(weights["ticker"])


def test_score_weighted_portfolio_is_fully_invested() -> None:
    """Projected score weights must sum exactly to one."""
    weights = build_score_weighted_portfolios(
        make_signal(),
        config=make_config(),
    )

    assert weights["weight"].sum() == pytest.approx(1.0)


def test_score_weighted_portfolio_respects_constraints() -> None:
    """Projected score weights must obey security and sector caps."""
    weights = build_score_weighted_portfolios(
        make_signal(),
        config=make_config(),
    )

    assert weights["weight"].max() <= (0.30 + 1e-8)

    sector_weights = weights.groupby("sector")["weight"].sum()

    assert sector_weights.max() <= (0.50 + 1e-8)


def test_diagnostics_report_expected_concentration() -> None:
    """Four equal positions must imply effective N equal to four."""
    weights = build_equal_weight_portfolios(
        make_signal(),
        config=make_config(),
    )

    diagnostics = compute_portfolio_diagnostics(weights)

    assert diagnostics.loc[
        0,
        "concentration_hhi",
    ] == pytest.approx(0.25)

    assert diagnostics.loc[
        0,
        "effective_positions",
    ] == pytest.approx(4.0)


def test_infeasible_security_cap_is_rejected() -> None:
    """Too little aggregate capacity must fail configuration validation."""
    config = BaselinePortfolioConfig(
        candidate_count=4,
        equal_weight_positions=4,
        max_security_weight=0.20,
        max_sector_weight=0.50,
        minimum_positions=3,
    )

    with pytest.raises(BaselinePortfolioConstructionError):
        config.validate()


def test_readiness_checks_pass_for_both_methods() -> None:
    """Valid baseline methods must satisfy all stored constraints."""
    signal = make_signal()
    config = make_config()

    weights = pd.concat(
        [
            build_equal_weight_portfolios(
                signal,
                config=config,
            ),
            build_score_weighted_portfolios(
                signal,
                config=config,
            ),
        ],
        ignore_index=True,
    )

    checks = validate_baseline_portfolios(
        weights,
        config=config,
    )

    assert checks["status"].eq("PASS").all()


def test_unused_future_columns_do_not_change_weights() -> None:
    """Future outcomes stored elsewhere must not affect target weights."""
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

    first = build_score_weighted_portfolios(
        first_signal,
        config=make_config(),
    )

    second = build_score_weighted_portfolios(
        second_signal,
        config=make_config(),
    )

    assert np.allclose(
        first["weight"],
        second["weight"],
    )
