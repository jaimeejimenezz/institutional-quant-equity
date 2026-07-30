"""Tests for MVP portfolio construction."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_equity.portfolio import (
    PortfolioConstructionConfig,
    PortfolioConstructionError,
    build_mvp_target_portfolios,
)


def make_config() -> PortfolioConstructionConfig:
    """Create a compact test configuration."""
    return PortfolioConstructionConfig(
        primary_model_name="elastic_net",
        challenger_model_name="ridge",
        momentum_model_name="momentum_3m",
        top_n=20,
        score_weighted_candidate_count=25,
        max_weight=0.05,
        max_sector_weight=0.25,
        minimum_cross_section_size=30,
        weight_tolerance=1.0e-8,
        optimization_tolerance=1.0e-10,
        optimization_max_iterations=2000,
    )


def make_predictions() -> pd.DataFrame:
    """Create aligned synthetic model predictions."""
    dates = pd.to_datetime(
        [
            "2024-01-31",
            "2024-02-29",
        ]
    )

    tickers = [f"T{number:02d}" for number in range(30)]

    sectors = (
        ["Sector A"] * 10
        + ["Sector B"] * 5
        + ["Sector C"] * 5
        + ["Sector D"] * 5
        + ["Sector E"] * 5
    )

    rows: list[dict[str, object]] = []

    for date_number, as_of_date in enumerate(dates):
        for model_name, model_offset in (
            ("elastic_net", 0.30),
            ("ridge", 0.20),
            ("momentum_3m", 0.10),
        ):
            for ticker_number, (
                ticker,
                sector,
            ) in enumerate(
                zip(
                    tickers,
                    sectors,
                    strict=True,
                )
            ):
                prediction = 1.0 - ticker_number * 0.02 + model_offset + date_number * 0.001

                rows.append(
                    {
                        "as_of_date": as_of_date,
                        "ticker": ticker,
                        "sector": sector,
                        "model_name": model_name,
                        "prediction": prediction,
                    }
                )

    return pd.DataFrame(rows)


def test_builds_expected_strategies() -> None:
    """All MVP strategies should be generated."""
    outputs = build_mvp_target_portfolios(
        make_predictions(),
        config=make_config(),
    )

    strategies = set(outputs.target_weights["strategy_name"].unique())

    assert strategies == {
        "universe_equal_weight",
        "momentum_3m_top20_equal_weight",
        "elastic_net_top20_equal_weight",
        "ridge_top20_equal_weight",
        "elastic_net_score_weighted",
    }


def test_all_portfolios_are_fully_invested() -> None:
    """Every strategy-date portfolio should sum to one."""
    outputs = build_mvp_target_portfolios(
        make_predictions(),
        config=make_config(),
    )

    weight_sums = outputs.target_weights.groupby(
        [
            "strategy_name",
            "as_of_date",
        ]
    )["target_weight"].sum()

    assert (weight_sums.sub(1.0).abs() <= 1.0e-8).all()


def test_active_portfolios_respect_caps() -> None:
    """Active strategies should respect asset and sector caps."""
    outputs = build_mvp_target_portfolios(
        make_predictions(),
        config=make_config(),
    )

    active_checks = outputs.constraint_checks.loc[
        outputs.constraint_checks["constraint_set"].eq("active_long_only")
    ]

    assert active_checks["asset_cap_ok"].all()

    assert active_checks["sector_cap_ok"].all()

    assert active_checks["constraints_pass"].all()


def test_top_equal_weight_has_twenty_holdings() -> None:
    """Top-N equal-weight strategies should hold twenty names."""
    outputs = build_mvp_target_portfolios(
        make_predictions(),
        config=make_config(),
    )

    top_equal_weight = outputs.constraint_checks.loc[
        outputs.constraint_checks["strategy_name"].str.endswith("top20_equal_weight")
    ]

    assert top_equal_weight["holdings"].eq(20).all()

    assert (top_equal_weight["maximum_weight"].sub(0.05).abs() <= 1.0e-8).all()


def test_sector_selection_skips_excess_names() -> None:
    """A highly ranked sector cannot exceed five holdings."""
    outputs = build_mvp_target_portfolios(
        make_predictions(),
        config=make_config(),
    )

    strategy = outputs.target_weights.loc[
        outputs.target_weights["strategy_name"].eq("elastic_net_top20_equal_weight")
        & outputs.target_weights["as_of_date"].eq(pd.Timestamp("2024-01-31"))
        & outputs.target_weights["selected"]
    ]

    sector_counts = strategy.groupby("sector")["ticker"].count()

    assert int(sector_counts.max()) <= 5


def test_score_weighted_strategy_uses_valid_holding_count() -> None:
    """Score-weighted strategy should hold between 20 and 25 names."""
    outputs = build_mvp_target_portfolios(
        make_predictions(),
        config=make_config(),
    )

    score_checks = outputs.constraint_checks.loc[
        outputs.constraint_checks["strategy_name"].eq("elastic_net_score_weighted")
    ]

    assert (
        score_checks["holdings"]
        .between(
            20,
            25,
        )
        .all()
    )


def test_invalid_top_n_is_rejected() -> None:
    """Too few holdings cannot satisfy a five-percent cap."""
    with pytest.raises(
        PortfolioConstructionError,
        match="top_n is too small",
    ):
        PortfolioConstructionConfig(
            top_n=10,
            score_weighted_candidate_count=25,
            max_weight=0.05,
            max_sector_weight=0.25,
            minimum_cross_section_size=30,
        ).validate()
