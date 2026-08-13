"""Tests for the monthly fundamental input base."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_equity.features import (
    FundamentalBaseConfig,
    FundamentalBaseError,
    build_monthly_fundamental_base,
)


def make_config() -> FundamentalBaseConfig:
    """Create a small test configuration."""
    return FundamentalBaseConfig(
        ttm_metrics=(
            "revenue",
            "net_income",
        ),
        require_exact_market_date=True,
    )


def make_universe() -> pd.DataFrame:
    """Create a two-company universe."""
    return pd.DataFrame(
        {
            "ticker": [
                "AAPL",
                "MSFT",
            ],
            "company_name": [
                "Apple Inc.",
                "Microsoft Corp.",
            ],
            "sector": [
                "Technology",
                "Technology",
            ],
            "industry": [
                "Hardware",
                "Software",
            ],
            "cik": [
                "0000320193",
                "0000789019",
            ],
            "start_date": [
                "2020-01-01",
                "2020-01-01",
            ],
            "end_date": [
                None,
                None,
            ],
        }
    )


def make_calendar() -> pd.DataFrame:
    """Create one rebalance date."""
    return pd.DataFrame(
        {
            "as_of_date": [
                "2024-03-28",
            ]
        }
    )


def make_market() -> pd.DataFrame:
    """Create exact rebalance-date prices."""
    return pd.DataFrame(
        {
            "date": [
                "2024-03-28",
                "2024-03-28",
            ],
            "ticker": [
                "AAPL",
                "MSFT",
            ],
            "close": [
                170.0,
                420.0,
            ],
        }
    )


def make_pit() -> pd.DataFrame:
    """Create point-in-time balance facts."""
    return pd.DataFrame(
        {
            "as_of_date": [
                "2024-03-28",
                "2024-03-28",
            ],
            "ticker": [
                "AAPL",
                "MSFT",
            ],
            "canonical_metric": [
                "assets",
                "assets",
            ],
            "value": [
                350.0,
                410.0,
            ],
            "statement_type": [
                "instant",
                "instant",
            ],
            "available_date": [
                "2024-02-05",
                "2024-01-31",
            ],
        }
    )


def make_ttm() -> pd.DataFrame:
    """Create TTM facts."""
    rows = []

    for ticker, revenue, income in (
        (
            "AAPL",
            400.0,
            100.0,
        ),
        (
            "MSFT",
            220.0,
            80.0,
        ),
    ):
        rows.extend(
            [
                {
                    "as_of_date": ("2024-03-28"),
                    "ticker": ticker,
                    "canonical_metric": ("revenue"),
                    "ttm_value": revenue,
                    "latest_component_available_date": ("2024-02-05"),
                },
                {
                    "as_of_date": ("2024-03-28"),
                    "ticker": ticker,
                    "canonical_metric": ("net_income"),
                    "ttm_value": income,
                    "latest_component_available_date": ("2024-02-05"),
                },
            ]
        )

    return pd.DataFrame(rows)


def test_base_contains_balance_ttm_and_price() -> None:
    """All fundamental input families should be joined."""
    result = build_monthly_fundamental_base(
        pit_snapshots=make_pit(),
        ttm_snapshots=make_ttm(),
        market_daily=make_market(),
        universe=make_universe(),
        rebalance_calendar=make_calendar(),
        config=make_config(),
    )

    apple = result.loc[result["ticker"].eq("AAPL")].iloc[0]

    assert len(result) == 2

    assert apple["close_price"] == 170.0

    assert apple["assets"] == 350.0

    assert apple["revenue_ttm"] == 400.0

    assert apple["net_income_ttm"] == 100.0


def test_missing_metric_does_not_drop_company() -> None:
    """Missing accounting data should remain missing."""
    ttm = make_ttm()

    ttm = ttm.loc[~(ttm["ticker"].eq("MSFT") & ttm["canonical_metric"].eq("net_income"))].copy()

    result = build_monthly_fundamental_base(
        pit_snapshots=make_pit(),
        ttm_snapshots=ttm,
        market_daily=make_market(),
        universe=make_universe(),
        rebalance_calendar=make_calendar(),
        config=make_config(),
    )

    microsoft = result.loc[result["ticker"].eq("MSFT")].iloc[0]

    assert pd.isna(microsoft["net_income_ttm"])


def test_future_pit_information_is_rejected() -> None:
    """Future balance information must not enter the base."""
    pit = make_pit()

    pit.loc[
        pit["ticker"].eq("AAPL"),
        "available_date",
    ] = "2024-04-01"

    with pytest.raises(
        FundamentalBaseError,
        match="future information",
    ):
        build_monthly_fundamental_base(
            pit_snapshots=pit,
            ttm_snapshots=make_ttm(),
            market_daily=make_market(),
            universe=make_universe(),
            rebalance_calendar=make_calendar(),
            config=make_config(),
        )


def test_future_ttm_information_is_rejected() -> None:
    """Future TTM components must not enter the base."""
    ttm = make_ttm()

    ttm.loc[
        ttm["ticker"].eq("AAPL"),
        "latest_component_available_date",
    ] = "2024-04-01"

    with pytest.raises(
        FundamentalBaseError,
        match="future information",
    ):
        build_monthly_fundamental_base(
            pit_snapshots=make_pit(),
            ttm_snapshots=ttm,
            market_daily=make_market(),
            universe=make_universe(),
            rebalance_calendar=make_calendar(),
            config=make_config(),
        )


def test_exact_market_price_is_required() -> None:
    """Every active company needs its rebalance close."""
    market = make_market()

    market = market.loc[market["ticker"].ne("MSFT")].copy()

    with pytest.raises(
        FundamentalBaseError,
        match="exact close price",
    ):
        build_monthly_fundamental_base(
            pit_snapshots=make_pit(),
            ttm_snapshots=make_ttm(),
            market_daily=market,
            universe=make_universe(),
            rebalance_calendar=make_calendar(),
            config=make_config(),
        )


def test_invalid_non_rebalance_price_is_ignored() -> None:
    """An irrelevant invalid market row should not break the base."""
    market = make_market()

    extra = pd.DataFrame(
        {
            "date": [
                "2024-03-27",
            ],
            "ticker": [
                "AAPL",
            ],
            "close": [
                None,
            ],
        }
    )

    market = pd.concat(
        [
            market,
            extra,
        ],
        ignore_index=True,
    )

    result = build_monthly_fundamental_base(
        pit_snapshots=make_pit(),
        ttm_snapshots=make_ttm(),
        market_daily=market,
        universe=make_universe(),
        rebalance_calendar=make_calendar(),
        config=make_config(),
    )

    assert len(result) == 2

    assert result["close_price"].notna().all()
