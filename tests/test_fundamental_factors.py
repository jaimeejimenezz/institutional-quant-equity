"""Tests for raw fundamental factors."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_equity.features import (
    FundamentalFactorConfig,
    FundamentalFactorError,
    build_raw_fundamental_factors,
)


def make_config() -> FundamentalFactorConfig:
    """Create test configuration."""
    return FundamentalFactorConfig(
        diluted_share_metric=("diluted_shares"),
        diluted_share_duration_class=("quarter"),
        capex_positive_outflow=True,
        min_abs_denominator=1.0e-12,
    )


def make_base() -> pd.DataFrame:
    """Create simple accounting inputs."""
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
            "close_price": [
                100.0,
                50.0,
            ],
            "assets": [
                500.0,
                800.0,
            ],
            "cash": [
                50.0,
                100.0,
            ],
            "equity": [
                200.0,
                400.0,
            ],
            "shares_outstanding": [
                10.0,
                None,
            ],
            "shares_outstanding_available_date": [
                "2024-02-01",
                None,
            ],
            "current_assets": [
                120.0,
                200.0,
            ],
            "current_liabilities": [
                60.0,
                100.0,
            ],
            "debt_current": [
                20.0,
                30.0,
            ],
            "debt_noncurrent": [
                80.0,
                70.0,
            ],
            "revenue_ttm": [
                400.0,
                500.0,
            ],
            "gross_profit_ttm": [
                200.0,
                250.0,
            ],
            "operating_income_ttm": [
                80.0,
                100.0,
            ],
            "net_income_ttm": [
                50.0,
                60.0,
            ],
            "operating_cash_flow_ttm": [
                70.0,
                90.0,
            ],
            "capex_ttm": [
                20.0,
                30.0,
            ],
            "interest_expense_ttm": [
                10.0,
                20.0,
            ],
        }
    )


def make_pit() -> pd.DataFrame:
    """Create quarterly diluted-share observations."""
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
                "diluted_shares",
                "diluted_shares",
            ],
            "statement_type": [
                "duration",
                "duration",
            ],
            "duration_class": [
                "quarter",
                "quarter",
            ],
            "value": [
                11.0,
                20.0,
            ],
            "end_date": [
                "2023-12-31",
                "2023-12-31",
            ],
            "available_date": [
                "2024-02-02",
                "2024-02-02",
            ],
        }
    )


def test_share_count_prefers_outstanding_then_diluted() -> None:
    """Use outstanding shares first and diluted shares as fallback."""
    result = build_raw_fundamental_factors(
        fundamental_base=make_base(),
        pit_snapshots=make_pit(),
        config=make_config(),
    )

    apple = result.loc[result["ticker"].eq("AAPL")].iloc[0]

    microsoft = result.loc[result["ticker"].eq("MSFT")].iloc[0]

    assert apple["valuation_share_count"] == 10.0

    assert apple["valuation_share_count_source"] == "shares_outstanding"

    assert microsoft["valuation_share_count"] == 20.0

    assert microsoft["valuation_share_count_source"] == "diluted_shares_quarter"


def test_market_cap_proxy_is_calculated() -> None:
    """Price times selected share count should form market-cap proxy."""
    result = build_raw_fundamental_factors(
        fundamental_base=make_base(),
        pit_snapshots=make_pit(),
        config=make_config(),
    )

    apple = result.loc[result["ticker"].eq("AAPL")].iloc[0]

    microsoft = result.loc[result["ticker"].eq("MSFT")].iloc[0]

    assert apple["market_cap_proxy"] == 1000.0

    assert microsoft["market_cap_proxy"] == 1000.0


def test_positive_capex_is_subtracted_from_cfo() -> None:
    """Positive CAPEX represents a cash outflow."""
    result = build_raw_fundamental_factors(
        fundamental_base=make_base(),
        pit_snapshots=make_pit(),
        config=make_config(),
    )

    apple = result.loc[result["ticker"].eq("AAPL")].iloc[0]

    assert apple["free_cash_flow_ttm"] == 50.0

    assert apple["fcf_yield"] == pytest.approx(0.05)


def test_manual_ratios_are_correct() -> None:
    """Core raw ratios should match manual calculations."""
    result = build_raw_fundamental_factors(
        fundamental_base=make_base(),
        pit_snapshots=make_pit(),
        config=make_config(),
    )

    apple = result.loc[result["ticker"].eq("AAPL")].iloc[0]

    assert apple["earnings_yield"] == pytest.approx(0.05)

    assert apple["sales_yield"] == pytest.approx(0.40)

    assert apple["book_to_market"] == pytest.approx(0.20)

    assert apple["roe"] == pytest.approx(0.25)

    assert apple["roa"] == pytest.approx(0.10)

    assert apple["gross_profitability"] == pytest.approx(0.40)

    assert apple["gross_margin"] == pytest.approx(0.50)

    assert apple["operating_margin"] == pytest.approx(0.20)

    assert apple["net_margin"] == pytest.approx(0.125)

    assert apple["cash_conversion"] == pytest.approx(1.40)

    assert apple["debt_to_assets"] == pytest.approx(0.20)

    assert apple["net_debt_to_assets"] == pytest.approx(0.10)

    assert apple["current_ratio"] == pytest.approx(2.0)

    assert apple["interest_coverage"] == pytest.approx(8.0)

    assert apple["capex_to_assets"] == pytest.approx(0.04)

    assert apple["accruals"] == pytest.approx(-0.04)


def test_missing_debt_component_does_not_assume_zero() -> None:
    """Missing debt should not silently be interpreted as zero."""
    base = make_base()

    base.loc[
        base["ticker"].eq("AAPL"),
        "debt_noncurrent",
    ] = None

    result = build_raw_fundamental_factors(
        fundamental_base=base,
        pit_snapshots=make_pit(),
        config=make_config(),
    )

    apple = result.loc[result["ticker"].eq("AAPL")].iloc[0]

    assert pd.isna(apple["total_debt"])

    assert pd.isna(apple["debt_to_assets"])


def test_future_diluted_shares_are_rejected() -> None:
    """A share count unavailable at the rebalance date cannot be used."""
    pit = make_pit()

    pit.loc[
        pit["ticker"].eq("MSFT"),
        "available_date",
    ] = "2024-04-01"

    with pytest.raises(
        FundamentalFactorError,
        match="future information",
    ):
        build_raw_fundamental_factors(
            fundamental_base=make_base(),
            pit_snapshots=pit,
            config=make_config(),
        )
