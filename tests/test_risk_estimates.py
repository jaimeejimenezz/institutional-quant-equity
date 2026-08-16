"""Tests for point-in-time security-level risk estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_equity.risk import (
    RiskEstimateConfig,
    RiskEstimateError,
    build_risk_estimates,
    validate_risk_estimates,
)


def _prices_from_returns(
    initial_price: float,
    returns: np.ndarray,
) -> np.ndarray:
    """Convert a return path into prices."""
    return initial_price * np.cumprod(1.0 + returns)


def make_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create synthetic aligned stock, SPY and signal data."""
    dates = pd.bdate_range(
        "2024-01-02",
        periods=14,
    )

    spy_returns = np.array(
        [
            0.0,
            0.010,
            -0.006,
            0.004,
            0.008,
            -0.003,
            0.005,
            -0.004,
            0.007,
            0.002,
            -0.005,
            0.006,
            -0.002,
            0.004,
        ]
    )

    market_rows = []

    for (
        ticker,
        multiplier,
        _sector,
    ) in (
        (
            "AAA",
            2.0,
            "Technology",
        ),
        (
            "BBB",
            0.5,
            "Financials",
        ),
    ):
        asset_returns = multiplier * spy_returns

        prices = _prices_from_returns(
            50.0,
            asset_returns,
        )

        for (
            number,
            (
                date,
                price,
            ),
        ) in enumerate(
            zip(
                dates,
                prices,
                strict=True,
            )
        ):
            market_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "close": price,
                    "adjusted_close": (price),
                    "volume": (1_000_000.0 + 10_000.0 * number),
                }
            )

    spy_prices = _prices_from_returns(
        100.0,
        spy_returns,
    )

    spy = pd.DataFrame(
        {
            "date": dates,
            "ticker": "SPY",
            "adjusted_close": (spy_prices),
        }
    )

    signal_date = dates[10]

    signal = pd.DataFrame(
        {
            "as_of_date": [
                signal_date,
                signal_date,
            ],
            "ticker": [
                "AAA",
                "BBB",
            ],
            "sector": [
                "Technology",
                "Financials",
            ],
        }
    )

    return (
        pd.DataFrame(market_rows),
        spy,
        signal,
    )


def small_config() -> RiskEstimateConfig:
    """Return a compact configuration for synthetic tests."""
    return RiskEstimateConfig(
        volatility_window_sessions=8,
        beta_window_sessions=8,
        liquidity_window_sessions=5,
        minimum_return_observations=5,
        minimum_liquidity_observations=3,
        annualization_factor=252,
    )


def test_beta_recovers_known_linear_market_exposure() -> None:
    """Synthetic returns proportional to SPY should recover beta."""
    (
        market,
        spy,
        signal,
    ) = make_inputs()

    estimates = build_risk_estimates(
        market,
        spy,
        signal,
        config=small_config(),
    ).set_index("ticker")

    assert estimates.loc[
        "AAA",
        "beta_vs_spy",
    ] == pytest.approx(
        2.0,
        rel=1e-10,
    )

    assert estimates.loc[
        "BBB",
        "beta_vs_spy",
    ] == pytest.approx(
        0.5,
        rel=1e-10,
    )


def test_risk_estimates_preserve_signal_coverage() -> None:
    """Every signal key must receive one risk estimate."""
    (
        market,
        spy,
        signal,
    ) = make_inputs()

    estimates = build_risk_estimates(
        market,
        spy,
        signal,
        config=small_config(),
    )

    assert len(estimates) == len(signal)

    assert (
        estimates.duplicated(
            [
                "as_of_date",
                "ticker",
            ]
        ).sum()
        == 0
    )


def test_estimation_windows_never_extend_beyond_as_of_date() -> None:
    """All stored market dates must respect the point-in-time boundary."""
    (
        market,
        spy,
        signal,
    ) = make_inputs()

    estimates = build_risk_estimates(
        market,
        spy,
        signal,
        config=small_config(),
    )

    assert estimates["latest_market_date"].le(estimates["as_of_date"]).all()

    assert estimates["latest_spy_date"].le(estimates["as_of_date"]).all()

    assert estimates["risk_window_end_date"].le(estimates["as_of_date"]).all()


def test_future_market_changes_do_not_change_existing_estimates() -> None:
    """Future observations must not alter previously calculated risk."""
    (
        market,
        spy,
        signal,
    ) = make_inputs()

    config = small_config()

    first = build_risk_estimates(
        market,
        spy,
        signal,
        config=config,
    )

    cutoff = signal["as_of_date"].iloc[0]

    changed_market = market.copy()

    future_market = changed_market["date"].gt(cutoff)

    changed_market.loc[
        future_market,
        "close",
    ] *= 20.0

    changed_market.loc[
        future_market,
        "adjusted_close",
    ] *= 20.0

    changed_market.loc[
        future_market,
        "volume",
    ] *= 30.0

    changed_spy = spy.copy()

    future_spy = changed_spy["date"].gt(cutoff)

    changed_spy.loc[
        future_spy,
        "adjusted_close",
    ] *= 50.0

    second = build_risk_estimates(
        changed_market,
        changed_spy,
        signal,
        config=config,
    )

    columns = [
        "annualized_volatility",
        "annualized_downside_volatility",
        "beta_vs_spy",
        "correlation_vs_spy",
        "average_dollar_volume",
        "median_dollar_volume",
    ]

    assert np.allclose(
        first[columns],
        second[columns],
    )


def test_validation_checks_pass_for_valid_estimates() -> None:
    """Synthetic valid estimates should pass readiness checks."""
    (
        market,
        spy,
        signal,
    ) = make_inputs()

    config = small_config()

    estimates = build_risk_estimates(
        market,
        spy,
        signal,
        config=config,
    )

    checks = validate_risk_estimates(
        estimates,
        signal,
        config=config,
    )

    assert checks["status"].eq("PASS").all()


def test_duplicate_signal_keys_are_rejected() -> None:
    """Duplicate signal keys would make coverage ambiguous."""
    (
        market,
        spy,
        signal,
    ) = make_inputs()

    duplicated_signal = pd.concat(
        [
            signal,
            signal.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(RiskEstimateError):
        build_risk_estimates(
            market,
            spy,
            duplicated_signal,
            config=small_config(),
        )


def test_market_rows_after_last_signal_date_are_ignored() -> None:
    """Irrelevant future market rows must not affect risk estimation."""
    (
        market,
        spy,
        signal,
    ) = make_inputs()

    future_date = signal["as_of_date"].max() + pd.Timedelta(days=30)

    future_market = pd.DataFrame(
        {
            "date": [
                future_date,
            ],
            "ticker": [
                "AAA",
            ],
            "close": [
                np.nan,
            ],
            "adjusted_close": [
                np.nan,
            ],
            "volume": [
                0.0,
            ],
        }
    )

    market_with_future_missing = pd.concat(
        [
            market,
            future_market,
        ],
        ignore_index=True,
    )

    estimates = build_risk_estimates(
        market_with_future_missing,
        spy,
        signal,
        config=small_config(),
    )

    assert len(estimates) == len(signal)
