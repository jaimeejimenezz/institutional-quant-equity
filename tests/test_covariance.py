"""Tests for rolling shrinkage covariance estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_equity.risk import (
    CovarianceConfig,
    build_covariance_matrices,
    validate_covariance_matrices,
)


def _prices_from_returns(
    initial_price: float,
    returns: np.ndarray,
) -> np.ndarray:
    """Convert synthetic daily returns into prices."""
    return initial_price * np.cumprod(1.0 + returns)


def make_covariance_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create synthetic correlated asset histories."""
    dates = pd.bdate_range(
        "2024-01-02",
        periods=40,
    )

    market_factor = np.array(
        [0.001 * np.sin(index) + 0.0005 * np.cos(index / 3.0) for index in range(len(dates))]
    )

    asset_returns = {
        "AAA": (1.2 * market_factor + 0.0003 * np.sin(np.arange(len(dates)) / 2.0)),
        "BBB": (0.8 * market_factor + 0.0004 * np.cos(np.arange(len(dates)) / 4.0)),
        "CCC": (-0.2 * market_factor + 0.0005 * np.sin(np.arange(len(dates)) / 5.0)),
    }

    rows = []

    for (
        ticker,
        returns,
    ) in asset_returns.items():
        prices = _prices_from_returns(
            100.0,
            returns,
        )

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
                    "adjusted_close": (price),
                }
            )

    signal_date = dates[-2]

    signal = pd.DataFrame(
        {
            "as_of_date": [
                signal_date,
                signal_date,
                signal_date,
            ],
            "ticker": [
                "AAA",
                "BBB",
                "CCC",
            ],
        }
    )

    return (
        pd.DataFrame(rows),
        signal,
    )


def small_covariance_config() -> CovarianceConfig:
    """Return compact configuration for synthetic data."""
    return CovarianceConfig(
        window_sessions=20,
        minimum_observations=12,
        annualization_factor=252,
    )


def test_covariance_matrix_has_complete_asset_cross_product() -> None:
    """Three assets must generate a three-by-three matrix."""
    (
        market,
        signal,
    ) = make_covariance_inputs()

    matrices, diagnostics = build_covariance_matrices(
        market,
        signal,
        config=(small_covariance_config()),
    )

    assert len(matrices) == 9

    assert len(diagnostics) == 1


def test_covariance_matrix_is_symmetric() -> None:
    """Covariance estimates must be symmetric."""
    (
        market,
        signal,
    ) = make_covariance_inputs()

    matrices, _ = build_covariance_matrices(
        market,
        signal,
        config=(small_covariance_config()),
    )

    covariance = matrices.pivot(
        index="ticker_a",
        columns="ticker_b",
        values="annualized_covariance",
    )

    assert np.allclose(
        covariance.to_numpy(),
        covariance.to_numpy().T,
    )


def test_covariance_matrix_is_positive_semidefinite() -> None:
    """Ledoit-Wolf covariance must be positive semidefinite."""
    (
        market,
        signal,
    ) = make_covariance_inputs()

    matrices, _ = build_covariance_matrices(
        market,
        signal,
        config=(small_covariance_config()),
    )

    covariance = matrices.pivot(
        index="ticker_a",
        columns="ticker_b",
        values="annualized_covariance",
    )

    eigenvalues = np.linalg.eigvalsh(covariance.to_numpy())

    assert (eigenvalues >= -1e-12).all()


def test_correlations_are_bounded_and_diagonal_is_one() -> None:
    """Stored correlations must form a valid correlation matrix."""
    (
        market,
        signal,
    ) = make_covariance_inputs()

    matrices, _ = build_covariance_matrices(
        market,
        signal,
        config=(small_covariance_config()),
    )

    correlation = matrices.pivot(
        index="ticker_a",
        columns="ticker_b",
        values="correlation",
    )

    assert (correlation.abs() <= 1.0 + 1e-12).all().all()

    assert np.allclose(
        np.diag(correlation.to_numpy()),
        1.0,
    )


def test_future_prices_do_not_change_existing_covariance() -> None:
    """Market observations after the signal date must be irrelevant."""
    (
        market,
        signal,
    ) = make_covariance_inputs()

    config = small_covariance_config()

    first, _ = build_covariance_matrices(
        market,
        signal,
        config=config,
    )

    cutoff = signal["as_of_date"].max()

    changed_market = market.copy()

    future_mask = changed_market["date"].gt(cutoff)

    changed_market.loc[
        future_mask,
        "adjusted_close",
    ] *= 100.0

    second, _ = build_covariance_matrices(
        changed_market,
        signal,
        config=config,
    )

    assert np.allclose(
        first["annualized_covariance"],
        second["annualized_covariance"],
    )


def test_covariance_readiness_checks_pass() -> None:
    """Valid synthetic covariance artifacts should pass all checks."""
    (
        market,
        signal,
    ) = make_covariance_inputs()

    matrices, diagnostics = build_covariance_matrices(
        market,
        signal,
        config=(small_covariance_config()),
    )

    checks = validate_covariance_matrices(
        matrices,
        diagnostics,
        signal,
    )

    assert checks["status"].eq("PASS").all()
