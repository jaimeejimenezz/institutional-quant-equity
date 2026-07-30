"""Tests for extended technical-factor diagnostics."""

import numpy as np
import pandas as pd

from quant_equity.research.technical_factor_diagnostics import (
    ResearchPeriod,
    TechnicalFactorDiagnosticsConfig,
    build_mean_correlation_matrix,
    build_selection_diagnostics,
    calculate_monthly_sector_ic,
    calculate_monthly_signal_correlations,
    summarize_ic_by_period,
    summarize_sector_ic,
    summarize_signal_correlations,
)

SIGNALS = (
    "signal_up",
    "signal_down",
)


def make_panel() -> pd.DataFrame:
    """Create a synthetic monthly factor panel."""
    rows: list[dict[str, object]] = []

    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-28",
            "2020-03-31",
        ]
    )

    for date in dates:
        for position in range(10):
            rows.append(
                {
                    "as_of_date": date,
                    "ticker": f"T{position:02d}",
                    "sector": ("Sector A" if position < 5 else "Sector B"),
                    "signal_up": float(position),
                    "signal_down": float(-position),
                    "target": float(position),
                }
            )

    return pd.DataFrame(rows)


def test_monthly_correlations_detect_inverse_signals() -> None:
    """Perfect inverse signals should correlate at minus one."""
    monthly = calculate_monthly_signal_correlations(
        make_panel(),
        signal_columns=SIGNALS,
        minimum_observations=5,
    )

    assert np.allclose(
        monthly["spearman_correlation"],
        -1.0,
    )

    summary = summarize_signal_correlations(monthly)

    assert len(summary) == 1

    assert np.isclose(
        summary.loc[
            0,
            "mean_correlation",
        ],
        -1.0,
    )

    assert np.isclose(
        summary.loc[
            0,
            "mean_absolute_correlation",
        ],
        1.0,
    )


def test_correlation_matrix_is_symmetric() -> None:
    """The correlation matrix should be symmetric."""
    monthly = calculate_monthly_signal_correlations(
        make_panel(),
        signal_columns=SIGNALS,
        minimum_observations=5,
    )

    summary = summarize_signal_correlations(monthly)

    matrix = build_mean_correlation_matrix(
        summary,
        signal_columns=SIGNALS,
    )

    assert np.allclose(
        matrix.to_numpy(),
        matrix.to_numpy().T,
    )

    assert np.allclose(
        np.diag(matrix),
        1.0,
    )


def test_period_summary_adjusts_negative_direction() -> None:
    """A negative IC can be positive after direction adjustment."""
    monthly_ic = pd.DataFrame(
        {
            "signal": [
                "signal_down",
                "signal_down",
                "signal_down",
            ],
            "as_of_date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-02-28",
                    "2020-03-31",
                ]
            ),
            "ic": [
                -0.10,
                -0.20,
                -0.30,
            ],
        }
    )

    result = summarize_ic_by_period(
        monthly_ic,
        periods=(
            ResearchPeriod(
                name="2020",
                start_date=pd.Timestamp("2020-01-01"),
                end_date=pd.Timestamp("2020-12-31"),
            ),
        ),
        preferred_direction_by_signal={"signal_down": "lower_is_better"},
    )

    assert np.isclose(
        result.loc[
            0,
            "mean_ic",
        ],
        -0.20,
    )

    assert np.isclose(
        result.loc[
            0,
            "directional_mean_ic",
        ],
        0.20,
    )

    assert np.isclose(
        result.loc[
            0,
            "directional_month_ratio",
        ],
        1.0,
    )


def test_sector_ic_respects_minimum_sector_size() -> None:
    """Sector IC should only exist for sufficiently large groups."""
    panel = make_panel()

    monthly = calculate_monthly_sector_ic(
        panel,
        target_column="target",
        signal_columns=SIGNALS,
        minimum_sector_size=5,
    )

    valid_values = monthly["ic"].dropna()

    assert not valid_values.empty

    assert np.allclose(
        monthly.loc[
            monthly["signal"].eq("signal_up"),
            "ic",
        ],
        1.0,
    )

    summary = summarize_sector_ic(
        monthly,
        preferred_direction_by_signal={
            "signal_up": "higher_is_better",
            "signal_down": "lower_is_better",
        },
    )

    assert summary["directional_mean_ic"].gt(0.0).all()


def test_selection_diagnostics_marks_inverse_pair_as_redundant() -> None:
    """Highly correlated signals should require redundancy review."""
    ic_summary = pd.DataFrame(
        {
            "signal": list(SIGNALS),
            "months": [36, 36],
            "mean_ic": [0.05, -0.05],
            "median_ic": [0.05, -0.05],
            "std_ic": [0.10, 0.10],
            "annualized_ic_ir": [1.0, -1.0],
            "ic_t_stat": [2.0, -2.0],
            "positive_month_ratio": [0.60, 0.40],
            "abs_mean_ic": [0.05, 0.05],
            "preferred_direction": [
                "higher_is_better",
                "lower_is_better",
            ],
        }
    )

    spread_summary = pd.DataFrame(
        {
            "signal": list(SIGNALS),
            "mean_top_bottom_spread": [
                0.01,
                -0.01,
            ],
            "positive_spread_ratio": [
                0.60,
                0.40,
            ],
            "mean_quintile_monotonicity": [
                0.30,
                -0.30,
            ],
        }
    )

    turnover_summary = pd.DataFrame(
        {
            "signal": list(SIGNALS),
            "mean_turnover": [0.20, 0.20],
            "median_turnover": [0.20, 0.20],
        }
    )

    yearly_ic = pd.DataFrame(
        {
            "period": [
                "2020",
                "2020",
            ],
            "signal": list(SIGNALS),
            "directional_mean_ic": [
                0.05,
                0.05,
            ],
        }
    )

    subperiod_ic = yearly_ic.copy()

    sector_ic = pd.DataFrame(
        {
            "signal": list(SIGNALS),
            "sector": [
                "Sector A",
                "Sector A",
            ],
            "months": [24, 24],
            "directional_mean_ic": [
                0.05,
                0.05,
            ],
        }
    )

    correlation_summary = pd.DataFrame(
        {
            "first_signal": ["signal_up"],
            "second_signal": ["signal_down"],
            "months": [36],
            "mean_correlation": [-1.0],
            "median_correlation": [-1.0],
            "mean_absolute_correlation": [1.0],
            "maximum_absolute_correlation": [1.0],
        }
    )

    diagnostics = build_selection_diagnostics(
        ic_summary,
        spread_summary,
        turnover_summary,
        yearly_ic,
        subperiod_ic,
        sector_ic,
        correlation_summary,
        config=TechnicalFactorDiagnosticsConfig(
            correlation_threshold=0.90,
            minimum_pair_observations=5,
            minimum_sector_cross_section_size=3,
            minimum_sector_months=12,
            top_signals_in_figures=2,
            subperiods=(),
        ),
    )

    assert set(diagnostics["preliminary_status"]) == {"review_redundancy"}
