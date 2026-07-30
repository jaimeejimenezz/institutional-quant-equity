"""Tests for univariate technical-factor research."""

import numpy as np
import pandas as pd
import pytest

from quant_equity.research import (
    TechnicalFactorResearchConfig,
    TechnicalFactorResearchError,
    build_factor_research_panel,
    calculate_ic_summary,
    calculate_monthly_information_coefficients,
    calculate_quintile_research,
    calculate_selected_quantile_turnover,
)

TEST_SIGNALS = (
    "signal_up",
    "signal_down",
)


def make_config() -> TechnicalFactorResearchConfig:
    """Create deterministic research configuration."""
    return TechnicalFactorResearchConfig(
        target_column="target_21d_excess",
        absolute_return_column="target_21d",
        number_of_quantiles=5,
        minimum_cross_section_size=10,
        annualization_periods=12,
        research_start_date=pd.Timestamp("2020-01-01"),
        research_end_date=pd.Timestamp("2020-12-31"),
    )


def make_features() -> pd.DataFrame:
    """Create synthetic technical signals."""
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
                    "latest_market_date": date,
                    "signal_up": float(position),
                    "signal_down": float(-position),
                }
            )

    return pd.DataFrame(rows)


def make_labels() -> pd.DataFrame:
    """Create synthetic future returns."""
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
            relative_return = (position - 4.5) / 100.0

            rows.append(
                {
                    "as_of_date": date,
                    "ticker": f"T{position:02d}",
                    "first_future_date": (date + pd.Timedelta(days=1)),
                    "target_end_date": (date + pd.Timedelta(days=30)),
                    "horizon_sessions": 21,
                    "target_21d": (relative_return + 0.01),
                    "target_21d_excess": (relative_return),
                    "label_top_quintile": int(position >= 8),
                }
            )

    return pd.DataFrame(rows)


def test_factor_panel_preserves_temporal_order() -> None:
    """Future returns must begin after the signal date."""
    panel = build_factor_research_panel(
        make_features(),
        make_labels(),
        config=make_config(),
        signal_columns=TEST_SIGNALS,
    )

    assert len(panel) == 30

    assert panel["latest_market_date"].le(panel["as_of_date"]).all()

    assert panel["first_future_date"].gt(panel["as_of_date"]).all()


def test_monthly_information_coefficients_detect_direction() -> None:
    """Increasing and decreasing signals should have opposite IC."""
    panel = build_factor_research_panel(
        make_features(),
        make_labels(),
        config=make_config(),
        signal_columns=TEST_SIGNALS,
    )

    monthly_ic = calculate_monthly_information_coefficients(
        panel,
        config=make_config(),
        signal_columns=TEST_SIGNALS,
    )

    upward = monthly_ic.loc[
        monthly_ic["signal"].eq("signal_up"),
        "ic",
    ]

    downward = monthly_ic.loc[
        monthly_ic["signal"].eq("signal_down"),
        "ic",
    ]

    assert np.allclose(
        upward,
        1.0,
    )

    assert np.allclose(
        downward,
        -1.0,
    )


def test_ic_summary_reports_preferred_direction() -> None:
    """IC summary should identify the economic direction."""
    panel = build_factor_research_panel(
        make_features(),
        make_labels(),
        config=make_config(),
        signal_columns=TEST_SIGNALS,
    )

    monthly_ic = calculate_monthly_information_coefficients(
        panel,
        config=make_config(),
        signal_columns=TEST_SIGNALS,
    )

    summary = calculate_ic_summary(
        monthly_ic,
        config=make_config(),
    ).set_index("signal")

    assert (
        summary.loc[
            "signal_up",
            "preferred_direction",
        ]
        == "higher_is_better"
    )

    assert (
        summary.loc[
            "signal_down",
            "preferred_direction",
        ]
        == "lower_is_better"
    )


def test_quintile_spreads_detect_signal_direction() -> None:
    """Q5-Q1 spread should reflect the signal direction."""
    panel = build_factor_research_panel(
        make_features(),
        make_labels(),
        config=make_config(),
        signal_columns=TEST_SIGNALS,
    )

    (
        _,
        _,
        _,
        spread_summary,
    ) = calculate_quintile_research(
        panel,
        config=make_config(),
        signal_columns=TEST_SIGNALS,
    )

    spreads = spread_summary.set_index("signal")

    assert (
        spreads.loc[
            "signal_up",
            "mean_top_bottom_spread",
        ]
        > 0.0
    )

    assert (
        spreads.loc[
            "signal_down",
            "mean_top_bottom_spread",
        ]
        < 0.0
    )


def test_selected_quantile_turnover_is_bounded() -> None:
    """Preferred-quantile turnover should remain between zero and one."""
    features = make_features()

    second_date = pd.Timestamp("2020-02-28")

    second_date_rows = features["as_of_date"].eq(second_date)

    features.loc[
        second_date_rows,
        "signal_up",
    ] = np.roll(
        np.arange(
            10,
            dtype=float,
        ),
        5,
    )

    panel = build_factor_research_panel(
        features,
        make_labels(),
        config=make_config(),
        signal_columns=TEST_SIGNALS,
    )

    monthly_ic = calculate_monthly_information_coefficients(
        panel,
        config=make_config(),
        signal_columns=TEST_SIGNALS,
    )

    summary = calculate_ic_summary(
        monthly_ic,
        config=make_config(),
    )

    monthly_turnover, _ = calculate_selected_quantile_turnover(
        panel,
        summary,
        config=make_config(),
        signal_columns=TEST_SIGNALS,
    )

    assert (
        monthly_turnover["turnover"]
        .between(
            0.0,
            1.0,
        )
        .all()
    )


def test_panel_rejects_target_starting_on_signal_date() -> None:
    """A target may not begin on as_of_date."""
    labels = make_labels()

    labels.loc[
        0,
        "first_future_date",
    ] = labels.loc[
        0,
        "as_of_date",
    ]

    with pytest.raises(
        TechnicalFactorResearchError,
        match="begin",
    ):
        build_factor_research_panel(
            make_features(),
            labels,
            config=make_config(),
            signal_columns=TEST_SIGNALS,
        )


def test_panel_ignores_unlabelled_features_outside_research_window() -> None:
    """Features outside the research window need no label."""
    features = make_features()

    future_rows = features.loc[features["as_of_date"].eq(pd.Timestamp("2020-03-31"))].copy()

    future_rows["as_of_date"] = pd.Timestamp("2021-01-29")

    future_rows["latest_market_date"] = pd.Timestamp("2021-01-29")

    features = pd.concat(
        [
            features,
            future_rows,
        ],
        ignore_index=True,
    )

    panel = build_factor_research_panel(
        features,
        make_labels(),
        config=make_config(),
        signal_columns=TEST_SIGNALS,
    )

    assert len(panel) == 30

    assert panel["as_of_date"].max() == pd.Timestamp("2020-03-31")
