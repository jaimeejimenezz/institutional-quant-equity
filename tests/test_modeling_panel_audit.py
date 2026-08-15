"""Tests for master modeling-panel leakage auditing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_equity.models.modeling_panel import (
    MODEL_FEATURE_COLUMNS,
)
from quant_equity.validation import (
    audit_modeling_panel,
)


def make_panel() -> pd.DataFrame:
    """Create a small valid master panel."""
    rows = []

    dates = [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-29"),
    ]

    for date in dates:
        for ticker in (
            "AAA",
            "BBB",
        ):
            has_target = int(date == pd.Timestamp("2024-01-31"))

            row = {
                "as_of_date": date,
                "ticker": ticker,
                "sector": "Technology",
                "technical_latest_market_date": date,
            }

            for column in MODEL_FEATURE_COLUMNS:
                row[column] = 0.0

            if has_target:
                row.update(
                    {
                        "first_future_date": pd.Timestamp("2024-02-01"),
                        "target_end_date": pd.Timestamp("2024-03-01"),
                        "horizon_sessions": 21,
                        "target_21d": 0.05,
                        "target_21d_excess": 0.01,
                        "label_top_quintile": 1,
                        "has_target": 1,
                        "sample_role": "modeling",
                    }
                )
            else:
                row.update(
                    {
                        "first_future_date": pd.NaT,
                        "target_end_date": pd.NaT,
                        "horizon_sessions": np.nan,
                        "target_21d": np.nan,
                        "target_21d_excess": np.nan,
                        "label_top_quintile": np.nan,
                        "has_target": 0,
                        "sample_role": "inference_only",
                    }
                )

            rows.append(row)

    return pd.DataFrame(rows)


def make_calendar() -> pd.DataFrame:
    """Create a valid rebalance calendar."""
    return pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(
                [
                    "2024-01-31",
                    "2024-02-29",
                ]
            ),
            "first_future_date": pd.to_datetime(
                [
                    "2024-02-01",
                    "2024-03-01",
                ]
            ),
            "target_end_date": [
                pd.Timestamp("2024-03-01"),
                pd.NaT,
            ],
            "horizon_sessions": [
                21,
                21,
            ],
            "has_full_horizon": [
                True,
                False,
            ],
        }
    )


def make_ttm() -> pd.DataFrame:
    """Create valid TTM provenance rows."""
    rows = []

    for date in pd.to_datetime(
        [
            "2024-01-31",
            "2024-02-29",
        ]
    ):
        for ticker in (
            "AAA",
            "BBB",
        ):
            rows.append(
                {
                    "as_of_date": date,
                    "ticker": ticker,
                    "canonical_metric": "revenue",
                    "latest_quarter_end": pd.Timestamp("2023-12-31"),
                    "quarter_count": 4,
                    "latest_component_available_date": (pd.Timestamp("2024-01-30")),
                }
            )

    return pd.DataFrame(rows)


def audit(
    panel: pd.DataFrame | None = None,
    calendar: pd.DataFrame | None = None,
    ttm: pd.DataFrame | None = None,
):
    """Run the synthetic audit."""
    return audit_modeling_panel(
        (make_panel() if panel is None else panel),
        (make_calendar() if calendar is None else calendar),
        (make_ttm() if ttm is None else ttm),
        expected_horizon_sessions=21,
    )


def test_valid_panel_passes_audit() -> None:
    """A correctly aligned dataset should pass."""
    result = audit()

    assert result.is_valid
    assert not result.issues

    assert (result.checks["status"] == "PASS").all()


def test_duplicate_panel_key_is_detected() -> None:
    """Duplicate master keys must fail."""
    panel = make_panel()

    panel = pd.concat(
        [
            panel,
            panel.iloc[[0]],
        ],
        ignore_index=True,
    )

    result = audit(panel=panel)

    assert not result.is_valid

    check = result.checks.set_index("check").loc["unique_panel_keys"]

    assert check["violations"] > 0


def test_future_technical_date_is_detected() -> None:
    """Technical features cannot see future market data."""
    panel = make_panel()

    panel.loc[
        panel.index[0],
        "technical_latest_market_date",
    ] = pd.Timestamp("2024-02-01")

    result = audit(panel=panel)

    check = result.checks.set_index("check").loc["technical_point_in_time"]

    assert check["violations"] == 1


def test_same_day_target_start_is_detected() -> None:
    """Forward targets must start after the signal date."""
    panel = make_panel()

    first_date = panel["as_of_date"].eq(pd.Timestamp("2024-01-31"))

    panel.loc[
        first_date,
        "first_future_date",
    ] = pd.Timestamp("2024-01-31")

    result = audit(panel=panel)

    check = result.checks.set_index("check").loc["target_starts_after_as_of"]

    assert check["violations"] == 2


def test_wrong_target_horizon_is_detected() -> None:
    """A target with the wrong horizon must fail."""
    panel = make_panel()

    first_date = panel["as_of_date"].eq(pd.Timestamp("2024-01-31"))

    panel.loc[
        first_date,
        "horizon_sessions",
    ] = 42

    result = audit(panel=panel)

    check = result.checks.set_index("check").loc["target_horizon"]

    assert check["violations"] == 2


def test_inference_row_cannot_contain_target() -> None:
    """Inference-only observations cannot expose future returns."""
    panel = make_panel()

    row = panel["as_of_date"].eq(pd.Timestamp("2024-02-29"))

    panel.loc[
        row,
        "target_21d",
    ] = 0.10

    result = audit(panel=panel)

    check = result.checks.set_index("check").loc["complete_target_tuples"]

    assert check["violations"] == 2


def test_infinite_model_feature_is_detected() -> None:
    """Infinite predictor values must fail the audit."""
    panel = make_panel()

    panel.loc[
        panel.index[0],
        MODEL_FEATURE_COLUMNS[0],
    ] = np.inf

    result = audit(panel=panel)

    check = result.checks.set_index("check").loc["finite_model_features"]

    assert check["violations"] == 1


def test_future_ttm_availability_is_detected() -> None:
    """TTM data unavailable at as_of_date must fail."""
    ttm = make_ttm()

    ttm.loc[
        ttm.index[0],
        "latest_component_available_date",
    ] = pd.Timestamp("2024-02-01")

    result = audit(ttm=ttm)

    check = result.checks.set_index("check").loc["ttm_point_in_time"]

    assert check["violations"] == 1


def test_ttm_key_outside_panel_is_detected() -> None:
    """TTM snapshots must belong to panel date-ticker keys."""
    ttm = make_ttm()

    extra = ttm.iloc[[0]].copy()

    extra["ticker"] = "CCC"

    ttm = pd.concat(
        [
            ttm,
            extra,
        ],
        ignore_index=True,
    )

    result = audit(ttm=ttm)

    check = result.checks.set_index("check").loc["ttm_panel_alignment"]

    assert check["violations"] == 1
