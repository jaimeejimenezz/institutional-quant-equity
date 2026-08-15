"""Tests for the master modeling panel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_equity.features.fundamental_transforms import (
    FUNDAMENTAL_FACTOR_COLUMNS,
)
from quant_equity.features.technical_processing import (
    TECHNICAL_MODEL_FEATURE_COLUMNS,
)
from quant_equity.models import (
    FUNDAMENTAL_GLOBAL_PANEL_COLUMNS,
    FUNDAMENTAL_MISSING_PANEL_COLUMNS,
    FUNDAMENTAL_SECTOR_PANEL_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TECHNICAL_PANEL_MODEL_COLUMNS,
    ModelingPanelError,
    build_modeling_panel,
    write_modeling_panel,
)


def make_technical() -> pd.DataFrame:
    """Create synthetic processed technical features."""
    rows = []

    for date in pd.to_datetime(
        [
            "2024-01-31",
            "2024-02-29",
        ]
    ):
        for index, ticker in enumerate(
            [
                "AAA",
                "BBB",
            ]
        ):
            row = {
                "as_of_date": date,
                "ticker": ticker,
                "sector": "Technology",
                "latest_market_date": date,
                "observations_available": 300,
            }

            for position, column in enumerate(TECHNICAL_MODEL_FEATURE_COLUMNS):
                row[column] = index + position * 0.01

            rows.append(row)

    return pd.DataFrame(rows)


def make_fundamental() -> pd.DataFrame:
    """Create synthetic processed fundamental features."""
    rows = []

    for date in pd.to_datetime(
        [
            "2024-01-31",
            "2024-02-29",
        ]
    ):
        for index, ticker in enumerate(
            [
                "AAA",
                "BBB",
            ]
        ):
            row = {
                "as_of_date": date,
                "ticker": ticker,
                "sector": "Technology",
            }

            for position, factor in enumerate(FUNDAMENTAL_FACTOR_COLUMNS):
                row[f"{factor}_zscore"] = index + position * 0.01

                row[f"{factor}_sector_zscore"] = index + position * 0.02

                row[f"{factor}_missing"] = 0

            rows.append(row)

    return pd.DataFrame(rows)


def make_labels() -> pd.DataFrame:
    """Create labels only for the first synthetic date."""
    rows = []

    for index, ticker in enumerate(
        [
            "AAA",
            "BBB",
        ]
    ):
        rows.append(
            {
                "as_of_date": pd.Timestamp("2024-01-31"),
                "ticker": ticker,
                "first_future_date": pd.Timestamp("2024-02-01"),
                "target_end_date": pd.Timestamp("2024-03-01"),
                "horizon_sessions": 21,
                "target_21d": (0.05 + index * 0.01),
                "target_21d_excess": (-0.005 + index * 0.01),
                "label_top_quintile": (1 if index == 1 else 0),
            }
        )

    return pd.DataFrame(rows)


def test_master_panel_keeps_feature_spine() -> None:
    """Rows without targets should remain available."""
    panel = build_modeling_panel(
        make_technical(),
        make_fundamental(),
        make_labels(),
    )

    assert len(panel) == 4

    assert panel["has_target"].sum() == 2

    assert panel["sample_role"].value_counts().to_dict() == {
        "modeling": 2,
        "inference_only": 2,
    }


def test_master_panel_prefixes_model_features() -> None:
    """Technical and fundamental features need provenance prefixes."""
    panel = build_modeling_panel(
        make_technical(),
        make_fundamental(),
        make_labels(),
    )

    assert set(TECHNICAL_PANEL_MODEL_COLUMNS).issubset(panel.columns)

    assert set(FUNDAMENTAL_GLOBAL_PANEL_COLUMNS).issubset(panel.columns)

    assert set(FUNDAMENTAL_SECTOR_PANEL_COLUMNS).issubset(panel.columns)

    assert set(FUNDAMENTAL_MISSING_PANEL_COLUMNS).issubset(panel.columns)


def test_duplicate_technical_keys_are_rejected() -> None:
    """Duplicated technical rows must fail."""
    technical = make_technical()

    duplicated = pd.concat(
        [
            technical,
            technical.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ModelingPanelError,
        match="duplicated",
    ):
        build_modeling_panel(
            duplicated,
            make_fundamental(),
            make_labels(),
        )


def test_feature_key_mismatch_is_rejected() -> None:
    """Technical and fundamental panels must align."""
    fundamental = make_fundamental().iloc[:-1].copy()

    with pytest.raises(
        ModelingPanelError,
        match="identical",
    ):
        build_modeling_panel(
            make_technical(),
            fundamental,
            make_labels(),
        )


def test_sector_mismatch_is_rejected() -> None:
    """Sector metadata must agree across sources."""
    fundamental = make_fundamental()

    fundamental.loc[
        fundamental.index[0],
        "sector",
    ] = "Financials"

    with pytest.raises(
        ModelingPanelError,
        match="inconsistent",
    ):
        build_modeling_panel(
            make_technical(),
            fundamental,
            make_labels(),
        )


def test_future_target_start_is_rejected() -> None:
    """The target must begin strictly after the as-of date."""
    labels = make_labels()

    labels.loc[
        labels.index[0],
        "first_future_date",
    ] = labels.loc[
        labels.index[0],
        "as_of_date",
    ]

    with pytest.raises(
        ModelingPanelError,
        match="future target dates",
    ):
        build_modeling_panel(
            make_technical(),
            make_fundamental(),
            labels,
        )


def test_writer_sorts_master_panel(
    tmp_path: Path,
) -> None:
    """Stored master panel should be deterministically sorted."""
    panel = build_modeling_panel(
        make_technical(),
        make_fundamental(),
        make_labels(),
    )

    path = tmp_path / "modeling_panel.parquet"

    write_modeling_panel(
        panel.iloc[::-1],
        path,
    )

    stored = pd.read_parquet(path)

    expected = stored.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        stored,
        expected,
    )

    assert len(MODEL_FEATURE_COLUMNS) == 91
