"""Tests for master modeling-panel documentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_equity.models.modeling_panel import (
    MODEL_FEATURE_COLUMNS,
    MODELING_PANEL_COLUMNS,
)
from quant_equity.reporting import (
    build_modeling_feature_coverage,
    build_modeling_panel_data_dictionary,
    render_data_dictionary,
    write_modeling_panel_dictionary_artifacts,
)


def make_panel() -> pd.DataFrame:
    """Create a complete synthetic modeling panel."""
    rows = []

    dates = pd.to_datetime(
        [
            "2024-01-31",
            "2024-02-29",
        ]
    )

    tickers = [
        "AAA",
        "BBB",
    ]

    for date in dates:
        for ticker in tickers:
            has_target = int(date == pd.Timestamp("2024-01-31"))

            row = {
                "as_of_date": date,
                "ticker": ticker,
                "sector": "Technology",
                "technical_latest_market_date": date,
                "observations_available": 300,
            }

            for column in MODEL_FEATURE_COLUMNS:
                if column.endswith("_missing"):
                    row[column] = 0
                else:
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
                        "technical_missing_count": 0,
                        "fundamental_global_missing_count": 0,
                        "fundamental_sector_missing_count": 0,
                        "model_feature_missing_count": 0,
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
                        "technical_missing_count": 0,
                        "fundamental_global_missing_count": 0,
                        "fundamental_sector_missing_count": 0,
                        "model_feature_missing_count": 0,
                        "has_target": 0,
                        "sample_role": "inference_only",
                    }
                )

            rows.append(row)

    return pd.DataFrame(rows).loc[
        :,
        MODELING_PANEL_COLUMNS,
    ]


def test_dictionary_covers_every_panel_column() -> None:
    """Every master-panel column must be documented."""
    dictionary = build_modeling_panel_data_dictionary(make_panel())

    assert len(dictionary) == len(MODELING_PANEL_COLUMNS)

    assert set(dictionary["column"]) == set(MODELING_PANEL_COLUMNS)

    assert not dictionary["column"].duplicated().any()


def test_dictionary_marks_exact_model_inputs() -> None:
    """Only candidate predictor columns may be model inputs."""
    dictionary = build_modeling_panel_data_dictionary(make_panel())

    model_inputs = set(
        dictionary.loc[
            dictionary["model_input"],
            "column",
        ]
    )

    assert model_inputs == set(MODEL_FEATURE_COLUMNS)

    assert len(model_inputs) == 91


def test_technical_lineage_is_documented() -> None:
    """Technical predictors should retain source lineage."""
    dictionary = build_modeling_panel_data_dictionary(make_panel())

    row = dictionary.loc[dictionary["feature_group"].eq("technical")].iloc[0]

    assert row["column"].startswith("tech__")

    assert row["source_column"].endswith("_sector_neutral")

    assert "winsorized" in row["calculation"]

    assert "sector-neutralized" in row["calculation"]


def test_fundamental_formula_is_documented() -> None:
    """Fundamental predictors should retain financial formulas."""
    dictionary = build_modeling_panel_data_dictionary(make_panel())

    row = dictionary.loc[dictionary["column"].eq("fund__earnings_yield_zscore")].iloc[0]

    assert "Net Income TTM" in row["calculation"]

    assert row["feature_group"] == "fundamental_global"


def test_missingness_flag_is_documented_as_predictor() -> None:
    """Fundamental missing flags remain explicit predictors."""
    dictionary = build_modeling_panel_data_dictionary(make_panel())

    row = dictionary.loc[dictionary["column"].eq("fund__earnings_yield_missing")].iloc[0]

    assert row["model_input"]

    assert row["feature_group"] == "fundamental_missing"

    assert "Binary flag" in row["calculation"]


def test_coverage_detects_missing_predictor() -> None:
    """Coverage should reflect missing feature values."""
    panel = make_panel()

    feature = MODEL_FEATURE_COLUMNS[0]

    panel.loc[
        panel.index[0],
        feature,
    ] = np.nan

    dictionary = build_modeling_panel_data_dictionary(panel)

    row = dictionary.loc[dictionary["column"].eq(feature)].iloc[0]

    assert row["non_missing_count"] == 3

    assert row["missing_count"] == 1

    assert row["overall_coverage"] == 0.75


def test_feature_coverage_contains_91_rows() -> None:
    """Coverage output needs one row per predictor."""
    dictionary = build_modeling_panel_data_dictionary(make_panel())

    coverage = build_modeling_feature_coverage(dictionary)

    assert len(coverage) == 91

    assert set(coverage["column"]) == set(MODEL_FEATURE_COLUMNS)


def test_rendered_dictionary_documents_temporal_boundary() -> None:
    """Documentation must explain the future-data boundary."""
    panel = make_panel()

    dictionary = build_modeling_panel_data_dictionary(panel)

    content = render_data_dictionary(
        panel,
        dictionary,
    )

    assert "# Data Dictionary — Modeling Panel" in content

    assert "Temporal contract" in content

    assert "target_21d_excess" in content

    assert "walk-forward" in content


def test_dictionary_artifacts_are_written(
    tmp_path: Path,
) -> None:
    """Step 11C should persist all documentation outputs."""
    dictionary_path = tmp_path / "dictionary.csv"

    coverage_path = tmp_path / "coverage.csv"

    documentation_path = tmp_path / "DATA_DICTIONARY.md"

    dictionary, coverage = write_modeling_panel_dictionary_artifacts(
        make_panel(),
        dictionary_csv_path=(dictionary_path),
        coverage_csv_path=(coverage_path),
        documentation_path=(documentation_path),
    )

    assert dictionary_path.exists()
    assert coverage_path.exists()
    assert documentation_path.exists()

    assert len(dictionary) == len(MODELING_PANEL_COLUMNS)

    assert len(coverage) == 91
