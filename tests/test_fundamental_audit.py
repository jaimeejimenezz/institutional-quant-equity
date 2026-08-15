"""Tests for fundamental feature diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_equity.features import (
    FUNDAMENTAL_FEATURE_METADATA,
    FundamentalAuditConfig,
    build_cross_sectional_correlations,
    build_fundamental_feature_summary,
    build_zscore_audit,
    select_high_correlations,
    update_feature_dictionary,
    validate_feature_metadata,
)
from quant_equity.features.fundamental_transforms import (
    FUNDAMENTAL_FACTOR_COLUMNS,
)


def make_config() -> FundamentalAuditConfig:
    """Create a compact diagnostic configuration."""
    return FundamentalAuditConfig(
        high_correlation_threshold=0.80,
        min_pair_dates=3,
        min_pair_observations=4,
        zscore_mean_tolerance=1.0e-10,
    )


def make_data() -> pd.DataFrame:
    """Create synthetic processed fundamental features."""
    rows = []

    dates = pd.date_range(
        "2024-01-31",
        periods=6,
        freq="ME",
    )

    tickers = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ]

    sectors = [
        "Technology",
        "Technology",
        "Technology",
        "Financials",
        "Financials",
        "Financials",
    ]

    global_scores = np.asarray(
        [
            -1.463850,
            -0.878310,
            -0.292770,
            0.292770,
            0.878310,
            1.463850,
        ]
    )

    sector_scores = np.asarray(
        [
            -1.224745,
            0.0,
            1.224745,
            -1.224745,
            0.0,
            1.224745,
        ]
    )

    for date_index, date in enumerate(dates):
        for company_index, (
            ticker,
            sector,
        ) in enumerate(
            zip(
                tickers,
                sectors,
                strict=True,
            )
        ):
            row = {
                "as_of_date": date,
                "ticker": ticker,
                "sector": sector,
            }

            for factor_index, factor in enumerate(FUNDAMENTAL_FACTOR_COLUMNS):
                raw_value = company_index + 1.0 + date_index * 0.1 + factor_index * 0.01

                row[factor] = raw_value

                row[f"{factor}_zscore"] = global_scores[company_index]

                row[f"{factor}_sector_zscore"] = sector_scores[company_index]

            rows.append(row)

    return pd.DataFrame(rows)


def test_metadata_covers_all_factors() -> None:
    """Every raw fundamental factor must be documented."""
    validate_feature_metadata()

    assert set(FUNDAMENTAL_FEATURE_METADATA) == set(FUNDAMENTAL_FACTOR_COLUMNS)


def test_summary_contains_all_factors() -> None:
    """Summary should contain one row per fundamental factor."""
    summary = build_fundamental_feature_summary(make_data())

    assert len(summary) == len(FUNDAMENTAL_FACTOR_COLUMNS)

    assert (summary["latest_raw_coverage"] == 1.0).all()


def test_correlations_detect_perfect_redundancy() -> None:
    """Identical cross-sectional rankings should correlate perfectly."""
    correlations = build_cross_sectional_correlations(
        make_data(),
        config=make_config(),
    )

    pair = correlations.loc[
        (correlations["factor_1"].eq("earnings_yield") & correlations["factor_2"].eq("sales_yield"))
    ].iloc[0]

    assert pair["mean_spearman"] == 1.0


def test_high_correlation_selection() -> None:
    """Highly redundant synthetic factors should be selected."""
    correlations = build_cross_sectional_correlations(
        make_data(),
        config=make_config(),
    )

    selected = select_high_correlations(
        correlations,
        config=make_config(),
    )

    assert not selected.empty

    assert selected["mean_spearman"].abs().ge(0.80).all()


def test_global_zscores_are_centered() -> None:
    """Global z-scores should remain centered by date."""
    audit = build_zscore_audit(
        make_data(),
        config=make_config(),
    )

    global_rows = audit.loc[audit["version"].eq("zscore")]

    assert (global_rows["centering_violations"] == 0).all()


def test_sector_zscores_are_centered() -> None:
    """Sector z-scores should remain centered by date and sector."""
    audit = build_zscore_audit(
        make_data(),
        config=make_config(),
    )

    sector_rows = audit.loc[audit["version"].eq("sector_zscore")]

    assert (sector_rows["centering_violations"] == 0).all()


def test_dictionary_preserves_existing_content(
    tmp_path: Path,
) -> None:
    """Fundamental documentation must not overwrite other features."""
    path = tmp_path / "FEATURE_DICTIONARY.md"

    path.write_text(
        "# Feature Dictionary\n\nExisting technical feature documentation.\n",
        encoding="utf-8",
    )

    update_feature_dictionary(path)

    content = path.read_text(
        encoding="utf-8",
    )

    assert "Existing technical feature documentation." in content

    assert "<!-- FUNDAMENTAL_FEATURES_START -->" in content

    assert "`earnings_yield`" in content
