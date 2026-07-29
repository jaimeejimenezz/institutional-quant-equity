"""Validation and reporting for processed technical features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_equity.features.technical import (
    TECHNICAL_FEATURE_COLUMNS,
)
from quant_equity.features.technical_processing import (
    PROCESSED_TECHNICAL_COLUMNS,
    SECTOR_NEUTRAL_FEATURE_COLUMNS,
    STANDARDIZED_FEATURE_COLUMNS,
    WINSORIZED_FEATURE_COLUMNS,
    TechnicalFeatureProcessingConfig,
)


class TechnicalFeatureQualityError(ValueError):
    """Raised when technical features cannot be validated."""


@dataclass
class TechnicalFeatureQualityResult:
    """Complete technical-feature validation result."""

    is_valid: bool
    summary: dict[str, Any]
    coverage_by_feature: pd.DataFrame
    issues: tuple[str, ...]
    warnings: tuple[str, ...]


def _require_columns(
    data: pd.DataFrame,
    required_columns: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    """Require a dataset to contain all expected columns."""
    missing_columns = sorted(set(required_columns).difference(data.columns))

    if missing_columns:
        raise TechnicalFeatureQualityError(
            f"{dataset_name} is missing columns: " + ", ".join(missing_columns) + "."
        )


def validate_processed_technical_features(
    raw_features: pd.DataFrame,
    processed_features: pd.DataFrame,
    *,
    expected_tickers: pd.Series,
    processing_config: (TechnicalFeatureProcessingConfig),
) -> TechnicalFeatureQualityResult:
    """Validate transformed technical features."""
    _require_columns(
        raw_features,
        (
            "as_of_date",
            "ticker",
            "latest_market_date",
            "observations_available",
            *TECHNICAL_FEATURE_COLUMNS,
        ),
        dataset_name="Raw technical features",
    )

    _require_columns(
        processed_features,
        PROCESSED_TECHNICAL_COLUMNS,
        dataset_name=("Processed technical features"),
    )

    if raw_features.empty:
        raise TechnicalFeatureQualityError("Raw technical features are empty.")

    if processed_features.empty:
        raise TechnicalFeatureQualityError("Processed technical features are empty.")

    raw = raw_features.copy()
    processed = processed_features.copy()

    for data in (
        raw,
        processed,
    ):
        data["as_of_date"] = pd.to_datetime(
            data["as_of_date"],
            errors="coerce",
        ).dt.normalize()

        data["latest_market_date"] = pd.to_datetime(
            data["latest_market_date"],
            errors="coerce",
        ).dt.normalize()

        data["ticker"] = data["ticker"].astype("string").str.strip().str.upper()

    raw = raw.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)

    processed = processed.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)

    issues: list[str] = []
    warnings: list[str] = []

    duplicate_rows = int(
        processed.duplicated(
            subset=[
                "as_of_date",
                "ticker",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_rows:
        issues.append(f"{duplicate_rows} duplicated processed date-ticker rows were found.")

    if not processed[
        [
            "as_of_date",
            "ticker",
        ]
    ].equals(
        processed[
            [
                "as_of_date",
                "ticker",
            ]
        ]
        .sort_values(
            [
                "as_of_date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    ):
        issues.append("Processed features are not sorted by date and ticker.")

    temporal_violations = int(processed["latest_market_date"].gt(processed["as_of_date"]).sum())

    if temporal_violations:
        issues.append(f"{temporal_violations} processed rows use future observations.")

    missing_sector_rows = int(processed["sector"].isna().sum())

    if missing_sector_rows:
        issues.append(f"{missing_sector_rows} processed rows have no sector.")

    raw_keys = raw[
        [
            "as_of_date",
            "ticker",
        ]
    ]

    processed_keys = processed[
        [
            "as_of_date",
            "ticker",
        ]
    ]

    keys_match = len(raw_keys) == len(processed_keys) and raw_keys.equals(processed_keys)

    if not keys_match:
        issues.append("Raw and processed datasets do not contain identical date-ticker keys.")

    changed_raw_values = 0

    if keys_match:
        for feature in TECHNICAL_FEATURE_COLUMNS:
            observed = pd.to_numeric(
                processed[feature],
                errors="coerce",
            ).to_numpy(dtype=float)

            expected = pd.to_numeric(
                raw[feature],
                errors="coerce",
            ).to_numpy(dtype=float)

            changed_raw_values += int(
                (
                    ~np.isclose(
                        observed,
                        expected,
                        rtol=1e-12,
                        atol=1e-12,
                        equal_nan=True,
                    )
                ).sum()
            )

    if changed_raw_values:
        issues.append(f"{changed_raw_values} raw feature values changed during processing.")

    transformed_columns = (
        *WINSORIZED_FEATURE_COLUMNS,
        *STANDARDIZED_FEATURE_COLUMNS,
        *SECTOR_NEUTRAL_FEATURE_COLUMNS,
    )

    non_finite_values = 0

    for column in transformed_columns:
        values = pd.to_numeric(
            processed[column],
            errors="coerce",
        )

        non_finite_values += int(
            (values.notna() & ~np.isfinite(values.to_numpy(dtype=float))).sum()
        )

    if non_finite_values:
        issues.append(f"{non_finite_values} transformed values are non-finite.")

    incorrect_winsorized_values = 0
    invalid_standardized_dates = 0
    invalid_sector_neutral_groups = 0

    coverage_rows: list[dict[str, Any]] = []

    for feature in TECHNICAL_FEATURE_COLUMNS:
        winsorized_column = f"{feature}_winsorized"

        standardized_column = f"{feature}_zscore"

        sector_neutral_column = f"{feature}_sector_neutral"

        clipped_observations = 0

        for _, date_data in processed.groupby(
            "as_of_date",
            sort=False,
        ):
            raw_values = date_data[feature].dropna().astype(float)

            if len(raw_values) < processing_config.minimum_cross_section_size:
                continue

            lower_bound = float(raw_values.quantile(processing_config.winsor_lower_quantile))

            upper_bound = float(raw_values.quantile(processing_config.winsor_upper_quantile))

            expected_winsorized = (
                date_data[feature]
                .astype(float)
                .clip(
                    lower=lower_bound,
                    upper=upper_bound,
                )
            )

            observed_winsorized = date_data[winsorized_column].astype(float)

            incorrect_winsorized_values += int(
                (
                    ~np.isclose(
                        observed_winsorized,
                        expected_winsorized,
                        rtol=1e-12,
                        atol=1e-12,
                        equal_nan=True,
                    )
                ).sum()
            )

            comparable = date_data[feature].notna() & date_data[winsorized_column].notna()

            clipped_observations += int(
                (
                    comparable
                    & ~np.isclose(
                        date_data[feature],
                        date_data[winsorized_column],
                        rtol=1e-12,
                        atol=1e-12,
                        equal_nan=True,
                    )
                ).sum()
            )

            standardized = date_data[standardized_column].dropna().astype(float)

            if len(standardized) != len(raw_values):
                invalid_standardized_dates += 1
                continue

            if expected_winsorized.dropna().nunique() <= 1:
                if not np.allclose(
                    standardized,
                    0.0,
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    invalid_standardized_dates += 1
            else:
                mean = float(standardized.mean())

                standard_deviation = float(standardized.std(ddof=0))

                if abs(mean) > 1e-10 or not np.isclose(
                    standard_deviation,
                    1.0,
                    rtol=1e-10,
                    atol=1e-10,
                ):
                    invalid_standardized_dates += 1

        if processing_config.sector_neutralization:
            eligible_data = processed.loc[processed[sector_neutral_column].notna()]

            for _, sector_data in eligible_data.groupby(
                [
                    "as_of_date",
                    "sector",
                ],
                sort=False,
            ):
                if len(sector_data) < processing_config.minimum_sector_size:
                    continue

                sector_mean = float(sector_data[sector_neutral_column].mean())

                if abs(sector_mean) > 1e-10:
                    invalid_sector_neutral_groups += 1

        raw_non_missing = int(processed[feature].notna().sum())

        raw_missing_ratio = float(processed[feature].isna().mean())

        first_available = processed.loc[
            processed[feature].notna(),
            "as_of_date",
        ].min()

        last_available = processed.loc[
            processed[feature].notna(),
            "as_of_date",
        ].max()

        coverage_rows.append(
            {
                "feature": feature,
                "raw_non_missing": (raw_non_missing),
                "raw_missing_ratio": (raw_missing_ratio),
                "winsorized_non_missing": int(processed[winsorized_column].notna().sum()),
                "zscore_non_missing": int(processed[standardized_column].notna().sum()),
                "sector_neutral_non_missing": int(processed[sector_neutral_column].notna().sum()),
                "clipped_observations": (clipped_observations),
                "first_available_date": (first_available),
                "last_available_date": (last_available),
            }
        )

    if incorrect_winsorized_values:
        issues.append(f"{incorrect_winsorized_values} values were winsorized incorrectly.")

    if invalid_standardized_dates:
        issues.append(
            f"{invalid_standardized_dates} "
            "feature-date groups have invalid "
            "cross-sectional standardization."
        )

    if invalid_sector_neutral_groups:
        issues.append(
            f"{invalid_sector_neutral_groups} eligible sector groups are not centered on zero."
        )

    coverage_by_feature = pd.DataFrame(coverage_rows)

    expected_ticker_set = {str(ticker).strip().upper() for ticker in expected_tickers}

    observed_ticker_set = set(processed["ticker"].dropna())

    missing_tickers = sorted(expected_ticker_set.difference(observed_ticker_set))

    unexpected_tickers = sorted(observed_ticker_set.difference(expected_ticker_set))

    if missing_tickers:
        issues.append(
            "Expected tickers missing from the "
            "processed dataset: " + ", ".join(missing_tickers) + "."
        )

    if unexpected_tickers:
        issues.append("Unexpected tickers found: " + ", ".join(unexpected_tickers) + ".")

    coverage_by_date = processed.groupby(
        "as_of_date",
        sort=True,
    )["ticker"].nunique()

    incomplete_universe_dates = int(coverage_by_date.ne(len(expected_ticker_set)).sum())

    if incomplete_universe_dates:
        warnings.append(
            f"{incomplete_universe_dates} dates do not contain the full expected universe."
        )

    sector_group_sizes = processed.groupby(
        [
            "as_of_date",
            "sector",
        ],
        sort=False,
    )["ticker"].nunique()

    small_sector_groups = int(sector_group_sizes.lt(processing_config.minimum_sector_size).sum())

    if small_sector_groups:
        warnings.append(
            f"{small_sector_groups} date-sector "
            "groups are too small for sector "
            "neutralization and retain their "
            "global z-score."
        )

    summary: dict[str, Any] = {
        "rows": len(processed),
        "dates": processed["as_of_date"].nunique(),
        "tickers": processed["ticker"].nunique(),
        "sectors": processed["sector"].nunique(),
        "raw_features": len(TECHNICAL_FEATURE_COLUMNS),
        "model_features": len(SECTOR_NEUTRAL_FEATURE_COLUMNS),
        "first_as_of_date": processed["as_of_date"].min(),
        "last_as_of_date": processed["as_of_date"].max(),
        "duplicate_rows": duplicate_rows,
        "temporal_violations": (temporal_violations),
        "missing_sector_rows": (missing_sector_rows),
        "changed_raw_values": (changed_raw_values),
        "non_finite_values": (non_finite_values),
        "incorrect_winsorized_values": (incorrect_winsorized_values),
        "invalid_standardized_dates": (invalid_standardized_dates),
        "invalid_sector_neutral_groups": (invalid_sector_neutral_groups),
        "incomplete_universe_dates": (incomplete_universe_dates),
        "small_sector_groups": (small_sector_groups),
    }

    return TechnicalFeatureQualityResult(
        is_valid=not issues,
        summary=summary,
        coverage_by_feature=(coverage_by_feature),
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


def _format_markdown_value(
    value: Any,
) -> str:
    """Format a scalar for Markdown output."""
    if value is None or pd.isna(value):
        return ""

    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()

    if isinstance(value, float):
        return f"{value:.12f}"

    return str(value).replace(
        "|",
        "\\|",
    )


def _dataframe_to_markdown(
    frame: pd.DataFrame,
) -> str:
    """Convert a DataFrame to a Markdown table."""
    if frame.empty:
        return "_No observations._"

    columns = [str(column) for column in frame.columns]

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in frame.itertuples(
        index=False,
        name=None,
    ):
        values = [_format_markdown_value(value) for value in row]

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def write_technical_features_report(
    result: TechnicalFeatureQualityResult,
    path: Path,
    *,
    processing_config: (TechnicalFeatureProcessingConfig),
) -> Path:
    """Write the technical-feature quality report."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    status = "PASS" if result.is_valid else "FAIL"

    summary_frame = pd.DataFrame(
        [
            {
                "metric": metric,
                "value": value,
            }
            for metric, value in result.summary.items()
        ]
    )

    lines = [
        "# Technical Features Quality Report",
        "",
        ("- Generated UTC: " + datetime.now(UTC).isoformat(timespec="seconds")),
        f"- Status: **{status}**",
        (
            "- Winsorization: "
            f"{processing_config.winsor_lower_quantile:.2%} "
            "to "
            f"{processing_config.winsor_upper_quantile:.2%}"
        ),
        (f"- Minimum cross-section size: `{processing_config.minimum_cross_section_size}`"),
        (f"- Sector neutralization: `{processing_config.sector_neutralization}`"),
        "",
        "## Summary",
        "",
        _dataframe_to_markdown(summary_frame),
        "",
        "## Blocking issues",
        "",
    ]

    if result.issues:
        lines.extend(f"- {issue}" for issue in result.issues)
    else:
        lines.append("- No blocking validation issues.")

    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )

    if result.warnings:
        lines.extend(f"- {warning}" for warning in result.warnings)
    else:
        lines.append("- No warnings.")

    lines.extend(
        [
            "",
            "## Coverage by feature",
            "",
            _dataframe_to_markdown(result.coverage_by_feature),
            "",
            "## Processing interpretation",
            "",
            (
                "Winsorization, standardization and "
                "sector adjustment are calculated "
                "independently for every rebalance date."
            ),
            "",
            ("No future month contributes to the transformation of a previous month."),
            "",
            ("Raw feature values are retained in the processed dataset for auditability."),
            "",
            (
                "Missing feature values remain missing. "
                "No future-aware imputation is applied "
                "during this processing stage."
            ),
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path
