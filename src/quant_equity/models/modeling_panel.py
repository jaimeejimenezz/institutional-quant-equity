"""Master modeling panel for technical and fundamental signals."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_equity.features.fundamental_transforms import (
    FUNDAMENTAL_FACTOR_COLUMNS,
)
from quant_equity.features.technical_processing import (
    TECHNICAL_MODEL_FEATURE_COLUMNS,
)


class ModelingPanelError(ValueError):
    """Raised when the master modeling panel cannot be built."""


TECHNICAL_SOURCE_MODEL_COLUMNS = tuple(
    TECHNICAL_MODEL_FEATURE_COLUMNS
)

FUNDAMENTAL_GLOBAL_SOURCE_COLUMNS = tuple(
    f"{factor}_zscore"
    for factor in FUNDAMENTAL_FACTOR_COLUMNS
)

FUNDAMENTAL_SECTOR_SOURCE_COLUMNS = tuple(
    f"{factor}_sector_zscore"
    for factor in FUNDAMENTAL_FACTOR_COLUMNS
)

FUNDAMENTAL_MISSING_SOURCE_COLUMNS = tuple(
    f"{factor}_missing"
    for factor in FUNDAMENTAL_FACTOR_COLUMNS
)

TECHNICAL_PANEL_MODEL_COLUMNS = tuple(
    f"tech__{column}"
    for column in TECHNICAL_SOURCE_MODEL_COLUMNS
)

FUNDAMENTAL_GLOBAL_PANEL_COLUMNS = tuple(
    f"fund__{column}"
    for column in FUNDAMENTAL_GLOBAL_SOURCE_COLUMNS
)

FUNDAMENTAL_SECTOR_PANEL_COLUMNS = tuple(
    f"fund__{column}"
    for column in FUNDAMENTAL_SECTOR_SOURCE_COLUMNS
)

FUNDAMENTAL_MISSING_PANEL_COLUMNS = tuple(
    f"fund__{column}"
    for column in FUNDAMENTAL_MISSING_SOURCE_COLUMNS
)

MODEL_FEATURE_COLUMNS = (
    *TECHNICAL_PANEL_MODEL_COLUMNS,
    *FUNDAMENTAL_GLOBAL_PANEL_COLUMNS,
    *FUNDAMENTAL_SECTOR_PANEL_COLUMNS,
    *FUNDAMENTAL_MISSING_PANEL_COLUMNS,
)

TARGET_VALUE_COLUMNS = (
    "target_21d",
    "target_21d_excess",
    "label_top_quintile",
)

TARGET_METADATA_COLUMNS = (
    "first_future_date",
    "target_end_date",
    "horizon_sessions",
)

PANEL_IDENTIFIER_COLUMNS = (
    "as_of_date",
    "ticker",
    "sector",
    "technical_latest_market_date",
    "observations_available",
)

PANEL_DIAGNOSTIC_COLUMNS = (
    "technical_missing_count",
    "fundamental_global_missing_count",
    "fundamental_sector_missing_count",
    "model_feature_missing_count",
    "has_target",
    "sample_role",
)

MODELING_PANEL_COLUMNS = (
    *PANEL_IDENTIFIER_COLUMNS,
    *MODEL_FEATURE_COLUMNS,
    *TARGET_METADATA_COLUMNS,
    *TARGET_VALUE_COLUMNS,
    *PANEL_DIAGNOSTIC_COLUMNS,
)


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    """Require a dataset to contain expected columns."""
    missing = sorted(
        set(columns).difference(
            data.columns
        )
    )

    if missing:
        raise ModelingPanelError(
            f"{dataset_name} is missing columns: "
            + ", ".join(
                missing
            )
            + "."
        )


def _normalize_keys(
    data: pd.DataFrame,
    *,
    dataset_name: str,
) -> pd.DataFrame:
    """Normalize date and ticker keys."""
    frame = data.copy()

    frame["as_of_date"] = pd.to_datetime(
        frame["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    frame["ticker"] = (
        frame["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    invalid_rows = (
        frame["as_of_date"].isna()
        | frame["ticker"].isna()
        | frame["ticker"].eq("")
    )

    if invalid_rows.any():
        raise ModelingPanelError(
            f"{dataset_name} contains "
            f"{int(invalid_rows.sum())} rows "
            "with invalid keys."
        )

    return frame


def _assert_unique_keys(
    data: pd.DataFrame,
    *,
    dataset_name: str,
) -> None:
    """Require one row per date and ticker."""
    duplicated = int(
        data.duplicated(
            [
                "as_of_date",
                "ticker",
            ],
            keep=False,
        ).sum()
    )

    if duplicated:
        raise ModelingPanelError(
            f"{dataset_name} contains "
            f"{duplicated} duplicated "
            "date-ticker rows."
        )


def _sorted_keys(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Return sorted date-ticker keys."""
    return (
        data.loc[
            :,
            [
                "as_of_date",
                "ticker",
            ],
        ]
        .sort_values(
            [
                "as_of_date",
                "ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def _validate_matching_feature_keys(
    technical: pd.DataFrame,
    fundamental: pd.DataFrame,
) -> None:
    """Require technical and fundamental panels to align."""
    technical_keys = _sorted_keys(
        technical
    )

    fundamental_keys = _sorted_keys(
        fundamental
    )

    if (
        len(
            technical_keys
        )
        != len(
            fundamental_keys
        )
        or not technical_keys.equals(
            fundamental_keys
        )
    ):
        raise ModelingPanelError(
            "Technical and fundamental feature "
            "panels do not contain identical "
            "date-ticker keys."
        )


def _validate_label_subset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    """Require every label key to exist in the feature panel."""
    feature_keys = (
        _sorted_keys(
            features
        )
        .assign(
            _feature_key=1
        )
    )

    label_keys = _sorted_keys(
        labels
    )

    comparison = label_keys.merge(
        feature_keys,
        on=[
            "as_of_date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    missing = int(
        comparison[
            "_feature_key"
        ].isna().sum()
    )

    if missing:
        raise ModelingPanelError(
            f"{missing} label rows do not "
            "have a matching feature row."
        )


def build_modeling_panel(
    technical_features: pd.DataFrame,
    fundamental_features: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Build the complete monthly modeling panel."""
    technical_required = (
        "as_of_date",
        "ticker",
        "sector",
        "latest_market_date",
        "observations_available",
        *TECHNICAL_SOURCE_MODEL_COLUMNS,
    )

    fundamental_required = (
        "as_of_date",
        "ticker",
        "sector",
        *FUNDAMENTAL_GLOBAL_SOURCE_COLUMNS,
        *FUNDAMENTAL_SECTOR_SOURCE_COLUMNS,
        *FUNDAMENTAL_MISSING_SOURCE_COLUMNS,
    )

    label_required = (
        "as_of_date",
        "ticker",
        *TARGET_METADATA_COLUMNS,
        *TARGET_VALUE_COLUMNS,
    )

    _require_columns(
        technical_features,
        technical_required,
        dataset_name="Technical features",
    )

    _require_columns(
        fundamental_features,
        fundamental_required,
        dataset_name="Fundamental features",
    )

    _require_columns(
        labels,
        label_required,
        dataset_name="Monthly labels",
    )

    technical = _normalize_keys(
        technical_features.loc[
            :,
            technical_required,
        ],
        dataset_name="Technical features",
    )

    fundamental = _normalize_keys(
        fundamental_features.loc[
            :,
            fundamental_required,
        ],
        dataset_name="Fundamental features",
    )

    monthly_labels = _normalize_keys(
        labels.loc[
            :,
            label_required,
        ],
        dataset_name="Monthly labels",
    )

    _assert_unique_keys(
        technical,
        dataset_name="Technical features",
    )

    _assert_unique_keys(
        fundamental,
        dataset_name="Fundamental features",
    )

    _assert_unique_keys(
        monthly_labels,
        dataset_name="Monthly labels",
    )

    _validate_matching_feature_keys(
        technical,
        fundamental,
    )

    _validate_label_subset(
        technical,
        monthly_labels,
    )

    technical[
        "latest_market_date"
    ] = pd.to_datetime(
        technical[
            "latest_market_date"
        ],
        errors="coerce",
    ).dt.normalize()

    market_violations = int(
        technical[
            "latest_market_date"
        ]
        .gt(
            technical[
                "as_of_date"
            ]
        )
        .sum()
    )

    if market_violations:
        raise ModelingPanelError(
            f"{market_violations} technical rows "
            "use future market information."
        )

    technical = technical.rename(
        columns={
            "latest_market_date": (
                "technical_latest_market_date"
            ),
            **{
                column: (
                    f"tech__{column}"
                )
                for column in (
                    TECHNICAL_SOURCE_MODEL_COLUMNS
                )
            },
        }
    )

    fundamental = fundamental.rename(
        columns={
            "sector": "fundamental_sector",
            **{
                column: (
                    f"fund__{column}"
                )
                for column in (
                    *FUNDAMENTAL_GLOBAL_SOURCE_COLUMNS,
                    *FUNDAMENTAL_SECTOR_SOURCE_COLUMNS,
                    *FUNDAMENTAL_MISSING_SOURCE_COLUMNS,
                )
            },
        }
    )

    panel = technical.merge(
        fundamental,
        on=[
            "as_of_date",
            "ticker",
        ],
        how="inner",
        validate="one_to_one",
    )

    sector_mismatch = (
        panel[
            "sector"
        ].notna()
        & panel[
            "fundamental_sector"
        ].notna()
        & panel[
            "sector"
        ].ne(
            panel[
                "fundamental_sector"
            ]
        )
    )

    if sector_mismatch.any():
        raise ModelingPanelError(
            f"{int(sector_mismatch.sum())} rows "
            "have inconsistent technical and "
            "fundamental sectors."
        )

    panel = panel.drop(
        columns=[
            "fundamental_sector",
        ]
    )

    for column in (
        FUNDAMENTAL_MISSING_PANEL_COLUMNS
    ):
        values = pd.to_numeric(
            panel[column],
            errors="coerce",
        )

        invalid = (
            values.isna()
            | ~values.isin(
                [
                    0,
                    1,
                ]
            )
        )

        if invalid.any():
            raise ModelingPanelError(
                f"Missingness feature {column} "
                "contains invalid values."
            )

        panel[column] = (
            values.astype(
                "int8"
            )
        )

    for column in (
        "first_future_date",
        "target_end_date",
    ):
        monthly_labels[column] = (
            pd.to_datetime(
                monthly_labels[column],
                errors="coerce",
            ).dt.normalize()
        )

    panel = panel.merge(
        monthly_labels,
        on=[
            "as_of_date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    target_presence = panel.loc[
        :,
        TARGET_VALUE_COLUMNS,
    ].notna()

    partial_targets = (
        target_presence.any(
            axis=1
        )
        & ~target_presence.all(
            axis=1
        )
    )

    if partial_targets.any():
        raise ModelingPanelError(
            f"{int(partial_targets.sum())} rows "
            "contain incomplete targets."
        )

    panel[
        "has_target"
    ] = (
        target_presence.all(
            axis=1
        )
        .astype(
            "int8"
        )
    )

    target_rows = panel[
        "has_target"
    ].eq(
        1
    )

    invalid_target_dates = (
        target_rows
        & (
            panel[
                "first_future_date"
            ].isna()
            | panel[
                "target_end_date"
            ].isna()
            | panel[
                "first_future_date"
            ].le(
                panel[
                    "as_of_date"
                ]
            )
            | panel[
                "target_end_date"
            ].lt(
                panel[
                    "first_future_date"
                ]
            )
        )
    )

    if invalid_target_dates.any():
        raise ModelingPanelError(
            f"{int(invalid_target_dates.sum())} rows "
            "have invalid future target dates."
        )

    panel[
        "sample_role"
    ] = "inference_only"

    panel.loc[
        target_rows,
        "sample_role",
    ] = "modeling"

    panel[
        "technical_missing_count"
    ] = (
        panel.loc[
            :,
            TECHNICAL_PANEL_MODEL_COLUMNS,
        ]
        .isna()
        .sum(
            axis=1
        )
        .astype(
            "int16"
        )
    )

    panel[
        "fundamental_global_missing_count"
    ] = (
        panel.loc[
            :,
            FUNDAMENTAL_GLOBAL_PANEL_COLUMNS,
        ]
        .isna()
        .sum(
            axis=1
        )
        .astype(
            "int16"
        )
    )

    panel[
        "fundamental_sector_missing_count"
    ] = (
        panel.loc[
            :,
            FUNDAMENTAL_SECTOR_PANEL_COLUMNS,
        ]
        .isna()
        .sum(
            axis=1
        )
        .astype(
            "int16"
        )
    )

    panel[
        "model_feature_missing_count"
    ] = (
        panel.loc[
            :,
            MODEL_FEATURE_COLUMNS,
        ]
        .isna()
        .sum(
            axis=1
        )
        .astype(
            "int16"
        )
    )

    panel[
        "label_top_quintile"
    ] = pd.to_numeric(
        panel[
            "label_top_quintile"
        ],
        errors="coerce",
    ).astype(
        "Int8"
    )

    duplicates = int(
        panel.duplicated(
            [
                "as_of_date",
                "ticker",
            ]
        ).sum()
    )

    if duplicates:
        raise ModelingPanelError(
            f"Master panel contains "
            f"{duplicates} duplicate keys."
        )

    return (
        panel.loc[
            :,
            MODELING_PANEL_COLUMNS,
        ]
        .sort_values(
            [
                "as_of_date",
                "ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def write_modeling_panel(
    panel: pd.DataFrame,
    path: Path,
) -> Path:
    """Write the master modeling panel atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered = (
        panel.sort_values(
            [
                "as_of_date",
                "ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    temporary = path.with_suffix(
        ".tmp.parquet"
    )

    temporary.unlink(
        missing_ok=True
    )

    ordered.to_parquet(
        temporary,
        index=False,
    )

    temporary.replace(
        path
    )

    return path