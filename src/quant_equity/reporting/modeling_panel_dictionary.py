"""Documentation and coverage for the master modeling panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_equity.features import (
    FUNDAMENTAL_FEATURE_METADATA,
)
from quant_equity.features.fundamental_transforms import (
    FUNDAMENTAL_FACTOR_COLUMNS,
)
from quant_equity.models.modeling_panel import (
    MODEL_FEATURE_COLUMNS,
    MODELING_PANEL_COLUMNS,
    TECHNICAL_PANEL_MODEL_COLUMNS,
)


class ModelingPanelDictionaryError(ValueError):
    """Raised when modeling-panel documentation is inconsistent."""


def _fixed_definitions() -> dict[
    str,
    dict[str, Any],
]:
    """Return metadata for non-feature columns."""
    return {
        "as_of_date": {
            "role": "identifier",
            "feature_group": "identifier",
            "family": "Calendar",
            "source_dataset": ("data/processed/rebalance_calendar.parquet"),
            "source_column": "as_of_date",
            "model_input": False,
            "availability_rule": (
                "Monthly rebalance date; all predictors must be available on or before this date."
            ),
            "calculation": (
                "Last available market session selected for the monthly rebalance observation."
            ),
            "missing_semantics": "Must never be missing.",
        },
        "ticker": {
            "role": "identifier",
            "feature_group": "identifier",
            "family": "Security",
            "source_dataset": ("data/reference/universe_v1.csv"),
            "source_column": "ticker",
            "model_input": False,
            "availability_rule": ("Security identifier attached to each monthly observation."),
            "calculation": ("Normalized uppercase ticker from the configured equity universe."),
            "missing_semantics": "Must never be missing.",
        },
        "sector": {
            "role": "metadata",
            "feature_group": "metadata",
            "family": "Security",
            "source_dataset": ("data/reference/universe_v1.csv"),
            "source_column": "sector",
            "model_input": False,
            "availability_rule": ("Universe metadata known for the security."),
            "calculation": ("Sector classification joined by ticker."),
            "missing_semantics": "Must never be missing.",
        },
        "technical_latest_market_date": {
            "role": "provenance",
            "feature_group": "provenance",
            "family": "Technical",
            "source_dataset": ("data/processed/features_technical_monthly.parquet"),
            "source_column": "latest_market_date",
            "model_input": False,
            "availability_rule": ("Must be less than or equal to as_of_date."),
            "calculation": (
                "Latest daily market observation used by the technical-feature calculation."
            ),
            "missing_semantics": "Must never be missing.",
        },
        "observations_available": {
            "role": "provenance",
            "feature_group": "provenance",
            "family": "Technical",
            "source_dataset": ("data/processed/features_technical_monthly.parquet"),
            "source_column": "observations_available",
            "model_input": False,
            "availability_rule": ("Counts historical observations available up to as_of_date."),
            "calculation": (
                "Number of historical market observations available to technical calculations."
            ),
            "missing_semantics": "Must never be missing.",
        },
        "first_future_date": {
            "role": "target_metadata",
            "feature_group": "target_metadata",
            "family": "Target",
            "source_dataset": ("data/processed/labels_monthly.parquet"),
            "source_column": "first_future_date",
            "model_input": False,
            "availability_rule": (
                "Future information. Never permitted inside the predictor matrix."
            ),
            "calculation": ("First market session strictly after as_of_date."),
            "missing_semantics": ("Missing for inference-only observations."),
        },
        "target_end_date": {
            "role": "target_metadata",
            "feature_group": "target_metadata",
            "family": "Target",
            "source_dataset": ("data/processed/labels_monthly.parquet"),
            "source_column": "target_end_date",
            "model_input": False,
            "availability_rule": (
                "Future information. Never permitted inside the predictor matrix."
            ),
            "calculation": (
                "Market date corresponding to the end of the configured forward horizon."
            ),
            "missing_semantics": ("Missing when a complete future horizon is not yet available."),
        },
        "horizon_sessions": {
            "role": "target_metadata",
            "feature_group": "target_metadata",
            "family": "Target",
            "source_dataset": ("data/processed/labels_monthly.parquet"),
            "source_column": "horizon_sessions",
            "model_input": False,
            "availability_rule": ("Target definition; does not enter the predictor matrix."),
            "calculation": ("Configured number of future market sessions used by the target."),
            "missing_semantics": ("Missing for inference-only observations."),
        },
        "target_21d": {
            "role": "target",
            "feature_group": "target",
            "family": "Target",
            "source_dataset": ("data/processed/labels_monthly.parquet"),
            "source_column": "target_21d",
            "model_input": False,
            "availability_rule": (
                "Known only after the complete future 21-session horizon has occurred."
            ),
            "calculation": ("Adjusted-close return from as_of_date to target_end_date."),
            "missing_semantics": ("Missing for inference-only observations."),
        },
        "target_21d_excess": {
            "role": "target",
            "feature_group": "target",
            "family": "Target",
            "source_dataset": ("data/processed/labels_monthly.parquet"),
            "source_column": "target_21d_excess",
            "model_input": False,
            "availability_rule": (
                "Known only after the complete future 21-session horizon has occurred."
            ),
            "calculation": (
                "target_21d minus the cross-sectional median target_21d on the same as_of_date."
            ),
            "missing_semantics": ("Missing for inference-only observations."),
        },
        "label_top_quintile": {
            "role": "target",
            "feature_group": "target",
            "family": "Target",
            "source_dataset": ("data/processed/labels_monthly.parquet"),
            "source_column": "label_top_quintile",
            "model_input": False,
            "availability_rule": (
                "Known only after the complete future 21-session horizon has occurred."
            ),
            "calculation": (
                "Binary label equal to 1 for securities "
                "inside the top 20% of future returns "
                "on the same rebalance date."
            ),
            "missing_semantics": ("Missing for inference-only observations."),
        },
        "technical_missing_count": {
            "role": "diagnostic",
            "feature_group": "diagnostic",
            "family": "Data quality",
            "source_dataset": ("data/processed/modeling_panel.parquet"),
            "source_column": "Derived",
            "model_input": False,
            "availability_rule": ("Calculated from predictor availability at as_of_date."),
            "calculation": ("Number of missing technical predictor values in the row."),
            "missing_semantics": "Must never be missing.",
        },
        "fundamental_global_missing_count": {
            "role": "diagnostic",
            "feature_group": "diagnostic",
            "family": "Data quality",
            "source_dataset": ("data/processed/modeling_panel.parquet"),
            "source_column": "Derived",
            "model_input": False,
            "availability_rule": ("Calculated from predictor availability at as_of_date."),
            "calculation": ("Number of missing global fundamental z-scores in the row."),
            "missing_semantics": "Must never be missing.",
        },
        "fundamental_sector_missing_count": {
            "role": "diagnostic",
            "feature_group": "diagnostic",
            "family": "Data quality",
            "source_dataset": ("data/processed/modeling_panel.parquet"),
            "source_column": "Derived",
            "model_input": False,
            "availability_rule": ("Calculated from predictor availability at as_of_date."),
            "calculation": ("Number of missing sector-relative fundamental z-scores in the row."),
            "missing_semantics": "Must never be missing.",
        },
        "model_feature_missing_count": {
            "role": "diagnostic",
            "feature_group": "diagnostic",
            "family": "Data quality",
            "source_dataset": ("data/processed/modeling_panel.parquet"),
            "source_column": "Derived",
            "model_input": False,
            "availability_rule": (
                "Calculated from all candidate predictors available at as_of_date."
            ),
            "calculation": ("Total number of missing values among the candidate model features."),
            "missing_semantics": "Must never be missing.",
        },
        "has_target": {
            "role": "diagnostic",
            "feature_group": "diagnostic",
            "family": "Sample",
            "source_dataset": ("data/processed/modeling_panel.parquet"),
            "source_column": "Derived",
            "model_input": False,
            "availability_rule": (
                "Describes whether the full future target exists; never used as a predictor."
            ),
            "calculation": ("1 when all target fields are available, otherwise 0."),
            "missing_semantics": "Must never be missing.",
        },
        "sample_role": {
            "role": "diagnostic",
            "feature_group": "diagnostic",
            "family": "Sample",
            "source_dataset": ("data/processed/modeling_panel.parquet"),
            "source_column": "Derived",
            "model_input": False,
            "availability_rule": ("Describes sample eligibility; never used as a predictor."),
            "calculation": ("'modeling' when has_target=1 and 'inference_only' otherwise."),
            "missing_semantics": "Must never be missing.",
        },
    }


def _technical_definition(
    column: str,
) -> dict[str, Any]:
    """Describe one processed technical predictor."""
    prefix = "tech__"

    source_column = column[len(prefix) :]

    suffix = "_sector_neutral"

    if not source_column.endswith(suffix):
        raise ModelingPanelDictionaryError(f"Unexpected technical feature: {column}")

    raw_factor = source_column[: -len(suffix)]

    return {
        "role": "predictor",
        "feature_group": "technical",
        "family": "Technical",
        "source_dataset": ("data/processed/features_technical_monthly.parquet"),
        "source_column": source_column,
        "model_input": True,
        "availability_rule": (
            "Uses market observations only through "
            "technical_latest_market_date, which must "
            "be <= as_of_date."
        ),
        "calculation": (
            f"Raw technical factor '{raw_factor}' is "
            "winsorized cross-sectionally by date, "
            "standardized by date, then sector-neutralized. "
            "The raw factor formula is documented in "
            "docs/TECHNICAL_FEATURES.md."
        ),
        "missing_semantics": (
            "NaN generally indicates insufficient historical "
            "market observations or an unavailable underlying "
            "technical calculation."
        ),
    }


def _fundamental_global_definition(
    factor: str,
) -> dict[str, Any]:
    """Describe one global fundamental z-score."""
    metadata = FUNDAMENTAL_FEATURE_METADATA[factor]

    return {
        "role": "predictor",
        "feature_group": "fundamental_global",
        "family": metadata["family"],
        "source_dataset": ("data/processed/features_fundamental_monthly.parquet"),
        "source_column": (f"{factor}_zscore"),
        "model_input": True,
        "availability_rule": (
            "Uses only fundamental information that was available on or before as_of_date."
        ),
        "calculation": (
            f"{metadata['formula']}. The raw factor is "
            "winsorized by rebalance date and standardized "
            "across available companies on that date."
        ),
        "missing_semantics": (
            "NaN means the underlying accounting factor "
            "or sufficient cross-sectional information "
            "was unavailable."
        ),
    }


def _fundamental_sector_definition(
    factor: str,
) -> dict[str, Any]:
    """Describe one sector fundamental z-score."""
    metadata = FUNDAMENTAL_FEATURE_METADATA[factor]

    return {
        "role": "predictor",
        "feature_group": "fundamental_sector",
        "family": metadata["family"],
        "source_dataset": ("data/processed/features_fundamental_monthly.parquet"),
        "source_column": (f"{factor}_sector_zscore"),
        "model_input": True,
        "availability_rule": (
            "Uses only fundamental information available "
            "on or before as_of_date and compares companies "
            "within the same sector and rebalance date."
        ),
        "calculation": (
            f"{metadata['formula']}. The factor is "
            "winsorized and standardized relative to "
            "available companies in the same sector "
            "and rebalance date."
        ),
        "missing_semantics": (
            "NaN means the raw factor was unavailable or the sector cross-section was insufficient."
        ),
    }


def _fundamental_missing_definition(
    factor: str,
) -> dict[str, Any]:
    """Describe one fundamental missingness flag."""
    metadata = FUNDAMENTAL_FEATURE_METADATA[factor]

    return {
        "role": "predictor",
        "feature_group": "fundamental_missing",
        "family": metadata["family"],
        "source_dataset": ("data/processed/features_fundamental_monthly.parquet"),
        "source_column": (f"{factor}_missing"),
        "model_input": True,
        "availability_rule": (
            "Derived solely from availability of the point-in-time raw fundamental factor."
        ),
        "calculation": (f"Binary flag equal to 1 when '{factor}' is unavailable and 0 otherwise."),
        "missing_semantics": ("The flag itself must never be missing."),
    }


def _feature_definitions() -> dict[
    str,
    dict[str, Any],
]:
    """Build predictor metadata."""
    definitions: dict[
        str,
        dict[str, Any],
    ] = {}

    for column in TECHNICAL_PANEL_MODEL_COLUMNS:
        definitions[column] = _technical_definition(column)

    for factor in FUNDAMENTAL_FACTOR_COLUMNS:
        definitions[f"fund__{factor}_zscore"] = _fundamental_global_definition(factor)

        definitions[f"fund__{factor}_sector_zscore"] = _fundamental_sector_definition(factor)

        definitions[f"fund__{factor}_missing"] = _fundamental_missing_definition(factor)

    return definitions


def build_modeling_panel_data_dictionary(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Build column-level provenance and coverage metadata."""
    missing_columns = sorted(set(MODELING_PANEL_COLUMNS).difference(panel.columns))

    if missing_columns:
        raise ModelingPanelDictionaryError(
            "Modeling panel is missing columns: " + ", ".join(missing_columns) + "."
        )

    if not panel.columns.is_unique:
        raise ModelingPanelDictionaryError("Modeling panel contains duplicated column names.")

    frame = panel.copy()

    frame["as_of_date"] = pd.to_datetime(
        frame["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    latest_date = frame["as_of_date"].max()

    latest = frame.loc[frame["as_of_date"].eq(latest_date)]

    definitions = {
        **_fixed_definitions(),
        **_feature_definitions(),
    }

    expected = set(MODELING_PANEL_COLUMNS)

    documented = set(definitions)

    missing_definitions = expected - documented

    unexpected_definitions = documented - expected

    if missing_definitions or unexpected_definitions:
        raise ModelingPanelDictionaryError(
            "Data dictionary definition mismatch. "
            f"Missing: {sorted(missing_definitions)}. "
            f"Unexpected: {sorted(unexpected_definitions)}."
        )

    rows = []

    for position, column in enumerate(
        MODELING_PANEL_COLUMNS,
        start=1,
    ):
        definition = definitions[column]

        available = frame[column].notna()

        latest_available = latest[column].notna()

        valid_dates = frame.loc[
            available,
            "as_of_date",
        ]

        rows.append(
            {
                "position": position,
                "column": column,
                "dtype": str(frame[column].dtype),
                **definition,
                "non_missing_count": int(available.sum()),
                "missing_count": int((~available).sum()),
                "overall_coverage": float(available.mean()),
                "latest_coverage": float(latest_available.mean()),
                "first_valid_date": (valid_dates.min() if not valid_dates.empty else pd.NaT),
                "last_valid_date": (valid_dates.max() if not valid_dates.empty else pd.NaT),
            }
        )

    result = pd.DataFrame(rows)

    validate_modeling_panel_data_dictionary(result)

    return result


def validate_modeling_panel_data_dictionary(
    dictionary: pd.DataFrame,
) -> None:
    """Validate complete dictionary coverage."""
    required = {
        "position",
        "column",
        "role",
        "feature_group",
        "family",
        "source_dataset",
        "source_column",
        "model_input",
        "availability_rule",
        "calculation",
        "missing_semantics",
        "overall_coverage",
        "latest_coverage",
    }

    missing = sorted(required.difference(dictionary.columns))

    if missing:
        raise ModelingPanelDictionaryError(
            "Data dictionary is missing fields: " + ", ".join(missing) + "."
        )

    duplicate_columns = int(dictionary["column"].duplicated(keep=False).sum())

    if duplicate_columns:
        raise ModelingPanelDictionaryError("Data dictionary contains duplicated columns.")

    expected = set(MODELING_PANEL_COLUMNS)

    documented = set(dictionary["column"])

    if documented != expected:
        raise ModelingPanelDictionaryError(
            "Data dictionary does not exactly cover the modeling panel."
        )

    predictor_columns = set(
        dictionary.loc[
            dictionary["model_input"].eq(True),
            "column",
        ]
    )

    if predictor_columns != set(MODEL_FEATURE_COLUMNS):
        raise ModelingPanelDictionaryError(
            "Model-input metadata does not match MODEL_FEATURE_COLUMNS."
        )

    invalid_coverage = (
        dictionary["overall_coverage"].lt(0.0)
        | dictionary["overall_coverage"].gt(1.0)
        | dictionary["latest_coverage"].lt(0.0)
        | dictionary["latest_coverage"].gt(1.0)
    )

    if invalid_coverage.any():
        raise ModelingPanelDictionaryError("Coverage values must be in [0, 1].")


def build_modeling_feature_coverage(
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """Return coverage information for candidate predictors."""
    validate_modeling_panel_data_dictionary(dictionary)

    return (
        dictionary.loc[
            dictionary["model_input"].eq(True),
            [
                "column",
                "feature_group",
                "family",
                "source_column",
                "dtype",
                "non_missing_count",
                "missing_count",
                "overall_coverage",
                "latest_coverage",
                "first_valid_date",
                "last_valid_date",
            ],
        ]
        .sort_values(
            [
                "feature_group",
                "overall_coverage",
                "column",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def _escape_markdown(
    value: Any,
) -> str:
    """Escape simple values for markdown tables."""
    if pd.isna(value):
        return ""

    return (
        str(value)
        .replace(
            "|",
            "\\|",
        )
        .replace(
            "\n",
            " ",
        )
    )


def _markdown_table(
    data: pd.DataFrame,
) -> str:
    """Render a compact markdown table without extra dependencies."""
    if data.empty:
        return "_None_"

    columns = list(data.columns)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in data.itertuples(
        index=False,
        name=None,
    ):
        lines.append("| " + " | ".join(_escape_markdown(value) for value in row) + " |")

    return "\n".join(lines)


def render_data_dictionary(
    panel: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> str:
    """Render docs/DATA_DICTIONARY.md."""
    validate_modeling_panel_data_dictionary(dictionary)

    frame = panel.copy()

    frame["as_of_date"] = pd.to_datetime(
        frame["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    feature_coverage = build_modeling_feature_coverage(dictionary)

    coverage_summary = feature_coverage.groupby(
        "feature_group",
        as_index=False,
    ).agg(
        features=(
            "column",
            "count",
        ),
        mean_coverage=(
            "overall_coverage",
            "mean",
        ),
        minimum_coverage=(
            "overall_coverage",
            "min",
        ),
        latest_mean_coverage=(
            "latest_coverage",
            "mean",
        ),
    )

    coverage_display = coverage_summary.copy()

    for column in (
        "mean_coverage",
        "minimum_coverage",
        "latest_mean_coverage",
    ):
        coverage_display[column] = coverage_display[column].map(lambda value: f"{value:.2%}")

    compact_dictionary = dictionary.loc[
        :,
        [
            "column",
            "role",
            "feature_group",
            "family",
            "source_column",
            "overall_coverage",
            "latest_coverage",
        ],
    ].copy()

    for column in (
        "overall_coverage",
        "latest_coverage",
    ):
        compact_dictionary[column] = compact_dictionary[column].map(lambda value: f"{value:.2%}")

    lines = [
        "# Data Dictionary — Modeling Panel",
        "",
        "## Dataset",
        "",
        (
            "`data/processed/modeling_panel.parquet` "
            "is the canonical monthly dataset used by "
            "the predictive-modeling stages."
        ),
        "",
        f"- Rows: `{len(frame)}`",
        (f"- Rebalance dates: `{frame['as_of_date'].nunique()}`"),
        (f"- Companies: `{frame['ticker'].nunique()}`"),
        (f"- Candidate model features: `{len(MODEL_FEATURE_COLUMNS)}`"),
        (f"- Modeling rows: `{int(frame['has_target'].sum())}`"),
        (f"- Inference-only rows: `{int(frame['has_target'].eq(0).sum())}`"),
        "",
        "## Temporal contract",
        "",
        ("- `as_of_date` is the information cutoff for every predictor."),
        (
            "- Technical features use market data no "
            "later than `technical_latest_market_date`, "
            "and that date must be <= `as_of_date`."
        ),
        (
            "- Fundamental factors use SEC accounting "
            "information that was available on or before "
            "`as_of_date`."
        ),
        (
            "- `first_future_date`, `target_end_date`, "
            "`target_21d`, `target_21d_excess` and "
            "`label_top_quintile` are future information "
            "and never belong to `MODEL_FEATURE_COLUMNS`."
        ),
        ("- Rows without a complete future horizon are preserved as `inference_only`."),
        "",
        "## Training-time transformation boundary",
        "",
        (
            "The master dataset intentionally does not "
            "perform global imputation, fitted scaling, "
            "PCA, model-based feature selection or "
            "hyperparameter fitting."
        ),
        "",
        (
            "Any operation that learns parameters from "
            "observations must be fitted only inside the "
            "training sample of each walk-forward fold."
        ),
        "",
        "## Predictor groups",
        "",
        "- Technical sector-neutral signals: `19`",
        "- Fundamental global z-scores: `24`",
        "- Fundamental sector z-scores: `24`",
        "- Fundamental missingness indicators: `24`",
        "- Total candidate predictors: `91`",
        "",
        "## Predictor coverage",
        "",
        _markdown_table(coverage_display),
        "",
        "## Column index",
        "",
        _markdown_table(compact_dictionary),
        "",
        "## Detailed column definitions",
        "",
    ]

    for row in dictionary.sort_values("position").itertuples(index=False):
        lines.extend(
            [
                f"### `{row.column}`",
                "",
                f"- **Role:** {row.role}",
                (f"- **Feature group:** {row.feature_group}"),
                f"- **Family:** {row.family}",
                (f"- **Source dataset:** `{row.source_dataset}`"),
                (f"- **Source column:** `{row.source_column}`"),
                (f"- **Model input:** `{bool(row.model_input)}`"),
                (f"- **Availability:** {row.availability_rule}"),
                (f"- **Calculation:** {row.calculation}"),
                (f"- **Missing values:** {row.missing_semantics}"),
                (f"- **Overall coverage:** {row.overall_coverage:.2%}"),
                (f"- **Latest-date coverage:** {row.latest_coverage:.2%}"),
                "",
            ]
        )

    return "\n".join(lines)


def write_modeling_panel_dictionary_artifacts(
    panel: pd.DataFrame,
    *,
    dictionary_csv_path: Path,
    coverage_csv_path: Path,
    documentation_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Write Step 11C dictionary artifacts."""
    dictionary = build_modeling_panel_data_dictionary(panel)

    coverage = build_modeling_feature_coverage(dictionary)

    dictionary_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coverage_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    documentation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dictionary.to_csv(
        dictionary_csv_path,
        index=False,
    )

    coverage.to_csv(
        coverage_csv_path,
        index=False,
    )

    documentation_path.write_text(
        render_data_dictionary(
            panel,
            dictionary,
        ),
        encoding="utf-8",
    )

    return (
        dictionary,
        coverage,
    )
