"""Build the technical modeling panel and walk-forward folds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_equity.config import (
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    load_config,
)
from quant_equity.features import (
    SELECTED_TECHNICAL_FEATURE_COLUMNS,
)
from quant_equity.logging_config import (
    configure_logging,
)
from quant_equity.validation import (
    LinearModelingConfig,
    build_expanding_walk_forward_folds,
    build_linear_modeling_panel,
)

TECHNICAL_FEATURES_PATH = PROCESSED_DATA_DIR / "features_technical_monthly.parquet"

MONTHLY_LABELS_PATH = PROCESSED_DATA_DIR / "labels_monthly.parquet"

MODELING_PANEL_PATH = PROCESSED_DATA_DIR / "technical_modeling_panel.parquet"

FOLDS_PATH = PROCESSED_DATA_DIR / "linear_walk_forward_folds.parquet"

REPORT_PATH = REPORTS_DIR / "data_quality" / "linear_modeling_panel_report.md"


def _write_parquet_atomically(
    data: pd.DataFrame,
    path: Path,
    *,
    sort_by: list[str],
) -> Path:
    """Write an ordered Parquet file atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered = data.sort_values(sort_by).reset_index(drop=True)

    temporary_path = path.with_suffix(".tmp.parquet")

    temporary_path.unlink(missing_ok=True)

    ordered.to_parquet(
        temporary_path,
        index=False,
    )

    temporary_path.replace(path)

    return path


def _format_value(
    value: Any,
) -> str:
    """Format a value for Markdown."""
    if value is None or pd.isna(value):
        return ""

    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()

    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value).replace(
        "|",
        "\\|",
    )


def _to_markdown(
    frame: pd.DataFrame,
) -> str:
    """Convert a dataframe to a Markdown table."""
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
        values = [_format_value(value) for value in row]

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _write_report(
    panel: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    config: LinearModelingConfig,
) -> Path:
    """Write the Step 7A data-quality report."""
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_missingness = pd.DataFrame(
        {
            "feature": (SELECTED_TECHNICAL_FEATURE_COLUMNS),
            "missing_rows": [
                int(panel[feature].isna().sum()) for feature in (SELECTED_TECHNICAL_FEATURE_COLUMNS)
            ],
            "missing_ratio": [
                float(panel[feature].isna().mean())
                for feature in (SELECTED_TECHNICAL_FEATURE_COLUMNS)
            ],
        }
    )

    fold_display = folds.loc[
        :,
        [
            "fold_id",
            "test_date",
            "train_start_date",
            "train_end_date",
            "validation_start_date",
            "validation_end_date",
            "latest_known_target_end_date",
            "training_dates",
            "validation_dates",
            "test_rows",
        ],
    ]

    lines = [
        "# Linear Modeling Panel Report",
        "",
        "## Status",
        "",
        "**PASS**",
        "",
        "## Purpose",
        "",
        (
            "This dataset joins the frozen technical-feature "
            "selection with the monthly forward-return labels."
        ),
        "",
        (
            "No feature selection is performed in this step. "
            "The eight technical variables were frozen before "
            "the out-of-sample period was inspected."
        ),
        "",
        "## Dataset summary",
        "",
        f"- Rows: `{len(panel)}`",
        (f"- Dates: `{panel['as_of_date'].nunique()}`"),
        (f"- Tickers: `{panel['ticker'].nunique()}`"),
        (f"- Selected features: `{len(SELECTED_TECHNICAL_FEATURE_COLUMNS)}`"),
        (f"- Earliest date: `{panel['as_of_date'].min().date()}`"),
        (f"- Latest date: `{panel['as_of_date'].max().date()}`"),
        (
            "- Research dates: "
            f"`{panel.loc[panel['sample_period'].eq('research'), 'as_of_date'].nunique()}`"
        ),
        (
            "- Out-of-sample dates: "
            f"`{panel.loc[panel['sample_period'].eq('out_of_sample'), 'as_of_date'].nunique()}`"
        ),
        (f"- Complete feature rows: `{int(panel['is_complete_feature_row'].sum())}`"),
        (
            "- Rows with at least one missing feature: "
            f"`{int(panel['feature_missing_count'].gt(0).sum())}`"
        ),
        "",
        "## Temporal validation",
        "",
        (f"- Out-of-sample start: `{config.out_of_sample_start_date.date()}`"),
        (f"- Minimum training months: `{config.minimum_training_months}`"),
        (f"- Validation months: `{config.validation_months}`"),
        f"- Walk-forward folds: `{len(folds)}`",
        "",
        (
            "A historical label can only enter training or "
            "validation when its target end date is no later "
            "than the current test date."
        ),
        "",
        "## Selected-feature missingness",
        "",
        _to_markdown(feature_missingness),
        "",
        "## Walk-forward folds",
        "",
        _to_markdown(fold_display),
        "",
        "## Leakage controls",
        "",
        (
            "- Every company from the same month remains in "
            "the same training, validation or test partition."
        ),
        ("- Test dates never appear in training or validation."),
        (
            "- Training and validation use only labels that "
            "were already fully realized by the test date."
        ),
        (
            "- Missing features remain missing at this stage. "
            "Any imputation must be fitted inside each fold."
        ),
        ("- No scaler or predictive model is fitted while building this dataset."),
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return REPORT_PATH


def main() -> None:
    """Execute Step 7A."""
    config = load_config()

    logger = configure_logging(
        level=config["runtime"]["log_level"],
        log_file=(PROJECT_ROOT / "logs" / "build_linear_modeling_panel.log"),
    )

    for required_path in (
        TECHNICAL_FEATURES_PATH,
        MONTHLY_LABELS_PATH,
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Required dataset not found: {required_path}")

    modeling_config = LinearModelingConfig.from_mapping(config["linear_modeling"])

    technical_features = pd.read_parquet(TECHNICAL_FEATURES_PATH)

    monthly_labels = pd.read_parquet(MONTHLY_LABELS_PATH)

    panel = build_linear_modeling_panel(
        technical_features,
        monthly_labels,
        feature_columns=(SELECTED_TECHNICAL_FEATURE_COLUMNS),
        config=modeling_config,
    )

    folds = build_expanding_walk_forward_folds(
        panel,
        config=modeling_config,
    )

    panel_path = _write_parquet_atomically(
        panel,
        MODELING_PANEL_PATH,
        sort_by=[
            "as_of_date",
            "ticker",
        ],
    )

    folds_path = _write_parquet_atomically(
        folds,
        FOLDS_PATH,
        sort_by=["test_date"],
    )

    report_path = _write_report(
        panel,
        folds,
        config=modeling_config,
    )

    logger.info("Linear modeling panel completed.")

    logger.info(
        "Panel rows: %s",
        len(panel),
    )

    logger.info(
        "Walk-forward folds: %s",
        len(folds),
    )

    first_fold = folds.iloc[0]
    last_fold = folds.iloc[-1]

    print()
    print("Institutional Quant Equity Research Platform")
    print("Linear modeling panel - Step 7A")
    print("-" * 48)
    print(f"Rows: {len(panel)}")
    print(f"Dates: {panel['as_of_date'].nunique()}")
    print(f"Tickers: {panel['ticker'].nunique()}")
    print(f"Selected features: {len(SELECTED_TECHNICAL_FEATURE_COLUMNS)}")
    print(
        "Research dates: "
        f"{panel.loc[panel['sample_period'].eq('research'), 'as_of_date'].nunique()}"
    )
    print(
        "Out-of-sample dates: "
        f"{panel.loc[panel['sample_period'].eq('out_of_sample'), 'as_of_date'].nunique()}"
    )
    print(f"Complete feature rows: {int(panel['is_complete_feature_row'].sum())}")
    print(f"Walk-forward folds: {len(folds)}")
    print()
    print(f"First test date: {first_fold['test_date'].date()}")
    print(f"First fold training dates: {first_fold['training_dates']}")
    print(f"First fold validation dates: {first_fold['validation_dates']}")
    print(f"Last test date: {last_fold['test_date'].date()}")
    print()
    print(f"Modeling panel: {panel_path}")
    print(f"Walk-forward folds: {folds_path}")
    print(f"Quality report: {report_path}")
    print()
    print("Linear modeling panel: OK")


if __name__ == "__main__":
    main()
