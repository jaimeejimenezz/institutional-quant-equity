"""Definitive simple model baselines"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


class ModelBaselineError(ValueError):
    """Raised when baseline prediction generation cannot run."""


MOMENTUM_FEATURE = (
    "tech__return_3m_sector_neutral"
)


COMPOSITE_FEATURE_DIRECTIONS: Mapping[
    str,
    float,
] = {
    (
        "tech__amihud_illiquidity_20d_"
        "sector_neutral"
    ): 1.0,
    (
        "tech__reversal_1m_"
        "sector_neutral"
    ): 1.0,
    (
        "tech__volatility_60d_"
        "sector_neutral"
    ): 1.0,
    (
        "tech__distance_sma_50d_"
        "sector_neutral"
    ): -1.0,
    (
        "tech__beta_60d_market_"
        "sector_neutral"
    ): 1.0,
    (
        "tech__return_3m_"
        "sector_neutral"
    ): 1.0,
    (
        "tech__max_drawdown_126d_"
        "sector_neutral"
    ): -1.0,
    (
        "tech__average_dollar_volume_20d_"
        "sector_neutral"
    ): -1.0,
}


BASELINE_MODEL_NAMES = (
    "constant",
    "momentum_3m",
    "technical_equal_weight_composite",
)


BASELINE_PREDICTION_COLUMNS = (
    "fold_id",
    "as_of_date",
    "ticker",
    "sector",
    "model_name",
    "prediction",
    "target_21d_excess",
    "label_top_quintile",
)


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    """Require columns to exist."""
    missing = sorted(
        set(columns).difference(
            data.columns
        )
    )

    if missing:
        raise ModelBaselineError(
            f"{dataset_name} is missing columns: "
            + ", ".join(missing)
            + "."
        )


def score_constant(
    test: pd.DataFrame,
) -> pd.Series:
    """Return the null-information baseline."""
    return pd.Series(
        0.0,
        index=test.index,
        dtype=float,
    )


def score_momentum(
    test: pd.DataFrame,
) -> pd.Series:
    """Use frozen 3-month momentum as prediction."""
    _require_columns(
        test,
        (MOMENTUM_FEATURE,),
        dataset_name="test cross-section",
    )

    return pd.to_numeric(
        test[MOMENTUM_FEATURE],
        errors="coerce",
    ).astype(float)


def score_technical_composite(
    test: pd.DataFrame,
    *,
    minimum_components: int = 6,
) -> pd.Series:
    """Build the frozen equal-weight technical composite."""
    if not 1 <= minimum_components <= len(
        COMPOSITE_FEATURE_DIRECTIONS
    ):
        raise ModelBaselineError(
            "minimum_components must be between "
            "1 and the number of composite features."
        )

    columns = tuple(
        COMPOSITE_FEATURE_DIRECTIONS
    )

    _require_columns(
        test,
        columns,
        dataset_name="test cross-section",
    )

    values = (
        test.loc[:, columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .astype(float)
    )

    directions = pd.Series(
        COMPOSITE_FEATURE_DIRECTIONS,
        dtype=float,
    )

    signed = values.mul(
        directions,
        axis="columns",
    )

    available_components = (
        signed.notna().sum(
            axis=1
        )
    )

    score = signed.mean(
        axis=1,
        skipna=True,
    )

    return score.where(
        available_components
        >= minimum_components
    )


def generate_baseline_predictions(
    panel: pd.DataFrame,
    fold_metadata: pd.DataFrame,
    *,
    minimum_composite_components: int = 6,
) -> pd.DataFrame:
    """Generate Step 13A predictions on frozen OOS dates."""
    required_panel_columns = (
        "as_of_date",
        "ticker",
        "sector",
        "target_21d_excess",
        "label_top_quintile",
        MOMENTUM_FEATURE,
        *tuple(
            COMPOSITE_FEATURE_DIRECTIONS
        ),
    )

    _require_columns(
        panel,
        required_panel_columns,
        dataset_name="modeling panel",
    )

    _require_columns(
        fold_metadata,
        (
            "fold_id",
            "test_date",
            "test_rows",
        ),
        dataset_name="walk-forward fold metadata",
    )

    panel = panel.copy()
    folds = fold_metadata.copy()

    panel["as_of_date"] = pd.to_datetime(
        panel["as_of_date"]
    )

    folds["test_date"] = pd.to_datetime(
        folds["test_date"]
    )

    duplicate_test_dates = int(
        folds["test_date"]
        .duplicated()
        .sum()
    )

    if duplicate_test_dates:
        raise ModelBaselineError(
            "Walk-forward metadata contains "
            f"{duplicate_test_dates} duplicated test dates."
        )

    output: list[pd.DataFrame] = []

    for fold in (
        folds.sort_values(
            "test_date"
        )
        .itertuples(
            index=False
        )
    ):
        test_date = pd.Timestamp(
            fold.test_date
        )

        test = (
            panel.loc[
                panel["as_of_date"].eq(
                    test_date
                )
            ]
            .sort_values(
                "ticker"
            )
            .copy()
        )

        expected_rows = int(
            fold.test_rows
        )

        if len(test) != expected_rows:
            raise ModelBaselineError(
                f"{fold.fold_id} expected "
                f"{expected_rows} test rows but "
                f"found {len(test)}."
            )

        scores = {
            "constant": score_constant(
                test
            ),
            "momentum_3m": score_momentum(
                test
            ),
            (
                "technical_equal_weight_composite"
            ): score_technical_composite(
                test,
                minimum_components=(
                    minimum_composite_components
                ),
            ),
        }

        for (
            model_name,
            prediction,
        ) in scores.items():
            block = test.loc[
                :,
                [
                    "as_of_date",
                    "ticker",
                    "sector",
                    "target_21d_excess",
                    "label_top_quintile",
                ],
            ].copy()

            block.insert(
                0,
                "fold_id",
                str(
                    fold.fold_id
                ),
            )

            block[
                "model_name"
            ] = model_name

            block[
                "prediction"
            ] = prediction.astype(
                float
            )

            output.append(
                block.loc[
                    :,
                    BASELINE_PREDICTION_COLUMNS,
                ]
            )

    if not output:
        raise ModelBaselineError(
            "No baseline predictions were generated."
        )

    result = pd.concat(
        output,
        ignore_index=True,
    )

    duplicate_keys = int(
        result.duplicated(
            subset=[
                "fold_id",
                "ticker",
                "model_name",
            ]
        ).sum()
    )

    if duplicate_keys:
        raise ModelBaselineError(
            "Baseline predictions contain "
            f"{duplicate_keys} duplicated keys."
        )

    return result.sort_values(
        [
            "as_of_date",
            "model_name",
            "ticker",
        ]
    ).reset_index(
        drop=True
    )