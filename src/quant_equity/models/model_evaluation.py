"""Common OOS model evaluation"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


class ModelEvaluationError(ValueError):
    """Raised when OOS model evaluation cannot run."""


MONTHLY_METRIC_COLUMNS = (
    "model_name",
    "as_of_date",
    "observations",
    "ic",
    "q1_mean_excess_return",
    "q2_mean_excess_return",
    "q3_mean_excess_return",
    "q4_mean_excess_return",
    "q5_mean_excess_return",
    "top_bottom_spread",
    "top_quintile_precision",
    "top_quintile_turnover",
)


def _spearman_ic(
    prediction: pd.Series,
    target: pd.Series,
) -> float:
    """Calculate cross-sectional Spearman IC."""
    valid = (
        prediction.notna()
        & target.notna()
    )

    x = prediction.loc[
        valid
    ]

    y = target.loc[
        valid
    ]

    if (
        len(x) < 2
        or x.nunique() < 2
        or y.nunique() < 2
    ):
        return float("nan")

    return float(
        x.corr(
            y,
            method="spearman",
        )
    )


def evaluate_model_predictions(
    predictions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Evaluate all models with one common OOS protocol."""
    required = {
        "as_of_date",
        "ticker",
        "model_name",
        "prediction",
        "target_21d_excess",
        "label_top_quintile",
    }

    missing = sorted(
        required.difference(
            predictions.columns
        )
    )

    if missing:
        raise ModelEvaluationError(
            "Predictions are missing columns: "
            + ", ".join(missing)
            + "."
        )

    data = predictions.copy()

    data["as_of_date"] = pd.to_datetime(
        data["as_of_date"]
    )

    monthly_rows: list[
        dict[str, object]
    ] = []

    for (
        model_name,
        model_data,
    ) in data.groupby(
        "model_name",
        sort=True,
    ):
        previous_top: set[str] | None = None

        for (
            as_of_date,
            month,
        ) in model_data.groupby(
            "as_of_date",
            sort=True,
        ):
            valid = month.dropna(
                subset=[
                    "prediction",
                    "target_21d_excess",
                ]
            ).copy()

            observations = len(
                valid
            )

            ic = _spearman_ic(
                valid["prediction"],
                valid["target_21d_excess"],
            )

            row: dict[
                str,
                object,
            ] = {
                "model_name": model_name,
                "as_of_date": pd.Timestamp(
                    as_of_date
                ),
                "observations": observations,
                "ic": ic,
                "q1_mean_excess_return": (
                    float("nan")
                ),
                "q2_mean_excess_return": (
                    float("nan")
                ),
                "q3_mean_excess_return": (
                    float("nan")
                ),
                "q4_mean_excess_return": (
                    float("nan")
                ),
                "q5_mean_excess_return": (
                    float("nan")
                ),
                "top_bottom_spread": (
                    float("nan")
                ),
                "top_quintile_precision": (
                    float("nan")
                ),
                "top_quintile_turnover": (
                    float("nan")
                ),
            }

            # A constant prediction cannot create a
            # meaningful ranking. Do not invent one.
            if (
                observations < 5
                or valid[
                    "prediction"
                ].nunique()
                < 2
            ):
                previous_top = None
                monthly_rows.append(
                    row
                )
                continue

            ranked = valid.sort_values(
                [
                    "prediction",
                    "ticker",
                ],
                ascending=[
                    False,
                    True,
                ],
            ).copy()

            number_of_rows = len(
                ranked
            )

            positions = np.arange(
                number_of_rows
            )

            ranked[
                "predicted_quintile"
            ] = (
                5
                - (
                    positions
                    * 5
                    // number_of_rows
                )
            )

            quintile_returns: dict[
                int,
                float,
            ] = {}

            for quintile in range(
                1,
                6,
            ):
                values = ranked.loc[
                    ranked[
                        "predicted_quintile"
                    ].eq(
                        quintile
                    ),
                    "target_21d_excess",
                ]

                quintile_returns[
                    quintile
                ] = (
                    float(
                        values.mean()
                    )
                    if not values.empty
                    else float("nan")
                )

                row[
                    (
                        f"q{quintile}_"
                        "mean_excess_return"
                    )
                ] = quintile_returns[
                    quintile
                ]

            row[
                "top_bottom_spread"
            ] = (
                quintile_returns[5]
                - quintile_returns[1]
            )

            top = ranked.loc[
                ranked[
                    "predicted_quintile"
                ].eq(
                    5
                )
            ]

            actual_labels = pd.to_numeric(
                top[
                    "label_top_quintile"
                ],
                errors="coerce",
            )

            row[
                "top_quintile_precision"
            ] = (
                float(
                    actual_labels.mean()
                )
                if actual_labels.notna().any()
                else float("nan")
            )

            current_top = set(
                top["ticker"].astype(
                    str
                )
            )

            if (
                previous_top is not None
                and current_top
            ):
                overlap = len(
                    current_top.intersection(
                        previous_top
                    )
                )

                row[
                    "top_quintile_turnover"
                ] = (
                    1.0
                    - overlap
                    / len(
                        current_top
                    )
                )

            previous_top = (
                current_top
            )

            monthly_rows.append(
                row
            )

    monthly = pd.DataFrame(
        monthly_rows,
        columns=MONTHLY_METRIC_COLUMNS,
    ).sort_values(
        [
            "model_name",
            "as_of_date",
        ]
    ).reset_index(
        drop=True
    )

    summary_rows: list[
        dict[str, object]
    ] = []

    for (
        model_name,
        model_monthly,
    ) in monthly.groupby(
        "model_name",
        sort=True,
    ):
        valid_ic = (
            model_monthly["ic"]
            .dropna()
            .astype(float)
        )

        mean_ic = (
            float(
                valid_ic.mean()
            )
            if not valid_ic.empty
            else float("nan")
        )

        std_ic = (
            float(
                valid_ic.std(
                    ddof=1
                )
            )
            if len(valid_ic) > 1
            else float("nan")
        )

        ic_ir = (
            mean_ic
            / std_ic
            * math.sqrt(
                12.0
            )
            if (
                np.isfinite(
                    mean_ic
                )
                and np.isfinite(
                    std_ic
                )
                and std_ic > 0
            )
            else float("nan")
        )

        positive_ic_ratio = (
            float(
                valid_ic.gt(
                    0.0
                ).mean()
            )
            if not valid_ic.empty
            else float("nan")
        )

        summary_rows.append(
            {
                "model_name": model_name,
                "months": len(
                    model_monthly
                ),
                "valid_ic_months": len(
                    valid_ic
                ),
                "mean_ic": mean_ic,
                "median_ic": (
                    float(
                        valid_ic.median()
                    )
                    if not valid_ic.empty
                    else float("nan")
                ),
                "std_ic": std_ic,
                "annualized_ic_ir": ic_ir,
                "positive_ic_ratio": (
                    positive_ic_ratio
                ),
                (
                    "mean_top_bottom_spread"
                ): float(
                    model_monthly[
                        "top_bottom_spread"
                    ].mean()
                ),
                (
                    "mean_top_quintile_precision"
                ): float(
                    model_monthly[
                        "top_quintile_precision"
                    ].mean()
                ),
                (
                    "mean_top_quintile_turnover"
                ): float(
                    model_monthly[
                        "top_quintile_turnover"
                    ].mean()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        "model_name"
    ).reset_index(
        drop=True
    )

    return (
        monthly,
        summary,
    )