"""Evaluate frozen alpha-signal robustness across return horizons."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_equity.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from quant_equity.labels import build_forward_return_labels
from quant_equity.logging_config import configure_logging

FINAL_SIGNAL_PATH = PROCESSED_DATA_DIR / "final_alpha_signal.parquet"

MARKET_DATA_PATH = PROCESSED_DATA_DIR / "market_daily.parquet"

TABLES_DIR = REPORTS_DIR / "tables"

MONTHLY_PATH = TABLES_DIR / "robustness_prediction_horizon_monthly.csv"

SUMMARY_PATH = TABLES_DIR / "robustness_prediction_horizon_summary.csv"

YEARLY_PATH = TABLES_DIR / "robustness_prediction_horizon_yearly.csv"

CHECKS_PATH = TABLES_DIR / "robustness_prediction_horizon_checks.csv"

REPORT_PATH = REPORTS_DIR / "robustness" / "prediction_horizon_robustness.md"

HORIZONS = (
    10,
    21,
    42,
)

BASELINE_HORIZON = 21

BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_BLOCK_LENGTH = 3
BOOTSTRAP_SEED = 42
CONFIDENCE_LEVEL = 0.95

MINIMUM_COMMON_DATES = 60
EXPECTED_CROSS_SECTION_SIZE = 50


def _write_csv(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write one CSV table."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        path,
        index=False,
    )


def _prepare_signal(
    final_signal: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the frozen production alpha signal."""
    required_columns = {
        "as_of_date",
        "ticker",
        "rank",
        "percentile_score",
    }

    missing = sorted(required_columns.difference(final_signal.columns))

    if missing:
        raise ValueError("Final alpha signal is missing columns: " + ", ".join(missing) + ".")

    data = final_signal.loc[
        :,
        [
            "as_of_date",
            "ticker",
            "rank",
            "percentile_score",
        ],
    ].copy()

    data["as_of_date"] = pd.to_datetime(
        data["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    data["ticker"] = data["ticker"].astype("string").str.strip()

    for column in (
        "rank",
        "percentile_score",
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if (
        data["as_of_date"].isna().any()
        or data["ticker"].isna().any()
        or data[
            [
                "rank",
                "percentile_score",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Final alpha signal contains invalid values.")

    if data.duplicated(
        [
            "as_of_date",
            "ticker",
        ]
    ).any():
        raise ValueError("Final alpha signal contains duplicate date-ticker rows.")

    cross_sections = data.groupby("as_of_date")["ticker"].nunique()

    if cross_sections.ne(EXPECTED_CROSS_SECTION_SIZE).any():
        raise ValueError("Final alpha signal does not contain 50 securities on every date.")

    return data.sort_values(
        [
            "as_of_date",
            "rank",
            "ticker",
        ]
    ).reset_index(drop=True)


def _build_horizon_labels(
    market_data: pd.DataFrame,
    signal: pd.DataFrame,
    *,
    horizon_sessions: int,
) -> pd.DataFrame:
    """Build realized labels for one horizon and align them to the signal."""
    labels = build_forward_return_labels(
        market_data,
        horizon_sessions=horizon_sessions,
        relative_to="cross_sectional_median",
        top_quantile_fraction=0.20,
    )

    required_columns = {
        "as_of_date",
        "ticker",
        "target_21d",
        "target_21d_excess",
        "target_rank",
    }

    missing = sorted(required_columns.difference(labels.columns))

    if missing:
        raise ValueError("Forward-return labels are missing columns: " + ", ".join(missing) + ".")

    labels = labels.loc[
        :,
        [
            "as_of_date",
            "ticker",
            "target_21d",
            "target_21d_excess",
            "target_rank",
        ],
    ].copy()

    labels["as_of_date"] = pd.to_datetime(labels["as_of_date"]).dt.normalize()

    labels["ticker"] = labels["ticker"].astype("string").str.strip()

    labels = labels.rename(
        columns={
            "target_21d": "realized_return",
            "target_21d_excess": "realized_excess_return",
            "target_rank": "realized_rank",
        }
    )

    aligned = signal.merge(
        labels,
        on=[
            "as_of_date",
            "ticker",
        ],
        how="inner",
        validate="one_to_one",
    )

    aligned["horizon_sessions"] = int(horizon_sessions)

    return aligned


def _common_dates(
    horizon_panels: dict[
        int,
        pd.DataFrame,
    ],
) -> pd.DatetimeIndex:
    """Return dates with complete data for every requested horizon."""
    common = None

    for panel in horizon_panels.values():
        counts = panel.groupby("as_of_date")["ticker"].nunique()

        complete_dates = set(counts.loc[counts.eq(EXPECTED_CROSS_SECTION_SIZE)].index)

        if common is None:
            common = complete_dates
        else:
            common = common.intersection(complete_dates)

    if common is None:
        return pd.DatetimeIndex([])

    return pd.DatetimeIndex(sorted(common))


def _spearman(
    left: pd.Series,
    right: pd.Series,
) -> float:
    """Calculate Spearman correlation through rank correlation."""
    left_rank = pd.to_numeric(
        left,
        errors="coerce",
    ).rank(method="average")

    right_rank = pd.to_numeric(
        right,
        errors="coerce",
    ).rank(method="average")

    if left_rank.nunique() < 2 or right_rank.nunique() < 2:
        return float("nan")

    return float(left_rank.corr(right_rank))


def _monthly_metrics(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate monthly ranking metrics for one realized horizon."""
    rows = []

    horizon_sessions = int(panel["horizon_sessions"].iloc[0])

    for as_of_date, group in panel.groupby(
        "as_of_date",
        sort=True,
    ):
        group = group.sort_values(
            [
                "rank",
                "ticker",
            ]
        ).reset_index(drop=True)

        count = len(group)

        quintile_size = max(
            1,
            int(np.ceil(count * 0.20)),
        )

        predicted_top = group.nsmallest(
            quintile_size,
            "rank",
        )

        predicted_bottom = group.nlargest(
            quintile_size,
            "rank",
        )

        actual_top = set(
            group.nsmallest(
                quintile_size,
                "realized_rank",
            )["ticker"].astype(str)
        )

        predicted_top_names = set(predicted_top["ticker"].astype(str))

        rows.append(
            {
                "horizon_sessions": horizon_sessions,
                "as_of_date": pd.Timestamp(as_of_date),
                "cross_section_size": int(count),
                "information_coefficient": _spearman(
                    group["percentile_score"],
                    group["realized_excess_return"],
                ),
                "top_bottom_spread": float(
                    predicted_top["realized_excess_return"].mean()
                    - predicted_bottom["realized_excess_return"].mean()
                ),
                "top_quintile_precision": float(
                    len(predicted_top_names.intersection(actual_top)) / quintile_size
                ),
                "top_quintile_mean_excess_return": float(
                    predicted_top["realized_excess_return"].mean()
                ),
                "bottom_quintile_mean_excess_return": float(
                    predicted_bottom["realized_excess_return"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def _annualized_ic_ir(
    values: np.ndarray,
) -> float:
    """Calculate annualized IC information ratio."""
    standard_deviation = float(
        np.std(
            values,
            ddof=1,
        )
    )

    if standard_deviation <= 0.0:
        return float("nan")

    return float(np.mean(values) / standard_deviation * np.sqrt(12.0))


def _bootstrap_indices(
    *,
    observations: int,
) -> np.ndarray:
    """Create circular block-bootstrap indices shared by all horizons."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    blocks_needed = int(np.ceil(observations / BOOTSTRAP_BLOCK_LENGTH))

    starts = rng.integers(
        0,
        observations,
        size=(
            BOOTSTRAP_REPLICATIONS,
            blocks_needed,
        ),
    )

    offsets = np.arange(
        BOOTSTRAP_BLOCK_LENGTH,
        dtype=int,
    )

    indices = (
        starts[
            :,
            :,
            None,
        ]
        + offsets[
            None,
            None,
            :,
        ]
    ) % observations

    return indices.reshape(
        BOOTSTRAP_REPLICATIONS,
        -1,
    )[
        :,
        :observations,
    ]


def _confidence_interval(
    values: np.ndarray,
) -> tuple[
    float,
    float,
]:
    """Return an equal-tailed confidence interval."""
    alpha = 1.0 - CONFIDENCE_LEVEL

    return (
        float(
            np.quantile(
                values,
                alpha / 2.0,
            )
        ),
        float(
            np.quantile(
                values,
                1.0 - alpha / 2.0,
            )
        ),
    )


def _build_summary(
    monthly: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize IC and spread robustness for each horizon."""
    dates = sorted(monthly["as_of_date"].unique())

    indices = _bootstrap_indices(observations=len(dates))

    rows = []

    for horizon_sessions, group in monthly.groupby(
        "horizon_sessions",
        sort=True,
    ):
        group = group.set_index("as_of_date").reindex(dates)

        ic = group["information_coefficient"].to_numpy(dtype=float)

        spread = group["top_bottom_spread"].to_numpy(dtype=float)

        precision = group["top_quintile_precision"].to_numpy(dtype=float)

        if not (
            np.isfinite(ic).all() and np.isfinite(spread).all() and np.isfinite(precision).all()
        ):
            raise ValueError("Horizon summary contains non-finite monthly metrics.")

        mean_ic_draws = np.mean(
            ic[indices],
            axis=1,
        )

        mean_spread_draws = np.mean(
            spread[indices],
            axis=1,
        )

        mean_precision_draws = np.mean(
            precision[indices],
            axis=1,
        )

        ic_ci = _confidence_interval(mean_ic_draws)

        spread_ci = _confidence_interval(mean_spread_draws)

        precision_ci = _confidence_interval(mean_precision_draws)

        rows.append(
            {
                "horizon_sessions": int(horizon_sessions),
                "months": int(len(group)),
                "mean_ic": float(np.mean(ic)),
                "median_ic": float(np.median(ic)),
                "annualized_ic_ir": _annualized_ic_ir(ic),
                "positive_ic_ratio": float(np.mean(ic > 0.0)),
                "mean_ic_ci_lower": ic_ci[0],
                "mean_ic_ci_upper": ic_ci[1],
                "probability_mean_ic_positive": float(np.mean(mean_ic_draws > 0.0)),
                "mean_top_bottom_spread": float(np.mean(spread)),
                "median_top_bottom_spread": float(np.median(spread)),
                "positive_spread_ratio": float(np.mean(spread > 0.0)),
                "spread_ci_lower": spread_ci[0],
                "spread_ci_upper": spread_ci[1],
                "probability_mean_spread_positive": float(np.mean(mean_spread_draws > 0.0)),
                "mean_top_quintile_precision": float(np.mean(precision)),
                "precision_ci_lower": precision_ci[0],
                "precision_ci_upper": precision_ci[1],
            }
        )

    summary = pd.DataFrame(rows)

    baseline = summary.loc[summary["horizon_sessions"].eq(BASELINE_HORIZON)]

    if len(baseline) != 1:
        raise ValueError("Baseline 21-session horizon is missing.")

    baseline_ic = float(baseline["mean_ic"].iloc[0])

    baseline_spread = float(baseline["mean_top_bottom_spread"].iloc[0])

    baseline_precision = float(baseline["mean_top_quintile_precision"].iloc[0])

    summary["mean_ic_difference_vs_21d"] = summary["mean_ic"] - baseline_ic

    summary["spread_difference_vs_21d"] = summary["mean_top_bottom_spread"] - baseline_spread

    summary["precision_difference_vs_21d"] = (
        summary["mean_top_quintile_precision"] - baseline_precision
    )

    return summary.sort_values("horizon_sessions").reset_index(drop=True)


def _build_yearly(
    monthly: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize horizon metrics by calendar year."""
    data = monthly.copy()

    data["year"] = data["as_of_date"].dt.year

    return (
        data.groupby(
            [
                "horizon_sessions",
                "year",
            ],
            as_index=False,
        )
        .agg(
            months=(
                "as_of_date",
                "size",
            ),
            mean_ic=(
                "information_coefficient",
                "mean",
            ),
            positive_ic_ratio=(
                "information_coefficient",
                lambda values: float(np.mean(values.to_numpy(dtype=float) > 0.0)),
            ),
            mean_top_bottom_spread=(
                "top_bottom_spread",
                "mean",
            ),
            positive_spread_ratio=(
                "top_bottom_spread",
                lambda values: float(np.mean(values.to_numpy(dtype=float) > 0.0)),
            ),
            mean_top_quintile_precision=(
                "top_quintile_precision",
                "mean",
            ),
        )
        .sort_values(
            [
                "year",
                "horizon_sessions",
            ]
        )
        .reset_index(drop=True)
    )


def _build_checks(
    common_dates: pd.DatetimeIndex,
    monthly: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Audit prediction-horizon robustness outputs."""
    horizons = set(summary["horizon_sessions"].astype(int))

    months_per_horizon = monthly.groupby("horizon_sessions")["as_of_date"].nunique()

    cross_sections = monthly["cross_section_size"]

    key_values = summary[
        [
            "mean_ic",
            "annualized_ic_ir",
            "positive_ic_ratio",
            "mean_top_bottom_spread",
            "positive_spread_ratio",
            "mean_top_quintile_precision",
        ]
    ].to_numpy(dtype=float)

    probability_columns = (
        "probability_mean_ic_positive",
        "probability_mean_spread_positive",
    )

    probability_violations = int(
        sum(
            (
                ~summary[column].between(
                    0.0,
                    1.0,
                )
            ).sum()
            for column in probability_columns
        )
    )

    checks = [
        (
            "expected_horizons",
            int(horizons != set(HORIZONS)),
            "The analysis must contain 10-, 21- and 42-session horizons.",
        ),
        (
            "minimum_common_dates",
            int(len(common_dates) < MINIMUM_COMMON_DATES),
            "Horizon comparison must use at least 60 common OOS dates.",
        ),
        (
            "same_dates_per_horizon",
            int(
                months_per_horizon.nunique() != 1 or months_per_horizon.iloc[0] != len(common_dates)
            ),
            "Every horizon must be evaluated on identical OOS dates.",
        ),
        (
            "complete_cross_sections",
            int(cross_sections.ne(EXPECTED_CROSS_SECTION_SIZE).sum()),
            "Every horizon-date observation must use the full 50-stock cross-section.",
        ),
        (
            "finite_summary_metrics",
            int((~np.isfinite(key_values)).sum()),
            "Key horizon robustness metrics must remain finite.",
        ),
        (
            "valid_bootstrap_probabilities",
            probability_violations,
            "Bootstrap probabilities must lie between zero and one.",
        ),
        (
            "confidence_interval_order",
            int(
                summary["mean_ic_ci_lower"].gt(summary["mean_ic_ci_upper"]).sum()
                + summary["spread_ci_lower"].gt(summary["spread_ci_upper"]).sum()
            ),
            "Confidence-interval lower bounds must not exceed upper bounds.",
        ),
        (
            "baseline_horizon_present",
            int(summary["horizon_sessions"].eq(BASELINE_HORIZON).sum() != 1),
            "The frozen 21-session baseline must be present exactly once.",
        ),
    ]

    return pd.DataFrame(
        [
            {
                "check": name,
                "status": ("PASS" if violations == 0 else "FAIL"),
                "violations": int(violations),
                "description": description,
            }
            for (
                name,
                violations,
                description,
            ) in checks
        ]
    )


def _format_value(
    value: Any,
) -> str:
    """Format one value for Markdown."""
    if value is None or pd.isna(value):
        return ""

    if isinstance(
        value,
        float,
    ):
        return f"{value:.6f}"

    return str(value).replace(
        "|",
        "\\|",
    )


def _to_markdown(
    data: pd.DataFrame,
) -> str:
    """Convert a dataframe to Markdown."""
    if data.empty:
        return "_No observations._"

    columns = [str(column) for column in data.columns]

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in data.itertuples(
        index=False,
        name=None,
    ):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")

    return "\n".join(lines)


def _build_report(
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    checks: pd.DataFrame,
    *,
    common_dates: int,
) -> str:
    """Build the frozen-signal horizon robustness report."""
    display = summary.loc[
        :,
        [
            "horizon_sessions",
            "months",
            "mean_ic",
            "mean_ic_ci_lower",
            "mean_ic_ci_upper",
            "probability_mean_ic_positive",
            "annualized_ic_ir",
            "positive_ic_ratio",
            "mean_top_bottom_spread",
            "spread_ci_lower",
            "spread_ci_upper",
            "probability_mean_spread_positive",
            "positive_spread_ratio",
            "mean_top_quintile_precision",
            "mean_ic_difference_vs_21d",
            "spread_difference_vs_21d",
        ],
    ]

    return "\n".join(
        [
            "# Prediction Horizon Robustness",
            "",
            "## Methodology",
            "",
            (
                "- The production alpha signal remains completely frozen. "
                "No model is retrained or re-tuned for this experiment."
            ),
            (
                "- Realized cross-sectional excess returns are reconstructed "
                "at 10, 21 and 42 market-session horizons."
            ),
            (
                f"- All horizon comparisons use the same `{common_dates}` "
                "out-of-sample dates and the same 50 securities per date."
            ),
            (
                "- This tests temporal persistence of the existing signal. "
                "It does not claim that a separately trained 10- or 42-session "
                "model would produce identical results."
            ),
            (
                f"- Confidence intervals use `{BOOTSTRAP_REPLICATIONS:,}` "
                f"circular block-bootstrap replications with "
                f"`{BOOTSTRAP_BLOCK_LENGTH}`-month blocks."
            ),
            "",
            "## Horizon summary",
            "",
            _to_markdown(display),
            "",
            "## Calendar-year stability",
            "",
            _to_markdown(yearly),
            "",
            "## Readiness checks",
            "",
            _to_markdown(checks),
            "",
        ]
    )


def main() -> None:
    """Run frozen-signal prediction-horizon robustness analysis."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    for path in (
        FINAL_SIGNAL_PATH,
        MARKET_DATA_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    signal = _prepare_signal(pd.read_parquet(FINAL_SIGNAL_PATH))

    market_data = pd.read_parquet(MARKET_DATA_PATH)

    horizon_panels = {}

    for horizon_sessions in HORIZONS:
        logger.info(
            "Building realized %s-session horizon labels.",
            horizon_sessions,
        )

        horizon_panels[horizon_sessions] = _build_horizon_labels(
            market_data,
            signal,
            horizon_sessions=horizon_sessions,
        )

    common_dates = _common_dates(horizon_panels)

    if len(common_dates) < MINIMUM_COMMON_DATES:
        raise ValueError("Too few common dates across requested prediction horizons.")

    monthly_blocks = []

    for horizon_sessions in HORIZONS:
        panel = horizon_panels[horizon_sessions]

        panel = panel.loc[panel["as_of_date"].isin(common_dates)].copy()

        logger.info(
            "Evaluating frozen signal at %s sessions.",
            horizon_sessions,
        )

        monthly_blocks.append(_monthly_metrics(panel))

    monthly = (
        pd.concat(
            monthly_blocks,
            ignore_index=True,
        )
        .sort_values(
            [
                "as_of_date",
                "horizon_sessions",
            ]
        )
        .reset_index(drop=True)
    )

    summary = _build_summary(monthly)

    yearly = _build_yearly(monthly)

    checks = _build_checks(
        common_dates,
        monthly,
        summary,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    _write_csv(
        monthly,
        MONTHLY_PATH,
    )

    _write_csv(
        summary,
        SUMMARY_PATH,
    )

    _write_csv(
        yearly,
        YEARLY_PATH,
    )

    _write_csv(
        checks,
        CHECKS_PATH,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        _build_report(
            summary,
            yearly,
            checks,
            common_dates=len(common_dates),
        ),
        encoding="utf-8",
    )

    if failed_checks:
        raise ValueError(
            f"Prediction-horizon robustness validation failed with {failed_checks} failed checks."
        )

    logger.info("Prediction-horizon robustness analysis completed.")

    print()

    print("Institutional Quant Equity Research Platform")

    print("Prediction horizon robustness")

    print("------------------------------------------------")

    print(f"horizons: {len(HORIZONS)}")

    print(f"common_oos_dates: {len(common_dates)}")

    print(f"first_common_date: {common_dates.min().date()}")

    print(f"last_common_date: {common_dates.max().date()}")

    print()

    print("Horizon summary:")

    print(
        summary.loc[
            :,
            [
                "horizon_sessions",
                "months",
                "mean_ic",
                "mean_ic_ci_lower",
                "mean_ic_ci_upper",
                "probability_mean_ic_positive",
                "annualized_ic_ir",
                "positive_ic_ratio",
                "mean_top_bottom_spread",
                "spread_ci_lower",
                "spread_ci_upper",
                "probability_mean_spread_positive",
                "positive_spread_ratio",
                "mean_top_quintile_precision",
                "mean_ic_difference_vs_21d",
                "spread_difference_vs_21d",
            ],
        ].to_string(index=False)
    )

    print()

    print("Yearly horizon stability:")

    print(yearly.to_string(index=False))

    print()

    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()

    print(f"Monthly table: {MONTHLY_PATH}")

    print(f"Summary table: {SUMMARY_PATH}")

    print(f"Yearly table: {YEARLY_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
