"""Stability, redundancy and sector diagnostics for technical factors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


class TechnicalFactorDiagnosticsError(ValueError):
    """Raised when technical-factor diagnostics cannot be completed."""


@dataclass(frozen=True)
class ResearchPeriod:
    """Named research period."""

    name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> ResearchPeriod:
        """Create a period from configuration values."""
        return cls(
            name=str(values["name"]),
            start_date=pd.Timestamp(values["start_date"]).normalize(),
            end_date=pd.Timestamp(values["end_date"]).normalize(),
        )

    def validate(self) -> None:
        """Validate period boundaries."""
        if self.start_date > self.end_date:
            raise TechnicalFactorDiagnosticsError(
                f"Invalid research period {self.name}: start_date is after end_date."
            )


@dataclass(frozen=True)
class TechnicalFactorDiagnosticsConfig:
    """Configuration for extended technical-factor diagnostics."""

    correlation_threshold: float = 0.90
    minimum_pair_observations: int = 20
    minimum_sector_cross_section_size: int = 3
    minimum_sector_months: int = 12
    top_signals_in_figures: int = 10
    subperiods: tuple[ResearchPeriod, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> TechnicalFactorDiagnosticsConfig:
        """Create diagnostics configuration from YAML values."""
        periods = tuple(
            ResearchPeriod.from_mapping(period)
            for period in values.get(
                "subperiods",
                (),
            )
        )

        config = cls(
            correlation_threshold=float(
                values.get(
                    "correlation_threshold",
                    0.90,
                )
            ),
            minimum_pair_observations=int(
                values.get(
                    "minimum_pair_observations",
                    20,
                )
            ),
            minimum_sector_cross_section_size=int(
                values.get(
                    "minimum_sector_cross_section_size",
                    3,
                )
            ),
            minimum_sector_months=int(
                values.get(
                    "minimum_sector_months",
                    12,
                )
            ),
            top_signals_in_figures=int(
                values.get(
                    "top_signals_in_figures",
                    10,
                )
            ),
            subperiods=periods,
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate diagnostics settings."""
        if not 0.0 < self.correlation_threshold <= 1.0:
            raise TechnicalFactorDiagnosticsError("correlation_threshold must be in (0, 1].")

        if self.minimum_pair_observations < 2:
            raise TechnicalFactorDiagnosticsError("minimum_pair_observations must be at least 2.")

        if self.minimum_sector_cross_section_size < 2:
            raise TechnicalFactorDiagnosticsError(
                "minimum_sector_cross_section_size must be at least 2."
            )

        if self.minimum_sector_months < 1:
            raise TechnicalFactorDiagnosticsError("minimum_sector_months must be positive.")

        if self.top_signals_in_figures < 1:
            raise TechnicalFactorDiagnosticsError("top_signals_in_figures must be positive.")

        for period in self.subperiods:
            period.validate()


def _require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    """Require columns to exist in a dataframe."""
    missing = sorted(set(columns).difference(data.columns))

    if missing:
        raise TechnicalFactorDiagnosticsError(
            f"{dataset_name} is missing columns: " + ", ".join(missing) + "."
        )


def _safe_spearman(
    first: pd.Series,
    second: pd.Series,
) -> float:
    """Calculate Spearman correlation after removing invalid values."""
    comparable = pd.DataFrame(
        {
            "first": pd.to_numeric(
                first,
                errors="coerce",
            ),
            "second": pd.to_numeric(
                second,
                errors="coerce",
            ),
        }
    ).dropna()

    if (
        len(comparable) < 2
        or comparable["first"].nunique() < 2
        or comparable["second"].nunique() < 2
    ):
        return float("nan")

    first_ranks = comparable["first"].rank(method="average")

    second_ranks = comparable["second"].rank(method="average")

    return float(first_ranks.corr(second_ranks))


def _direction_multiplier(
    preferred_direction: str,
) -> float:
    """Convert a preferred direction into a numerical sign."""
    if preferred_direction == "higher_is_better":
        return 1.0

    if preferred_direction == "lower_is_better":
        return -1.0

    raise TechnicalFactorDiagnosticsError(f"Unknown preferred direction: {preferred_direction}.")


def calculate_monthly_signal_correlations(
    panel: pd.DataFrame,
    *,
    signal_columns: Sequence[str],
    minimum_observations: int,
) -> pd.DataFrame:
    """Calculate monthly cross-sectional signal correlations."""
    _require_columns(
        panel,
        (
            "as_of_date",
            *signal_columns,
        ),
        dataset_name="Factor research panel",
    )

    rows: list[dict[str, Any]] = []

    signal_pairs = tuple(
        combinations(
            signal_columns,
            2,
        )
    )

    for as_of_date, date_data in panel.groupby(
        "as_of_date",
        sort=True,
    ):
        for first_signal, second_signal in signal_pairs:
            comparable = date_data.loc[
                :,
                [
                    first_signal,
                    second_signal,
                ],
            ].dropna()

            observations = len(comparable)

            if observations < minimum_observations:
                correlation = float("nan")
            else:
                correlation = _safe_spearman(
                    comparable[first_signal],
                    comparable[second_signal],
                )

            rows.append(
                {
                    "as_of_date": as_of_date,
                    "first_signal": first_signal,
                    "second_signal": second_signal,
                    "observations": observations,
                    "spearman_correlation": correlation,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "first_signal",
                "second_signal",
                "as_of_date",
            ]
        )
        .reset_index(drop=True)
    )


def summarize_signal_correlations(
    monthly_correlations: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize cross-sectional correlations through time."""
    _require_columns(
        monthly_correlations,
        (
            "first_signal",
            "second_signal",
            "spearman_correlation",
        ),
        dataset_name="Monthly signal correlations",
    )

    rows: list[dict[str, Any]] = []

    grouped = monthly_correlations.groupby(
        [
            "first_signal",
            "second_signal",
        ],
        sort=True,
    )

    for (
        first_signal,
        second_signal,
    ), pair_data in grouped:
        correlations = pair_data["spearman_correlation"].dropna()

        if correlations.empty:
            continue

        rows.append(
            {
                "first_signal": first_signal,
                "second_signal": second_signal,
                "months": len(correlations),
                "mean_correlation": float(correlations.mean()),
                "median_correlation": float(correlations.median()),
                "mean_absolute_correlation": float(correlations.abs().mean()),
                "maximum_absolute_correlation": float(correlations.abs().max()),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "first_signal",
                "second_signal",
                "months",
                "mean_correlation",
                "median_correlation",
                "mean_absolute_correlation",
                "maximum_absolute_correlation",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "mean_absolute_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_mean_correlation_matrix(
    correlation_summary: pd.DataFrame,
    *,
    signal_columns: Sequence[str],
) -> pd.DataFrame:
    """Build a symmetric matrix of mean signal correlations."""
    matrix = pd.DataFrame(
        np.eye(
            len(signal_columns),
            dtype=float,
        ),
        index=signal_columns,
        columns=signal_columns,
    )

    for row in correlation_summary.itertuples(index=False):
        matrix.loc[
            row.first_signal,
            row.second_signal,
        ] = row.mean_correlation

        matrix.loc[
            row.second_signal,
            row.first_signal,
        ] = row.mean_correlation

    matrix.index.name = "signal"

    return matrix


def summarize_ic_by_period(
    monthly_ic: pd.DataFrame,
    *,
    periods: Sequence[ResearchPeriod],
    preferred_direction_by_signal: Mapping[str, str],
) -> pd.DataFrame:
    """Summarize monthly IC values by named time period."""
    _require_columns(
        monthly_ic,
        (
            "signal",
            "as_of_date",
            "ic",
        ),
        dataset_name="Monthly factor IC",
    )

    normalized = monthly_ic.copy()

    normalized["as_of_date"] = pd.to_datetime(
        normalized["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    rows: list[dict[str, Any]] = []

    for period in periods:
        inside_period = normalized["as_of_date"].between(
            period.start_date,
            period.end_date,
            inclusive="both",
        )

        period_data = normalized.loc[inside_period]

        for signal, signal_data in period_data.groupby(
            "signal",
            sort=True,
        ):
            values = signal_data["ic"].dropna().astype(float)

            if values.empty:
                continue

            preferred_direction = preferred_direction_by_signal.get(
                signal,
                "higher_is_better",
            )

            multiplier = _direction_multiplier(preferred_direction)

            directional_values = values * multiplier

            rows.append(
                {
                    "period": period.name,
                    "period_start": period.start_date,
                    "period_end": period.end_date,
                    "signal": signal,
                    "months": len(values),
                    "mean_ic": float(values.mean()),
                    "median_ic": float(values.median()),
                    "std_ic": (float(values.std(ddof=1)) if len(values) > 1 else float("nan")),
                    "preferred_direction": (preferred_direction),
                    "directional_mean_ic": float(directional_values.mean()),
                    "directional_month_ratio": float(directional_values.gt(0.0).mean()),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "period_start",
                "signal",
            ]
        )
        .reset_index(drop=True)
    )


def calculate_monthly_sector_ic(
    panel: pd.DataFrame,
    *,
    target_column: str,
    signal_columns: Sequence[str],
    minimum_sector_size: int,
) -> pd.DataFrame:
    """Calculate monthly IC separately inside each sector."""
    _require_columns(
        panel,
        (
            "as_of_date",
            "sector",
            target_column,
            *signal_columns,
        ),
        dataset_name="Factor research panel",
    )

    rows: list[dict[str, Any]] = []

    grouped = panel.groupby(
        [
            "as_of_date",
            "sector",
        ],
        sort=True,
        dropna=False,
    )

    for (
        as_of_date,
        sector,
    ), sector_data in grouped:
        for signal in signal_columns:
            comparable = sector_data.loc[
                :,
                [
                    signal,
                    target_column,
                ],
            ].dropna()

            observations = len(comparable)

            if observations < minimum_sector_size:
                information_coefficient = float("nan")
            else:
                information_coefficient = _safe_spearman(
                    comparable[signal],
                    comparable[target_column],
                )

            rows.append(
                {
                    "as_of_date": as_of_date,
                    "sector": sector,
                    "signal": signal,
                    "observations": observations,
                    "ic": information_coefficient,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "signal",
                "sector",
                "as_of_date",
            ]
        )
        .reset_index(drop=True)
    )


def summarize_sector_ic(
    monthly_sector_ic: pd.DataFrame,
    *,
    preferred_direction_by_signal: Mapping[str, str],
) -> pd.DataFrame:
    """Summarize monthly sector IC values."""
    _require_columns(
        monthly_sector_ic,
        (
            "sector",
            "signal",
            "ic",
        ),
        dataset_name="Monthly sector IC",
    )

    rows: list[dict[str, Any]] = []

    grouped = monthly_sector_ic.groupby(
        [
            "signal",
            "sector",
        ],
        sort=True,
        dropna=False,
    )

    for (
        signal,
        sector,
    ), sector_data in grouped:
        values = sector_data["ic"].dropna().astype(float)

        if values.empty:
            continue

        preferred_direction = preferred_direction_by_signal.get(
            signal,
            "higher_is_better",
        )

        multiplier = _direction_multiplier(preferred_direction)

        directional_values = values * multiplier

        rows.append(
            {
                "signal": signal,
                "sector": sector,
                "months": len(values),
                "mean_ic": float(values.mean()),
                "median_ic": float(values.median()),
                "preferred_direction": (preferred_direction),
                "directional_mean_ic": float(directional_values.mean()),
                "directional_month_ratio": float(directional_values.gt(0.0).mean()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "signal",
                "sector",
            ]
        )
        .reset_index(drop=True)
    )


def _build_maximum_correlation_table(
    correlation_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Find the strongest correlated partner of every signal."""
    rows: list[dict[str, Any]] = []

    for row in correlation_summary.itertuples(index=False):
        rows.extend(
            [
                {
                    "signal": row.first_signal,
                    "strongest_correlated_signal": (row.second_signal),
                    "strongest_mean_correlation": (row.mean_correlation),
                    "strongest_mean_absolute_correlation": (row.mean_absolute_correlation),
                },
                {
                    "signal": row.second_signal,
                    "strongest_correlated_signal": (row.first_signal),
                    "strongest_mean_correlation": (row.mean_correlation),
                    "strongest_mean_absolute_correlation": (row.mean_absolute_correlation),
                },
            ]
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "signal",
                "strongest_correlated_signal",
                "strongest_mean_correlation",
                "strongest_mean_absolute_correlation",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "strongest_mean_absolute_correlation",
            ascending=False,
        )
        .drop_duplicates(
            subset="signal",
            keep="first",
        )
        .reset_index(drop=True)
    )


def build_selection_diagnostics(
    ic_summary: pd.DataFrame,
    spread_summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    yearly_ic: pd.DataFrame,
    subperiod_ic: pd.DataFrame,
    sector_ic_summary: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    *,
    config: TechnicalFactorDiagnosticsConfig,
) -> pd.DataFrame:
    """Build a table supporting the final feature-selection decision."""
    ic_columns = [
        "signal",
        "months",
        "mean_ic",
        "median_ic",
        "std_ic",
        "annualized_ic_ir",
        "ic_t_stat",
        "positive_month_ratio",
        "abs_mean_ic",
        "preferred_direction",
    ]

    spread_columns = [
        "signal",
        "mean_top_bottom_spread",
        "positive_spread_ratio",
        "mean_quintile_monotonicity",
    ]

    turnover_columns = [
        "signal",
        "mean_turnover",
        "median_turnover",
    ]

    diagnostics = (
        ic_summary.loc[
            :,
            ic_columns,
        ]
        .merge(
            spread_summary.loc[
                :,
                spread_columns,
            ],
            on="signal",
            how="left",
        )
        .merge(
            turnover_summary.loc[
                :,
                turnover_columns,
            ],
            on="signal",
            how="left",
        )
    )

    multiplier = diagnostics["preferred_direction"].map(
        {
            "higher_is_better": 1.0,
            "lower_is_better": -1.0,
        }
    )

    diagnostics["directional_month_ratio"] = np.where(
        multiplier.eq(1.0),
        diagnostics["positive_month_ratio"],
        1.0 - diagnostics["positive_month_ratio"],
    )

    diagnostics["directional_spread"] = diagnostics["mean_top_bottom_spread"] * multiplier

    diagnostics["directional_positive_spread_ratio"] = np.where(
        multiplier.eq(1.0),
        diagnostics["positive_spread_ratio"],
        1.0 - diagnostics["positive_spread_ratio"],
    )

    maximum_months = diagnostics["months"].max()

    diagnostics["coverage_ratio"] = diagnostics["months"] / maximum_months

    yearly_breadth = yearly_ic.groupby(
        "signal",
        as_index=False,
    ).agg(
        years_available=(
            "period",
            "nunique",
        ),
        positive_year_ratio=(
            "directional_mean_ic",
            lambda values: float(values.gt(0.0).mean()),
        ),
        worst_year_directional_ic=(
            "directional_mean_ic",
            "min",
        ),
    )

    subperiod_breadth = subperiod_ic.groupby(
        "signal",
        as_index=False,
    ).agg(
        subperiods_available=(
            "period",
            "nunique",
        ),
        positive_subperiod_ratio=(
            "directional_mean_ic",
            lambda values: float(values.gt(0.0).mean()),
        ),
        worst_subperiod_directional_ic=(
            "directional_mean_ic",
            "min",
        ),
    )

    valid_sector_results = sector_ic_summary.loc[
        sector_ic_summary["months"].ge(config.minimum_sector_months)
    ]

    sector_breadth = valid_sector_results.groupby(
        "signal",
        as_index=False,
    ).agg(
        sectors_available=(
            "sector",
            "nunique",
        ),
        positive_sector_ratio=(
            "directional_mean_ic",
            lambda values: float(values.gt(0.0).mean()),
        ),
        median_sector_directional_ic=(
            "directional_mean_ic",
            "median",
        ),
    )

    strongest_correlations = _build_maximum_correlation_table(correlation_summary)

    diagnostics = (
        diagnostics.merge(
            yearly_breadth,
            on="signal",
            how="left",
        )
        .merge(
            subperiod_breadth,
            on="signal",
            how="left",
        )
        .merge(
            sector_breadth,
            on="signal",
            how="left",
        )
        .merge(
            strongest_correlations,
            on="signal",
            how="left",
        )
    )

    def classify_signal(
        row: pd.Series,
    ) -> str:
        if row["months"] == 0 or pd.isna(row["abs_mean_ic"]):
            return "drop_no_variation"

        if row["abs_mean_ic"] < 0.005:
            return "drop_very_weak"

        if (
            pd.notna(row["strongest_mean_absolute_correlation"])
            and row["strongest_mean_absolute_correlation"] >= config.correlation_threshold
        ):
            return "review_redundancy"

        if row["abs_mean_ic"] < 0.010:
            return "weak_candidate"

        if pd.notna(row["mean_turnover"]) and row["mean_turnover"] > 0.75:
            return "candidate_high_turnover"

        if row["directional_month_ratio"] < 0.52 or row["directional_positive_spread_ratio"] < 0.52:
            return "candidate_unstable"

        return "candidate"

    diagnostics["preliminary_status"] = diagnostics.apply(
        classify_signal,
        axis=1,
    )

    output_columns = [
        "signal",
        "preliminary_status",
        "months",
        "coverage_ratio",
        "mean_ic",
        "abs_mean_ic",
        "annualized_ic_ir",
        "ic_t_stat",
        "preferred_direction",
        "directional_month_ratio",
        "directional_spread",
        "directional_positive_spread_ratio",
        "mean_quintile_monotonicity",
        "mean_turnover",
        "positive_year_ratio",
        "worst_year_directional_ic",
        "positive_subperiod_ratio",
        "worst_subperiod_directional_ic",
        "sectors_available",
        "positive_sector_ratio",
        "median_sector_directional_ic",
        "strongest_correlated_signal",
        "strongest_mean_correlation",
        "strongest_mean_absolute_correlation",
    ]

    return (
        diagnostics.loc[
            :,
            output_columns,
        ]
        .sort_values(
            [
                "abs_mean_ic",
                "directional_spread",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )
