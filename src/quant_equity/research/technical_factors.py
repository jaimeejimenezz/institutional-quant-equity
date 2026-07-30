"""Univariate research for monthly technical equity signals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_equity.features import (
    TECHNICAL_MODEL_FEATURE_COLUMNS,
)

REQUIRED_LABEL_COLUMNS = (
    "as_of_date",
    "ticker",
    "first_future_date",
    "target_end_date",
    "horizon_sessions",
    "target_21d",
    "target_21d_excess",
    "label_top_quintile",
)


class TechnicalFactorResearchError(ValueError):
    """Raised when technical-factor research cannot be completed."""


@dataclass(frozen=True)
class TechnicalFactorResearchConfig:
    """Configuration for univariate factor research."""

    target_column: str = "target_21d_excess"
    absolute_return_column: str = "target_21d"
    number_of_quantiles: int = 5
    minimum_cross_section_size: int = 20
    annualization_periods: int = 12
    research_start_date: pd.Timestamp = pd.Timestamp("2014-01-31")
    research_end_date: pd.Timestamp = pd.Timestamp("2019-12-31")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> TechnicalFactorResearchConfig:
        """Create research configuration from YAML values."""
        config = cls(
            target_column=str(
                values.get(
                    "target_column",
                    "target_21d_excess",
                )
            ),
            absolute_return_column=str(
                values.get(
                    "absolute_return_column",
                    "target_21d",
                )
            ),
            number_of_quantiles=int(
                values.get(
                    "number_of_quantiles",
                    5,
                )
            ),
            minimum_cross_section_size=int(
                values.get(
                    "minimum_cross_section_size",
                    20,
                )
            ),
            annualization_periods=int(
                values.get(
                    "annualization_periods",
                    12,
                )
            ),
            research_start_date=pd.Timestamp(
                values.get(
                    "research_start_date",
                    "2014-01-31",
                )
            ).normalize(),
            research_end_date=pd.Timestamp(
                values.get(
                    "research_end_date",
                    "2019-12-31",
                )
            ).normalize(),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate research settings."""
        if self.number_of_quantiles < 2:
            raise TechnicalFactorResearchError("number_of_quantiles must be at least 2.")

        if self.minimum_cross_section_size < self.number_of_quantiles:
            raise TechnicalFactorResearchError(
                "minimum_cross_section_size must be "
                "greater than or equal to the number "
                "of quantiles."
            )

        if self.annualization_periods < 1:
            raise TechnicalFactorResearchError("annualization_periods must be positive.")

        if self.research_start_date > self.research_end_date:
            raise TechnicalFactorResearchError(
                "research_start_date cannot be after research_end_date."
            )


@dataclass
class TechnicalFactorResearchResult:
    """Complete collection of factor-research results."""

    panel: pd.DataFrame
    monthly_ic: pd.DataFrame
    ic_summary: pd.DataFrame
    monthly_quintiles: pd.DataFrame
    quintile_summary: pd.DataFrame
    monthly_spreads: pd.DataFrame
    spread_summary: pd.DataFrame
    monthly_turnover: pd.DataFrame
    turnover_summary: pd.DataFrame


def _require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    """Require a dataframe to contain expected columns."""
    missing = sorted(set(columns).difference(data.columns))

    if missing:
        raise TechnicalFactorResearchError(
            f"{dataset_name} is missing columns: " + ", ".join(missing) + "."
        )


def _normalize_keys(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize date and ticker identifiers."""
    normalized = data.copy()

    normalized["as_of_date"] = pd.to_datetime(
        normalized["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    normalized["ticker"] = normalized["ticker"].astype("string").str.strip().str.upper()

    return normalized


def _rank_correlation(
    first: pd.Series,
    second: pd.Series,
) -> float:
    """Calculate Spearman correlation through ranks."""
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


def build_factor_research_panel(
    technical_features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    config: TechnicalFactorResearchConfig,
    signal_columns: Sequence[str] = (TECHNICAL_MODEL_FEATURE_COLUMNS),
) -> pd.DataFrame:
    """Join point-in-time features with future labels."""
    feature_columns = (
        "as_of_date",
        "ticker",
        "sector",
        "latest_market_date",
        *signal_columns,
    )

    _require_columns(
        technical_features,
        feature_columns,
        dataset_name="Technical features",
    )

    _require_columns(
        labels,
        REQUIRED_LABEL_COLUMNS,
        dataset_name="Monthly labels",
    )

    features = _normalize_keys(
        technical_features.loc[
            :,
            feature_columns,
        ]
    )

    labels_normalized = _normalize_keys(
        labels.loc[
            :,
            REQUIRED_LABEL_COLUMNS,
        ]
    )

    for column in ("latest_market_date",):
        features[column] = pd.to_datetime(
            features[column],
            errors="coerce",
        ).dt.normalize()

    for column in (
        "first_future_date",
        "target_end_date",
    ):
        labels_normalized[column] = pd.to_datetime(
            labels_normalized[column],
            errors="coerce",
        ).dt.normalize()

    feature_duplicates = int(
        features.duplicated(
            subset=[
                "as_of_date",
                "ticker",
            ],
            keep=False,
        ).sum()
    )

    label_duplicates = int(
        labels_normalized.duplicated(
            subset=[
                "as_of_date",
                "ticker",
            ],
            keep=False,
        ).sum()
    )

    if feature_duplicates:
        raise TechnicalFactorResearchError(
            f"Technical features contain {feature_duplicates} duplicated keys."
        )

    if label_duplicates:
        raise TechnicalFactorResearchError(
            f"Monthly labels contain {label_duplicates} duplicated keys."
        )

    features = features.loc[
        features["as_of_date"].between(
            config.research_start_date,
            config.research_end_date,
            inclusive="both",
        )
    ].copy()

    labels_normalized = labels_normalized.loc[
        labels_normalized["as_of_date"].between(
            config.research_start_date,
            config.research_end_date,
            inclusive="both",
        )
    ].copy()

    panel = features.merge(
        labels_normalized,
        on=[
            "as_of_date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched_rows = int(panel["_merge"].ne("both").sum())

    if unmatched_rows:
        raise TechnicalFactorResearchError(
            f"{unmatched_rows} feature rows do not have a corresponding monthly-label row."
        )

    panel = panel.drop(columns="_merge")

    feature_temporal_violations = int(panel["latest_market_date"].gt(panel["as_of_date"]).sum())

    target_start_violations = int(
        (
            panel["first_future_date"].notna() & panel["first_future_date"].le(panel["as_of_date"])
        ).sum()
    )

    target_end_violations = int(
        (
            panel["target_end_date"].notna()
            & panel["first_future_date"].notna()
            & panel["target_end_date"].lt(panel["first_future_date"])
        ).sum()
    )

    if feature_temporal_violations:
        raise TechnicalFactorResearchError(
            f"{feature_temporal_violations} features use information after as_of_date."
        )

    if target_start_violations:
        raise TechnicalFactorResearchError(
            f"{target_start_violations} labels begin on or before as_of_date."
        )

    if target_end_violations:
        raise TechnicalFactorResearchError(
            f"{target_end_violations} labels end before their first future date."
        )

    if config.target_column not in panel.columns:
        raise TechnicalFactorResearchError(
            f"Configured target column was not found: {config.target_column}."
        )

    if config.absolute_return_column not in panel.columns:
        raise TechnicalFactorResearchError(
            f"Configured absolute-return column was not found: {config.absolute_return_column}."
        )

    inside_research_window = panel["as_of_date"].ge(config.research_start_date) & panel[
        "as_of_date"
    ].le(config.research_end_date)

    valid_target = panel[config.target_column].notna()

    panel = (
        panel.loc[inside_research_window & valid_target]
        .sort_values(
            [
                "as_of_date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    if panel.empty:
        raise TechnicalFactorResearchError(
            "No valid observations exist inside the configured research window."
        )

    return panel


def calculate_monthly_information_coefficients(
    panel: pd.DataFrame,
    *,
    config: TechnicalFactorResearchConfig,
    signal_columns: Sequence[str] = (TECHNICAL_MODEL_FEATURE_COLUMNS),
) -> pd.DataFrame:
    """Calculate monthly Spearman Information Coefficients."""
    rows: list[dict[str, Any]] = []

    for signal in signal_columns:
        _require_columns(
            panel,
            (
                "as_of_date",
                signal,
                config.target_column,
            ),
            dataset_name="Factor research panel",
        )

        for as_of_date, date_data in panel.groupby(
            "as_of_date",
            sort=True,
        ):
            comparable = date_data.loc[
                :,
                [
                    signal,
                    config.target_column,
                ],
            ].dropna()

            observations = len(comparable)

            if observations < config.minimum_cross_section_size:
                information_coefficient = float("nan")
            else:
                information_coefficient = _rank_correlation(
                    comparable[signal],
                    comparable[config.target_column],
                )

            rows.append(
                {
                    "signal": signal,
                    "as_of_date": as_of_date,
                    "observations": observations,
                    "ic": information_coefficient,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "signal",
                "as_of_date",
            ]
        )
        .reset_index(drop=True)
    )


def calculate_ic_summary(
    monthly_ic: pd.DataFrame,
    *,
    config: TechnicalFactorResearchConfig,
) -> pd.DataFrame:
    """Summarize monthly Information Coefficients."""
    rows: list[dict[str, Any]] = []

    for signal, signal_data in monthly_ic.groupby(
        "signal",
        sort=True,
    ):
        values = signal_data["ic"].dropna().astype(float)

        months = len(values)

        mean_ic = float(values.mean()) if months else float("nan")

        median_ic = float(values.median()) if months else float("nan")

        std_ic = float(values.std(ddof=1)) if months > 1 else float("nan")

        if np.isfinite(std_ic) and std_ic > np.finfo(float).eps:
            ic_ir = mean_ic / std_ic
            annualized_ic_ir = ic_ir * np.sqrt(config.annualization_periods)
            ic_t_stat = mean_ic / (std_ic / np.sqrt(months))
        else:
            ic_ir = float("nan")
            annualized_ic_ir = float("nan")
            ic_t_stat = float("nan")

        positive_month_ratio = float(values.gt(0.0).mean()) if months else float("nan")

        rows.append(
            {
                "signal": signal,
                "months": months,
                "mean_ic": mean_ic,
                "median_ic": median_ic,
                "std_ic": std_ic,
                "ic_ir": ic_ir,
                "annualized_ic_ir": (annualized_ic_ir),
                "ic_t_stat": ic_t_stat,
                "positive_month_ratio": (positive_month_ratio),
                "abs_mean_ic": abs(mean_ic),
                "preferred_direction": (
                    "higher_is_better" if mean_ic >= 0.0 else "lower_is_better"
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "abs_mean_ic",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def _assign_quantiles(
    data: pd.DataFrame,
    *,
    signal: str,
    number_of_quantiles: int,
) -> pd.Series:
    """Assign deterministic cross-sectional quantiles."""
    ranks = data[signal].rank(method="first")

    return (
        pd.qcut(
            ranks,
            q=number_of_quantiles,
            labels=False,
        )
        .astype(int)
        .add(1)
    )


def calculate_quintile_research(
    panel: pd.DataFrame,
    *,
    config: TechnicalFactorResearchConfig,
    signal_columns: Sequence[str] = (TECHNICAL_MODEL_FEATURE_COLUMNS),
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Calculate monthly and aggregate quintile results."""
    quintile_rows: list[dict[str, Any]] = []

    spread_rows: list[dict[str, Any]] = []

    for signal in signal_columns:
        for as_of_date, date_data in panel.groupby(
            "as_of_date",
            sort=True,
        ):
            comparable = date_data.loc[
                :,
                [
                    "ticker",
                    signal,
                    config.target_column,
                    config.absolute_return_column,
                ],
            ].dropna(
                subset=[
                    signal,
                    config.target_column,
                ]
            )

            if (
                len(comparable) < config.minimum_cross_section_size
                or comparable[signal].nunique() < config.number_of_quantiles
            ):
                continue

            comparable = comparable.copy()

            comparable["quintile"] = _assign_quantiles(
                comparable,
                signal=signal,
                number_of_quantiles=(config.number_of_quantiles),
            )

            monthly_quantile_means: dict[
                int,
                float,
            ] = {}

            for quintile, quintile_data in comparable.groupby(
                "quintile",
                sort=True,
            ):
                mean_excess_return = float(quintile_data[config.target_column].mean())

                absolute_returns = (
                    quintile_data[config.absolute_return_column].dropna().astype(float)
                )

                mean_absolute_return = (
                    float(absolute_returns.mean()) if len(absolute_returns) else float("nan")
                )

                quintile_rows.append(
                    {
                        "signal": signal,
                        "as_of_date": as_of_date,
                        "quintile": int(quintile),
                        "observations": len(quintile_data),
                        "mean_target_21d_excess": (mean_excess_return),
                        "median_target_21d_excess": float(
                            quintile_data[config.target_column].median()
                        ),
                        "mean_target_21d": (mean_absolute_return),
                    }
                )

                monthly_quantile_means[int(quintile)] = mean_excess_return

            bottom_quantile = 1
            top_quantile = config.number_of_quantiles

            if (
                bottom_quantile not in monthly_quantile_means
                or top_quantile not in monthly_quantile_means
            ):
                continue

            ordered_quantiles = sorted(monthly_quantile_means)

            monotonicity = _rank_correlation(
                pd.Series(
                    ordered_quantiles,
                    dtype=float,
                ),
                pd.Series(
                    [monthly_quantile_means[quintile] for quintile in ordered_quantiles],
                    dtype=float,
                ),
            )

            spread_rows.append(
                {
                    "signal": signal,
                    "as_of_date": as_of_date,
                    "bottom_quintile_return": (monthly_quantile_means[bottom_quantile]),
                    "top_quintile_return": (monthly_quantile_means[top_quantile]),
                    "top_bottom_spread": (
                        monthly_quantile_means[top_quantile]
                        - monthly_quantile_means[bottom_quantile]
                    ),
                    "quintile_monotonicity": (monotonicity),
                }
            )

    monthly_quintiles = pd.DataFrame(quintile_rows)

    monthly_spreads = pd.DataFrame(spread_rows)

    if monthly_quintiles.empty:
        raise TechnicalFactorResearchError("No valid factor quintiles could be created.")

    quintile_summary = monthly_quintiles.groupby(
        [
            "signal",
            "quintile",
        ],
        as_index=False,
        sort=True,
    ).agg(
        months=(
            "as_of_date",
            "nunique",
        ),
        mean_target_21d_excess=(
            "mean_target_21d_excess",
            "mean",
        ),
        median_target_21d_excess=(
            "median_target_21d_excess",
            "median",
        ),
        mean_target_21d=(
            "mean_target_21d",
            "mean",
        ),
        mean_observations=(
            "observations",
            "mean",
        ),
    )

    spread_summary_rows: list[dict[str, Any]] = []

    for signal, signal_data in monthly_spreads.groupby(
        "signal",
        sort=True,
    ):
        spreads = signal_data["top_bottom_spread"].dropna()

        monotonicity = signal_data["quintile_monotonicity"].dropna()

        spread_summary_rows.append(
            {
                "signal": signal,
                "months": len(spreads),
                "mean_top_bottom_spread": float(spreads.mean()),
                "median_top_bottom_spread": float(spreads.median()),
                "spread_std": float(spreads.std(ddof=1)) if len(spreads) > 1 else float("nan"),
                "positive_spread_ratio": float(spreads.gt(0.0).mean()),
                "mean_quintile_monotonicity": (
                    float(monotonicity.mean()) if len(monotonicity) else float("nan")
                ),
                "positive_monotonicity_ratio": (
                    float(monotonicity.gt(0.0).mean()) if len(monotonicity) else float("nan")
                ),
            }
        )

    spread_summary = (
        pd.DataFrame(spread_summary_rows)
        .sort_values(
            "mean_top_bottom_spread",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return (
        monthly_quintiles,
        quintile_summary,
        monthly_spreads,
        spread_summary,
    )


def calculate_selected_quantile_turnover(
    panel: pd.DataFrame,
    ic_summary: pd.DataFrame,
    *,
    config: TechnicalFactorResearchConfig,
    signal_columns: Sequence[str] = (TECHNICAL_MODEL_FEATURE_COLUMNS),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate turnover of the economically preferred quantile."""
    direction_by_signal = ic_summary.set_index("signal")["preferred_direction"].to_dict()

    turnover_rows: list[dict[str, Any]] = []

    for signal in signal_columns:
        preferred_direction = direction_by_signal.get(
            signal,
            "higher_is_better",
        )

        selected_quantile = (
            config.number_of_quantiles if preferred_direction == "higher_is_better" else 1
        )

        previous_date: pd.Timestamp | None = None
        previous_tickers: set[str] | None = None

        for as_of_date, date_data in panel.groupby(
            "as_of_date",
            sort=True,
        ):
            comparable = date_data.loc[
                :,
                [
                    "ticker",
                    signal,
                ],
            ].dropna()

            if (
                len(comparable) < config.minimum_cross_section_size
                or comparable[signal].nunique() < config.number_of_quantiles
            ):
                continue

            comparable = comparable.copy()

            comparable["quintile"] = _assign_quantiles(
                comparable,
                signal=signal,
                number_of_quantiles=(config.number_of_quantiles),
            )

            current_tickers = set(
                comparable.loc[
                    comparable["quintile"].eq(selected_quantile),
                    "ticker",
                ]
            )

            if previous_tickers is not None and previous_date is not None:
                overlap = len(previous_tickers.intersection(current_tickers))

                denominator = len(previous_tickers) + len(current_tickers)

                turnover = 1.0 - (2.0 * overlap / denominator) if denominator else float("nan")

                turnover_rows.append(
                    {
                        "signal": signal,
                        "from_date": previous_date,
                        "to_date": as_of_date,
                        "preferred_direction": (preferred_direction),
                        "selected_quantile": (selected_quantile),
                        "previous_members": len(previous_tickers),
                        "current_members": len(current_tickers),
                        "overlapping_members": (overlap),
                        "turnover": turnover,
                    }
                )

            previous_date = as_of_date
            previous_tickers = current_tickers

    monthly_turnover = pd.DataFrame(turnover_rows)

    turnover_summary_rows: list[dict[str, Any]] = []

    for signal, signal_data in monthly_turnover.groupby(
        "signal",
        sort=True,
    ):
        turnover = signal_data["turnover"].dropna()

        turnover_summary_rows.append(
            {
                "signal": signal,
                "transitions": len(turnover),
                "mean_turnover": float(turnover.mean()),
                "median_turnover": float(turnover.median()),
                "maximum_turnover": float(turnover.max()),
                "preferred_direction": (signal_data["preferred_direction"].iloc[0]),
                "selected_quantile": int(signal_data["selected_quantile"].iloc[0]),
            }
        )

    turnover_summary = (
        pd.DataFrame(turnover_summary_rows).sort_values("mean_turnover").reset_index(drop=True)
    )

    return (
        monthly_turnover,
        turnover_summary,
    )


def run_technical_factor_research(
    technical_features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    config: TechnicalFactorResearchConfig,
    signal_columns: Sequence[str] = (TECHNICAL_MODEL_FEATURE_COLUMNS),
) -> TechnicalFactorResearchResult:
    """Run the complete univariate factor-research workflow."""
    panel = build_factor_research_panel(
        technical_features,
        labels,
        config=config,
        signal_columns=signal_columns,
    )

    monthly_ic = calculate_monthly_information_coefficients(
        panel,
        config=config,
        signal_columns=signal_columns,
    )

    ic_summary = calculate_ic_summary(
        monthly_ic,
        config=config,
    )

    (
        monthly_quintiles,
        quintile_summary,
        monthly_spreads,
        spread_summary,
    ) = calculate_quintile_research(
        panel,
        config=config,
        signal_columns=signal_columns,
    )

    (
        monthly_turnover,
        turnover_summary,
    ) = calculate_selected_quantile_turnover(
        panel,
        ic_summary,
        config=config,
        signal_columns=signal_columns,
    )

    return TechnicalFactorResearchResult(
        panel=panel,
        monthly_ic=monthly_ic,
        ic_summary=ic_summary,
        monthly_quintiles=monthly_quintiles,
        quintile_summary=quintile_summary,
        monthly_spreads=monthly_spreads,
        spread_summary=spread_summary,
        monthly_turnover=monthly_turnover,
        turnover_summary=turnover_summary,
    )
