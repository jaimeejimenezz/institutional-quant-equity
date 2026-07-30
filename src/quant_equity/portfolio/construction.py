"""Portfolio construction from out-of-sample model predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioConstructionError(ValueError):
    """Raised when target portfolios cannot be constructed."""


@dataclass(frozen=True)
class PortfolioConstructionConfig:
    """Configuration for the MVP portfolio-construction process."""

    primary_model_name: str = "elastic_net"
    challenger_model_name: str = "ridge"
    momentum_model_name: str = "momentum_3m"

    top_n: int = 20
    score_weighted_candidate_count: int = 25

    max_weight: float = 0.05
    max_sector_weight: float = 0.25

    minimum_cross_section_size: int = 30
    weight_tolerance: float = 1.0e-8

    optimization_tolerance: float = 1.0e-10
    optimization_max_iterations: int = 2000

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> PortfolioConstructionConfig:
        """Create configuration from YAML values."""
        config = cls(
            primary_model_name=str(
                values.get(
                    "primary_model_name",
                    "elastic_net",
                )
            ),
            challenger_model_name=str(
                values.get(
                    "challenger_model_name",
                    "ridge",
                )
            ),
            momentum_model_name=str(
                values.get(
                    "momentum_model_name",
                    "momentum_3m",
                )
            ),
            top_n=int(
                values.get(
                    "top_n",
                    20,
                )
            ),
            score_weighted_candidate_count=int(
                values.get(
                    "score_weighted_candidate_count",
                    25,
                )
            ),
            max_weight=float(
                values.get(
                    "max_weight",
                    0.05,
                )
            ),
            max_sector_weight=float(
                values.get(
                    "max_sector_weight",
                    0.25,
                )
            ),
            minimum_cross_section_size=int(
                values.get(
                    "minimum_cross_section_size",
                    30,
                )
            ),
            weight_tolerance=float(
                values.get(
                    "weight_tolerance",
                    1.0e-8,
                )
            ),
            optimization_tolerance=float(
                values.get(
                    "optimization_tolerance",
                    1.0e-10,
                )
            ),
            optimization_max_iterations=int(
                values.get(
                    "optimization_max_iterations",
                    2000,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate portfolio-construction settings."""
        for model_name in (
            self.primary_model_name,
            self.challenger_model_name,
            self.momentum_model_name,
        ):
            if not model_name:
                raise PortfolioConstructionError("Model names cannot be empty.")

        if self.top_n < 1:
            raise PortfolioConstructionError("top_n must be positive.")

        if self.score_weighted_candidate_count < self.top_n:
            raise PortfolioConstructionError(
                "score_weighted_candidate_count cannot be smaller than top_n."
            )

        if not 0.0 < self.max_weight <= 1.0:
            raise PortfolioConstructionError("max_weight must be in (0, 1].")

        if not 0.0 < self.max_sector_weight <= 1.0:
            raise PortfolioConstructionError("max_sector_weight must be in (0, 1].")

        if self.max_sector_weight < self.max_weight:
            raise PortfolioConstructionError("max_sector_weight cannot be smaller than max_weight.")

        minimum_required_holdings = int(np.ceil(1.0 / self.max_weight - self.weight_tolerance))

        if self.top_n < minimum_required_holdings:
            raise PortfolioConstructionError(
                "top_n is too small to invest 100% of the capital under max_weight."
            )

        if self.score_weighted_candidate_count < minimum_required_holdings:
            raise PortfolioConstructionError(
                "score_weighted_candidate_count is too small to invest 100% of the capital."
            )

        if self.minimum_cross_section_size < self.score_weighted_candidate_count:
            raise PortfolioConstructionError(
                "minimum_cross_section_size cannot be smaller than score_weighted_candidate_count."
            )

        if self.weight_tolerance <= 0.0:
            raise PortfolioConstructionError("weight_tolerance must be positive.")

        if self.optimization_tolerance <= 0.0:
            raise PortfolioConstructionError("optimization_tolerance must be positive.")

        if self.optimization_max_iterations < 1:
            raise PortfolioConstructionError("optimization_max_iterations must be positive.")


@dataclass(frozen=True)
class MVPPortfolioConstructionOutputs:
    """Outputs generated by Step 8A."""

    target_weights: pd.DataFrame
    constraint_checks: pd.DataFrame
    portfolio_summary: pd.DataFrame


def _require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    """Require dataframe columns."""
    missing = sorted(set(columns).difference(data.columns))

    if missing:
        raise PortfolioConstructionError(
            f"{dataset_name} is missing columns: " + ", ".join(missing) + "."
        )


def _validate_predictions(
    predictions: pd.DataFrame,
    *,
    config: PortfolioConstructionConfig,
) -> pd.DataFrame:
    """Validate and normalize evaluated OOS predictions."""
    required_columns = (
        "as_of_date",
        "ticker",
        "sector",
        "model_name",
        "prediction",
    )

    _require_columns(
        predictions,
        required_columns,
        dataset_name="Evaluated linear-model predictions",
    )

    data = predictions.copy()

    data["as_of_date"] = pd.to_datetime(
        data["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    if data["as_of_date"].isna().any():
        raise PortfolioConstructionError("Predictions contain invalid as_of_date values.")

    for column in (
        "ticker",
        "sector",
        "model_name",
    ):
        data[column] = data[column].astype("string").str.strip()

        if data[column].isna().any() or data[column].eq("").any():
            raise PortfolioConstructionError(f"Predictions contain missing values in {column}.")

    data["prediction"] = pd.to_numeric(
        data["prediction"],
        errors="coerce",
    )

    if data["prediction"].isna().any():
        raise PortfolioConstructionError("Predictions contain invalid numeric predictions.")

    if np.isinf(data["prediction"].to_numpy(dtype=float)).any():
        raise PortfolioConstructionError("Predictions contain infinite values.")

    duplicate_count = int(
        data.duplicated(
            [
                "as_of_date",
                "ticker",
                "model_name",
            ]
        ).sum()
    )

    if duplicate_count:
        raise PortfolioConstructionError("Predictions contain duplicated date-ticker-model rows.")

    required_models = {
        config.primary_model_name,
        config.challenger_model_name,
        config.momentum_model_name,
    }

    available_models = set(data["model_name"].unique())

    missing_models = sorted(required_models.difference(available_models))

    if missing_models:
        raise PortfolioConstructionError(
            "Required models are missing: "
            + ", ".join(missing_models)
            + ". Available models: "
            + ", ".join(sorted(available_models))
            + "."
        )

    relevant = data.loc[data["model_name"].isin(required_models)].copy()

    cross_section_sizes = relevant.groupby(
        [
            "model_name",
            "as_of_date",
        ]
    )["ticker"].nunique()

    undersized = cross_section_sizes.loc[cross_section_sizes.lt(config.minimum_cross_section_size)]

    if not undersized.empty:
        first_index = undersized.index[0]

        raise PortfolioConstructionError(
            "At least one model-date cross-section is too "
            "small. First invalid cross-section: "
            f"{first_index}, size={int(undersized.iloc[0])}."
        )

    return relevant.sort_values(
        [
            "as_of_date",
            "model_name",
            "ticker",
        ]
    ).reset_index(drop=True)


def _validate_model_alignment(
    primary: pd.DataFrame,
    other: pd.DataFrame,
    *,
    other_model_name: str,
    as_of_date: pd.Timestamp,
) -> None:
    """Ensure models contain the same companies and sectors."""
    primary_reference = (
        primary.loc[
            :,
            [
                "ticker",
                "sector",
            ],
        ]
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    other_reference = (
        other.loc[
            :,
            [
                "ticker",
                "sector",
            ],
        ]
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    if not primary_reference.equals(other_reference):
        raise PortfolioConstructionError(
            "Model cross-sections are not aligned on "
            f"{as_of_date.date()} for model "
            f"{other_model_name}."
        )


def _calculate_selection_ranks(
    month: pd.DataFrame,
) -> pd.Series:
    """Create deterministic descending prediction ranks."""
    ordered = month.sort_values(
        [
            "prediction",
            "ticker",
        ],
        ascending=[
            False,
            True,
        ],
    )

    rank_by_ticker = pd.Series(
        np.arange(
            1,
            len(ordered) + 1,
            dtype=int,
        ),
        index=ordered["ticker"],
    )

    return month["ticker"].map(rank_by_ticker).astype("Int64")


def _maximum_names_per_sector(
    *,
    config: PortfolioConstructionConfig,
) -> int:
    """Return the maximum fully capped names per sector."""
    maximum_names = int(
        np.floor(config.max_sector_weight / config.max_weight + config.weight_tolerance)
    )

    if maximum_names < 1:
        raise PortfolioConstructionError("The sector and asset caps are incompatible.")

    return maximum_names


def _select_ranked_candidates(
    month: pd.DataFrame,
    *,
    candidate_count: int,
    config: PortfolioConstructionConfig,
) -> list[str]:
    """Select candidates while limiting concentration by sector."""
    maximum_sector_names = _maximum_names_per_sector(
        config=config,
    )

    ordered = month.sort_values(
        [
            "prediction",
            "ticker",
        ],
        ascending=[
            False,
            True,
        ],
    )

    selected: list[str] = []
    sector_counts: dict[str, int] = {}

    for row in ordered.itertuples(index=False):
        sector = str(row.sector)

        current_sector_count = sector_counts.get(
            sector,
            0,
        )

        if current_sector_count >= maximum_sector_names:
            continue

        selected.append(str(row.ticker))

        sector_counts[sector] = current_sector_count + 1

        if len(selected) == candidate_count:
            break

    if len(selected) != candidate_count:
        raise PortfolioConstructionError(
            "Could not select enough companies under the "
            "sector constraint. "
            f"Requested={candidate_count}, "
            f"selected={len(selected)}."
        )

    return selected


def _format_strategy_frame(
    month: pd.DataFrame,
    *,
    strategy_name: str,
    source_model: str,
    constraint_set: str,
    allocation_method: str,
    target_weights: pd.Series,
    allocation_scores: pd.Series,
    config: PortfolioConstructionConfig,
) -> pd.DataFrame:
    """Create the standardized target-weight table."""
    frame = (
        month.loc[
            :,
            [
                "as_of_date",
                "ticker",
                "sector",
                "prediction",
            ],
        ]
        .copy()
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    weight_by_ticker = pd.Series(
        target_weights.to_numpy(dtype=float),
        index=target_weights.index.astype(str),
    )

    score_by_ticker = pd.Series(
        allocation_scores.to_numpy(dtype=float),
        index=allocation_scores.index.astype(str),
    )

    frame["strategy_name"] = strategy_name
    frame["source_model"] = source_model
    frame["constraint_set"] = constraint_set
    frame["allocation_method"] = allocation_method

    frame["model_prediction"] = frame["prediction"].astype(float)

    frame["selection_rank"] = _calculate_selection_ranks(
        frame.rename(columns={"model_prediction": "prediction"})
        if "prediction" not in frame.columns
        else frame
    )

    frame["allocation_score"] = frame["ticker"].map(score_by_ticker).fillna(0.0).astype(float)

    frame["target_weight"] = frame["ticker"].map(weight_by_ticker).fillna(0.0).astype(float)

    frame["selected"] = frame["target_weight"].gt(config.weight_tolerance)

    return frame.loc[
        :,
        [
            "as_of_date",
            "strategy_name",
            "source_model",
            "constraint_set",
            "allocation_method",
            "ticker",
            "sector",
            "model_prediction",
            "selection_rank",
            "allocation_score",
            "target_weight",
            "selected",
        ],
    ]


def _build_universe_equal_weight(
    month: pd.DataFrame,
    *,
    config: PortfolioConstructionConfig,
) -> pd.DataFrame:
    """Build the all-universe equal-weight benchmark."""
    tickers = month["ticker"].astype(str).sort_values().tolist()

    equal_weight = 1.0 / len(tickers)

    if equal_weight > config.max_weight + config.weight_tolerance:
        raise PortfolioConstructionError("The universe benchmark violates max_weight.")

    weights = pd.Series(
        equal_weight,
        index=tickers,
        dtype=float,
    )

    scores = pd.Series(
        1.0,
        index=tickers,
        dtype=float,
    )

    output = _format_strategy_frame(
        month,
        strategy_name="universe_equal_weight",
        source_model="none",
        constraint_set="benchmark",
        allocation_method="equal_weight_all",
        target_weights=weights,
        allocation_scores=scores,
        config=config,
    )

    output["model_prediction"] = np.nan
    output["selection_rank"] = pd.Series(
        pd.NA,
        index=output.index,
        dtype="Int64",
    )

    return output


def _build_top_equal_weight(
    month: pd.DataFrame,
    *,
    model_name: str,
    config: PortfolioConstructionConfig,
) -> pd.DataFrame:
    """Build a constrained top-N equal-weight strategy."""
    selected = _select_ranked_candidates(
        month,
        candidate_count=config.top_n,
        config=config,
    )

    target_weight = 1.0 / config.top_n

    if target_weight > config.max_weight + config.weight_tolerance:
        raise PortfolioConstructionError("Top-N equal weighting violates max_weight.")

    weights = pd.Series(
        target_weight,
        index=selected,
        dtype=float,
    )

    scores = pd.Series(
        1.0,
        index=selected,
        dtype=float,
    )

    strategy_name = f"{model_name}_top{config.top_n}_equal_weight"

    return _format_strategy_frame(
        month,
        strategy_name=strategy_name,
        source_model=model_name,
        constraint_set="active_long_only",
        allocation_method="top_n_equal_weight",
        target_weights=weights,
        allocation_scores=scores,
        config=config,
    )


def _project_scores_to_constraints(
    candidate_data: pd.DataFrame,
    *,
    config: PortfolioConstructionConfig,
) -> tuple[pd.Series, pd.Series]:
    """Project positive model scores onto portfolio constraints."""
    candidates = candidate_data.sort_values("ticker").reset_index(drop=True).copy()

    prediction = candidates["prediction"].astype(float)

    shifted_score = prediction - float(prediction.min()) + config.optimization_tolerance

    if not np.isfinite(shifted_score.to_numpy(dtype=float)).all():
        raise PortfolioConstructionError("Allocation scores are not finite.")

    if float(shifted_score.sum()) <= 0.0:
        shifted_score = pd.Series(
            1.0,
            index=candidates.index,
            dtype=float,
        )

    normalized_target = (shifted_score / float(shifted_score.sum())).to_numpy(dtype=float)

    candidate_count = len(candidates)

    initial_weights = np.full(
        candidate_count,
        1.0 / candidate_count,
        dtype=float,
    )

    sector_indices = [
        group.index.to_numpy(dtype=int)
        for _, group in candidates.groupby(
            "sector",
            sort=True,
        )
    ]

    constraints: list[dict[str, Any]] = [
        {
            "type": "eq",
            "fun": lambda weights: float(np.sum(weights) - 1.0),
        }
    ]

    for indices in sector_indices:
        constraints.append(
            {
                "type": "ineq",
                "fun": (
                    lambda weights, selected_indices=indices: float(
                        config.max_sector_weight - np.sum(weights[selected_indices])
                    )
                ),
            }
        )

    def objective(
        weights: np.ndarray,
    ) -> float:
        difference = weights - normalized_target

        return float(
            0.5
            * np.dot(
                difference,
                difference,
            )
        )

    def gradient(
        weights: np.ndarray,
    ) -> np.ndarray:
        return weights - normalized_target

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        jac=gradient,
        bounds=[
            (
                0.0,
                config.max_weight,
            )
            for _ in range(candidate_count)
        ],
        constraints=constraints,
        options={
            "ftol": config.optimization_tolerance,
            "maxiter": (config.optimization_max_iterations),
            "disp": False,
        },
    )

    if not result.success:
        raise PortfolioConstructionError(
            f"Constrained score-weight optimization failed: {result.message}"
        )

    weights = np.asarray(
        result.x,
        dtype=float,
    )

    if abs(float(weights.sum()) - 1.0) > config.weight_tolerance:
        raise PortfolioConstructionError("Optimized weights do not sum to one.")

    if float(weights.min()) < -config.weight_tolerance:
        raise PortfolioConstructionError("Optimized weights contain negative values.")

    if float(weights.max()) > config.max_weight + config.weight_tolerance:
        raise PortfolioConstructionError("Optimized weights violate max_weight.")

    result_frame = candidates.loc[
        :,
        [
            "ticker",
            "sector",
        ],
    ].copy()

    result_frame["weight"] = weights

    sector_exposure = result_frame.groupby("sector")["weight"].sum()

    if float(sector_exposure.max()) > config.max_sector_weight + config.weight_tolerance:
        raise PortfolioConstructionError("Optimized weights violate max_sector_weight.")

    weight_series = pd.Series(
        weights,
        index=candidates["ticker"].astype(str),
        dtype=float,
    )

    score_series = pd.Series(
        shifted_score.to_numpy(dtype=float),
        index=candidates["ticker"].astype(str),
        dtype=float,
    )

    return weight_series, score_series


def _build_primary_score_weighted(
    month: pd.DataFrame,
    *,
    config: PortfolioConstructionConfig,
) -> pd.DataFrame:
    """Build the constrained score-weighted primary strategy."""
    candidate_tickers = _select_ranked_candidates(
        month,
        candidate_count=(config.score_weighted_candidate_count),
        config=config,
    )

    candidates = month.loc[month["ticker"].isin(candidate_tickers)].copy()

    weights, scores = _project_scores_to_constraints(
        candidates,
        config=config,
    )

    strategy_name = f"{config.primary_model_name}_score_weighted"

    return _format_strategy_frame(
        month,
        strategy_name=strategy_name,
        source_model=(config.primary_model_name),
        constraint_set="active_long_only",
        allocation_method=("positive_score_projection"),
        target_weights=weights,
        allocation_scores=scores,
        config=config,
    )


def calculate_portfolio_constraint_checks(
    target_weights: pd.DataFrame,
    *,
    config: PortfolioConstructionConfig,
) -> pd.DataFrame:
    """Check investment constraints for every strategy and date."""
    required_columns = (
        "as_of_date",
        "strategy_name",
        "constraint_set",
        "ticker",
        "sector",
        "target_weight",
    )

    _require_columns(
        target_weights,
        required_columns,
        dataset_name="Target portfolio weights",
    )

    duplicate_count = int(
        target_weights.duplicated(
            [
                "as_of_date",
                "strategy_name",
                "ticker",
            ]
        ).sum()
    )

    if duplicate_count:
        raise PortfolioConstructionError(
            "Target weights contain duplicated date-strategy-ticker rows."
        )

    rows: list[dict[str, Any]] = []

    grouped = target_weights.groupby(
        [
            "strategy_name",
            "as_of_date",
        ],
        sort=True,
    )

    for (
        strategy_name,
        as_of_date,
    ), group in grouped:
        weights = group["target_weight"].astype(float)

        sector_exposure = group.groupby("sector")["target_weight"].sum()

        weight_sum = float(weights.sum())

        minimum_weight = float(weights.min())

        maximum_weight = float(weights.max())

        maximum_sector_weight = float(sector_exposure.max())

        holdings = int(weights.gt(config.weight_tolerance).sum())

        top_ten_concentration = float(
            weights.nlargest(
                min(
                    10,
                    len(weights),
                )
            ).sum()
        )

        constraint_set = str(group["constraint_set"].iloc[0])

        sector_cap_required = constraint_set == "active_long_only"

        weight_sum_ok = abs(weight_sum - 1.0) <= config.weight_tolerance

        long_only_ok = minimum_weight >= -config.weight_tolerance

        asset_cap_ok = maximum_weight <= config.max_weight + config.weight_tolerance

        sector_cap_ok = maximum_sector_weight <= config.max_sector_weight + config.weight_tolerance

        constraints_pass = bool(
            weight_sum_ok
            and long_only_ok
            and asset_cap_ok
            and (sector_cap_ok or not sector_cap_required)
        )

        rows.append(
            {
                "strategy_name": strategy_name,
                "as_of_date": as_of_date,
                "constraint_set": constraint_set,
                "holdings": holdings,
                "weight_sum": weight_sum,
                "minimum_weight": minimum_weight,
                "maximum_weight": maximum_weight,
                "maximum_sector_weight": (maximum_sector_weight),
                "top_ten_concentration": (top_ten_concentration),
                "weight_sum_ok": weight_sum_ok,
                "long_only_ok": long_only_ok,
                "asset_cap_ok": asset_cap_ok,
                "sector_cap_required": (sector_cap_required),
                "sector_cap_ok": sector_cap_ok,
                "constraints_pass": constraints_pass,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "as_of_date",
                "strategy_name",
            ]
        )
        .reset_index(drop=True)
    )


def summarize_target_portfolios(
    constraint_checks: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize target portfolios through time."""
    rows: list[dict[str, Any]] = []

    for strategy_name, group in constraint_checks.groupby(
        "strategy_name",
        sort=True,
    ):
        rows.append(
            {
                "strategy_name": strategy_name,
                "months": len(group),
                "mean_holdings": float(group["holdings"].mean()),
                "minimum_holdings": int(group["holdings"].min()),
                "maximum_holdings": int(group["holdings"].max()),
                "mean_maximum_weight": float(group["maximum_weight"].mean()),
                "maximum_observed_weight": float(group["maximum_weight"].max()),
                "mean_maximum_sector_weight": float(group["maximum_sector_weight"].mean()),
                "maximum_observed_sector_weight": float(group["maximum_sector_weight"].max()),
                "mean_top_ten_concentration": float(group["top_ten_concentration"].mean()),
                "constraint_pass_ratio": float(group["constraints_pass"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("strategy_name").reset_index(drop=True)


def build_mvp_target_portfolios(
    predictions: pd.DataFrame,
    *,
    config: PortfolioConstructionConfig,
) -> MVPPortfolioConstructionOutputs:
    """Build all Step 8A target portfolios."""
    data = _validate_predictions(
        predictions,
        config=config,
    )

    required_models = (
        config.primary_model_name,
        config.challenger_model_name,
        config.momentum_model_name,
    )

    dates_by_model = {
        model_name: set(
            data.loc[
                data["model_name"].eq(model_name),
                "as_of_date",
            ]
        )
        for model_name in required_models
    }

    primary_dates = dates_by_model[config.primary_model_name]

    for model_name, model_dates in dates_by_model.items():
        if model_dates != primary_dates:
            raise PortfolioConstructionError(
                "Required models do not contain the same "
                "out-of-sample dates. "
                f"Misaligned model: {model_name}."
            )

    frames: list[pd.DataFrame] = []

    for as_of_date in sorted(primary_dates):
        primary = data.loc[
            data["model_name"].eq(config.primary_model_name) & data["as_of_date"].eq(as_of_date)
        ].copy()

        challenger = data.loc[
            data["model_name"].eq(config.challenger_model_name) & data["as_of_date"].eq(as_of_date)
        ].copy()

        momentum = data.loc[
            data["model_name"].eq(config.momentum_model_name) & data["as_of_date"].eq(as_of_date)
        ].copy()

        _validate_model_alignment(
            primary,
            challenger,
            other_model_name=(config.challenger_model_name),
            as_of_date=as_of_date,
        )

        _validate_model_alignment(
            primary,
            momentum,
            other_model_name=(config.momentum_model_name),
            as_of_date=as_of_date,
        )

        frames.extend(
            [
                _build_universe_equal_weight(
                    primary,
                    config=config,
                ),
                _build_top_equal_weight(
                    momentum,
                    model_name=(config.momentum_model_name),
                    config=config,
                ),
                _build_top_equal_weight(
                    primary,
                    model_name=(config.primary_model_name),
                    config=config,
                ),
                _build_top_equal_weight(
                    challenger,
                    model_name=(config.challenger_model_name),
                    config=config,
                ),
                _build_primary_score_weighted(
                    primary,
                    config=config,
                ),
            ]
        )

    target_weights = (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "as_of_date",
                "strategy_name",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    constraint_checks = calculate_portfolio_constraint_checks(
        target_weights,
        config=config,
    )

    failed_checks = constraint_checks.loc[~constraint_checks["constraints_pass"]]

    if not failed_checks.empty:
        first_failure = failed_checks.iloc[0]

        raise PortfolioConstructionError(
            "At least one target portfolio violates "
            "its constraints. First failure: "
            f"strategy={first_failure['strategy_name']}, "
            f"date={first_failure['as_of_date']}."
        )

    portfolio_summary = summarize_target_portfolios(constraint_checks)

    return MVPPortfolioConstructionOutputs(
        target_weights=target_weights,
        constraint_checks=constraint_checks,
        portfolio_summary=portfolio_summary,
    )
