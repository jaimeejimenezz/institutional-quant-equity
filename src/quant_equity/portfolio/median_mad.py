"""Median and mean absolute deviation portfolio construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import (
    NonlinearConstraint,
    differential_evolution,
    minimize,
)

from quant_equity.portfolio.optimizer import (
    PortfolioOptimizerConfig,
    _calculate_realized_turnover,
    _prepare_candidate_risk,
    _prepare_signal,
    _select_candidates,
)


class MedianMadPortfolioError(ValueError):
    """Raised when median-MAD portfolio construction cannot be completed."""


@dataclass(frozen=True)
class MedianMadConfig:
    """Configuration for median-MAD Differential Evolution portfolios."""

    lookback_days: int = 252
    minimum_observations: int = 126

    mad_limit: float = 0.008
    mad_violation_penalty: float = 10.0

    turnover_penalty: float = 0.001

    seed: int = 42

    max_iterations: int = 80
    population_size: int = 8
    tolerance: float = 1e-7

    mutation_min: float = 0.5
    mutation_max: float = 1.0
    recombination: float = 0.7

    polish: bool = False

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> MedianMadConfig:
        """Build median-MAD settings from project configuration."""
        config = cls(
            lookback_days=int(
                values.get(
                    "lookback_days",
                    252,
                )
            ),
            minimum_observations=int(
                values.get(
                    "minimum_observations",
                    126,
                )
            ),
            mad_limit=float(
                values.get(
                    "mad_limit",
                    0.008,
                )
            ),
            mad_violation_penalty=float(
                values.get(
                    "mad_violation_penalty",
                    10.0,
                )
            ),
            turnover_penalty=float(
                values.get(
                    "turnover_penalty",
                    0.001,
                )
            ),
            seed=int(
                values.get(
                    "seed",
                    42,
                )
            ),
            max_iterations=int(
                values.get(
                    "max_iterations",
                    80,
                )
            ),
            population_size=int(
                values.get(
                    "population_size",
                    8,
                )
            ),
            tolerance=float(
                values.get(
                    "tolerance",
                    1e-7,
                )
            ),
            mutation_min=float(
                values.get(
                    "mutation_min",
                    0.5,
                )
            ),
            mutation_max=float(
                values.get(
                    "mutation_max",
                    1.0,
                )
            ),
            recombination=float(
                values.get(
                    "recombination",
                    0.7,
                )
            ),
            polish=bool(
                values.get(
                    "polish",
                    False,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate median-MAD configuration."""
        if self.lookback_days < 1:
            raise MedianMadPortfolioError("lookback_days must be positive.")

        if self.minimum_observations < 1:
            raise MedianMadPortfolioError("minimum_observations must be positive.")

        if self.minimum_observations > self.lookback_days:
            raise MedianMadPortfolioError("minimum_observations cannot exceed lookback_days.")

        if self.mad_limit <= 0.0:
            raise MedianMadPortfolioError("mad_limit must be positive.")

        if self.mad_violation_penalty < 0.0:
            raise MedianMadPortfolioError("mad_violation_penalty cannot be negative.")

        if self.turnover_penalty < 0.0:
            raise MedianMadPortfolioError("turnover_penalty cannot be negative.")

        if self.max_iterations < 1:
            raise MedianMadPortfolioError("max_iterations must be positive.")

        if self.population_size < 1:
            raise MedianMadPortfolioError("population_size must be positive.")

        if self.tolerance <= 0.0:
            raise MedianMadPortfolioError("tolerance must be positive.")

        if not (0.0 < self.mutation_min <= self.mutation_max <= 2.0):
            raise MedianMadPortfolioError(
                "mutation bounds must satisfy 0 < minimum <= maximum <= 2."
            )

        if not 0.0 <= self.recombination <= 1.0:
            raise MedianMadPortfolioError("recombination must be between zero and one.")


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    """Require expected columns."""
    missing = sorted(set(columns).difference(data.columns))

    if missing:
        raise MedianMadPortfolioError(
            f"{dataset_name} is missing columns: " + ", ".join(missing) + "."
        )


def _build_daily_return_window(
    market_daily: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    tickers: list[str],
    config: MedianMadConfig,
) -> np.ndarray:
    """Build a past-only daily return matrix for selected candidates."""
    _require_columns(
        market_daily,
        (
            "date",
            "ticker",
            "adjusted_close",
        ),
        dataset_name="daily market data",
    )

    data = market_daily.loc[
        :,
        [
            "date",
            "ticker",
            "adjusted_close",
        ],
    ].copy()

    data["date"] = pd.to_datetime(data["date"]).dt.normalize()

    data["ticker"] = data["ticker"].astype(str)

    data["adjusted_close"] = pd.to_numeric(
        data["adjusted_close"],
        errors="coerce",
    )

    data = data.loc[data["date"].le(as_of_date) & data["ticker"].isin(tickers)]

    prices = (
        data.pivot(
            index="date",
            columns="ticker",
            values="adjusted_close",
        )
        .reindex(columns=tickers)
        .sort_index()
        .dropna()
    )

    daily_returns = prices.pct_change(fill_method=None).dropna()

    window = daily_returns.tail(config.lookback_days)

    if len(window) < config.minimum_observations:
        raise MedianMadPortfolioError(
            "Insufficient historical observations for "
            f"{as_of_date.date()}: "
            f"{len(window)} available."
        )

    values = window.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise MedianMadPortfolioError("Median-MAD return window contains non-finite values.")

    return values


def _project_bounded_simplex(
    values: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Project values onto a long-only simplex with individual upper bounds."""
    vector = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    limits = np.asarray(
        upper_bounds,
        dtype=float,
    ).reshape(-1)

    if vector.shape != limits.shape:
        raise MedianMadPortfolioError(
            "Projection values and upper bounds must have the same shape."
        )

    if not np.isfinite(vector).all() or not np.isfinite(limits).all():
        raise MedianMadPortfolioError("Projection inputs must contain only finite values.")

    if (limits <= 0.0).any():
        raise MedianMadPortfolioError("All portfolio upper bounds must be positive.")

    if limits.sum() < 1.0 - 1e-12:
        raise MedianMadPortfolioError(
            "Asset and liquidity limits cannot support a fully invested portfolio."
        )

    lower_theta = float(np.min(vector - limits))

    upper_theta = float(np.max(vector))

    for _ in range(120):
        theta = (lower_theta + upper_theta) / 2.0

        projected = np.clip(
            vector - theta,
            0.0,
            limits,
        )

        if projected.sum() > 1.0:
            lower_theta = theta
        else:
            upper_theta = theta

    projected = np.clip(
        vector - upper_theta,
        0.0,
        limits,
    )

    residual = 1.0 - float(projected.sum())

    if residual > 1e-12:
        capacity = limits - projected

        for index in np.where(capacity > 1e-14)[0]:
            addition = min(
                residual,
                float(capacity[index]),
            )

            projected[index] += addition

            residual -= addition

            if residual <= 1e-12:
                break

    elif residual < -1e-12:
        excess = -residual

        for index in np.where(projected > 1e-14)[0]:
            removal = min(
                excess,
                float(projected[index]),
            )

            projected[index] -= removal

            excess -= removal

            if excess <= 1e-12:
                break

    if abs(float(projected.sum()) - 1.0) > 1e-9:
        raise MedianMadPortfolioError(
            "Bounded-simplex projection did not produce a fully invested portfolio."
        )

    return projected


def _portfolio_statistics(
    return_window: np.ndarray,
    weights: np.ndarray,
) -> tuple[
    float,
    float,
]:
    """Calculate portfolio median return and MAD around the median."""
    portfolio_returns = return_window @ weights

    median_return = float(np.median(portfolio_returns))

    mad = float(np.mean(np.abs(portfolio_returns - median_return)))

    return (
        median_return,
        mad,
    )


def _turnover_l1(
    weights: np.ndarray,
    tickers: list[str],
    previous_weights: pd.Series | None,
) -> float:
    """Calculate L1 turnover across current and dropped holdings."""
    if previous_weights is None:
        return 0.0

    previous_current = previous_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)

    dropped_weight = float(previous_weights.loc[~previous_weights.index.isin(tickers)].sum())

    return float(np.abs(weights - previous_current).sum() + dropped_weight)


def _is_feasible(
    weights: np.ndarray,
    upper_bounds: np.ndarray,
    beta: np.ndarray,
    sector_indices: list[np.ndarray],
    *,
    sector_limit: float,
    portfolio_config: PortfolioOptimizerConfig,
    tolerance: float,
) -> bool:
    """Check hard portfolio constraints."""
    if abs(float(weights.sum()) - 1.0) > tolerance:
        return False

    if (weights < -tolerance).any():
        return False

    if (weights > upper_bounds + tolerance).any():
        return False

    portfolio_beta = float(beta @ weights)

    if portfolio_beta < portfolio_config.minimum_portfolio_beta - tolerance:
        return False

    if portfolio_beta > portfolio_config.maximum_portfolio_beta + tolerance:
        return False

    return all(
        float(weights[indices].sum()) <= sector_limit + tolerance for indices in sector_indices
    )


def _find_feasible_reference(
    candidates: pd.DataFrame,
    beta: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    previous_weights: pd.Series | None,
    portfolio_config: PortfolioOptimizerConfig,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    float,
]:
    """Find a feasible portfolio used to initialize Differential Evolution."""
    tickers = candidates["ticker"].astype(str).tolist()

    sectors = candidates["sector"].astype(str).to_numpy()

    sector_indices = [np.where(sectors == sector)[0] for sector in sorted(set(sectors))]

    sector_limit = portfolio_config.max_sector_weight - portfolio_config.constraint_margin

    if previous_weights is None:
        target = np.full(
            len(candidates),
            1.0 / len(candidates),
            dtype=float,
        )
    else:
        target = previous_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)

    initial = _project_bounded_simplex(
        target,
        upper_bounds,
    )

    feasibility_tolerance = max(
        portfolio_config.weight_tolerance,
        1e-7,
    )

    if _is_feasible(
        initial,
        upper_bounds,
        beta,
        sector_indices,
        sector_limit=sector_limit,
        portfolio_config=portfolio_config,
        tolerance=feasibility_tolerance,
    ):
        return (
            initial,
            sector_indices,
            sector_limit,
        )

    constraints: list[dict[str, Any]] = [
        {
            "type": "eq",
            "fun": (lambda weights: float(weights.sum() - 1.0)),
        },
        {
            "type": "ineq",
            "fun": (
                lambda weights: float(beta @ weights - portfolio_config.minimum_portfolio_beta)
            ),
        },
        {
            "type": "ineq",
            "fun": (
                lambda weights: float(portfolio_config.maximum_portfolio_beta - beta @ weights)
            ),
        },
    ]

    for indices in sector_indices:
        constraints.append(
            {
                "type": "ineq",
                "fun": (
                    lambda weights, indices=indices: float(sector_limit - weights[indices].sum())
                ),
            }
        )

    result = minimize(
        fun=(lambda weights: float(np.square(weights - target).sum())),
        x0=initial,
        method="SLSQP",
        bounds=[
            (
                0.0,
                float(limit),
            )
            for limit in upper_bounds
        ],
        constraints=constraints,
        options={
            "ftol": max(
                portfolio_config.solver_tolerance,
                1e-12,
            ),
            "maxiter": 2_000,
        },
    )

    feasible_weights = np.asarray(
        result.x,
        dtype=float,
    ).reshape(-1)

    if not _is_feasible(
        feasible_weights,
        upper_bounds,
        beta,
        sector_indices,
        sector_limit=sector_limit,
        portfolio_config=portfolio_config,
        tolerance=feasibility_tolerance,
    ):
        raise MedianMadPortfolioError(
            "No feasible median-MAD portfolio satisfies "
            "security, sector, beta and liquidity constraints."
        )

    return (
        feasible_weights,
        sector_indices,
        sector_limit,
    )


def _solve_median_mad_portfolio(
    candidates: pd.DataFrame,
    return_window: np.ndarray,
    candidate_risk: pd.DataFrame,
    *,
    previous_weights: pd.Series | None,
    portfolio_config: PortfolioOptimizerConfig,
    median_mad_config: MedianMadConfig,
    seed: int,
) -> tuple[
    np.ndarray,
    dict[str, Any],
]:
    """Solve one median-MAD-turnover portfolio with Differential Evolution."""
    tickers = candidates["ticker"].astype(str).tolist()

    beta = candidate_risk["beta_vs_spy"].to_numpy(dtype=float)

    average_dollar_volume = candidate_risk["average_dollar_volume"].to_numpy(dtype=float)

    effective_security_limit = (
        portfolio_config.max_security_weight - portfolio_config.constraint_margin
    )

    liquidity_weight_limits = (
        portfolio_config.max_position_adv_fraction
        * average_dollar_volume
        / portfolio_config.reference_portfolio_value
    )

    upper_bounds = np.minimum(
        effective_security_limit,
        liquidity_weight_limits,
    )

    if upper_bounds.sum() < 1.0 - portfolio_config.weight_tolerance:
        raise MedianMadPortfolioError(
            "Security and liquidity constraints cannot support "
            "a fully invested median-MAD portfolio."
        )

    (
        feasible_reference,
        sector_indices,
        sector_limit,
    ) = _find_feasible_reference(
        candidates,
        beta,
        upper_bounds,
        previous_weights=previous_weights,
        portfolio_config=portfolio_config,
    )

    def objective(
        raw_weights: np.ndarray,
    ) -> float:
        weights = _project_bounded_simplex(
            raw_weights,
            upper_bounds,
        )

        (
            median_return,
            mad,
        ) = _portfolio_statistics(
            return_window,
            weights,
        )

        mad_violation = max(
            0.0,
            mad - median_mad_config.mad_limit,
        )

        turnover = _turnover_l1(
            weights,
            tickers,
            previous_weights,
        )

        return float(
            -median_return
            + (median_mad_config.mad_violation_penalty * mad_violation)
            + (median_mad_config.turnover_penalty * turnover)
        )

    def constraint_values(
        raw_weights: np.ndarray,
    ) -> np.ndarray:
        weights = _project_bounded_simplex(
            raw_weights,
            upper_bounds,
        )

        sector_weights = [float(weights[indices].sum()) for indices in sector_indices]

        return np.asarray(
            [
                *sector_weights,
                float(beta @ weights),
            ],
            dtype=float,
        )

    hard_constraints = NonlinearConstraint(
        constraint_values,
        lb=np.asarray(
            [
                *([0.0] * len(sector_indices)),
                portfolio_config.minimum_portfolio_beta,
            ],
            dtype=float,
        ),
        ub=np.asarray(
            [
                *([sector_limit] * len(sector_indices)),
                portfolio_config.maximum_portfolio_beta,
            ],
            dtype=float,
        ),
    )

    result = differential_evolution(
        func=objective,
        bounds=[
            (
                0.0,
                1.0,
            )
        ]
        * len(candidates),
        strategy="best1bin",
        maxiter=(median_mad_config.max_iterations),
        popsize=(median_mad_config.population_size),
        tol=(median_mad_config.tolerance),
        mutation=(
            median_mad_config.mutation_min,
            median_mad_config.mutation_max,
        ),
        recombination=(median_mad_config.recombination),
        seed=seed,
        polish=(median_mad_config.polish),
        updating="immediate",
        workers=1,
        constraints=(hard_constraints,),
        x0=feasible_reference,
    )

    solved_weights = _project_bounded_simplex(
        result.x,
        upper_bounds,
    )

    feasibility_tolerance = max(
        portfolio_config.weight_tolerance,
        1e-6,
    )

    if not _is_feasible(
        solved_weights,
        upper_bounds,
        beta,
        sector_indices,
        sector_limit=sector_limit,
        portfolio_config=portfolio_config,
        tolerance=feasibility_tolerance,
    ):
        raise MedianMadPortfolioError(
            "Differential Evolution returned a portfolio that violates hard constraints."
        )

    (
        median_return,
        mad,
    ) = _portfolio_statistics(
        return_window,
        solved_weights,
    )

    mad_violation = max(
        0.0,
        mad - median_mad_config.mad_limit,
    )

    turnover_l1 = _turnover_l1(
        solved_weights,
        tickers,
        previous_weights,
    )

    current_weights = pd.Series(
        solved_weights,
        index=tickers,
        dtype=float,
    )

    one_way_turnover = _calculate_realized_turnover(
        previous_weights,
        current_weights,
    )

    portfolio_beta = float(beta @ solved_weights)

    position_adv_fractions = (
        portfolio_config.reference_portfolio_value * solved_weights / average_dollar_volume
    )

    sector_weights = (
        candidates.assign(solved_weight=solved_weights).groupby("sector")["solved_weight"].sum()
    )

    objective_value = float(
        -median_return
        + (median_mad_config.mad_violation_penalty * mad_violation)
        + (median_mad_config.turnover_penalty * turnover_l1)
    )

    diagnostics = {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "objective_value": (objective_value),
        "function_evaluations": int(result.nfev),
        "iterations": int(result.nit),
        "seed": int(seed),
        "observation_count": int(len(return_window)),
        "median_daily_return": (median_return),
        "mad_daily": (mad),
        "mad_limit": float(median_mad_config.mad_limit),
        "mad_violation": (mad_violation),
        "mad_penalty_value": float(median_mad_config.mad_violation_penalty * mad_violation),
        "turnover_l1": (turnover_l1),
        "one_way_turnover": (one_way_turnover),
        "turnover_penalty": float(median_mad_config.turnover_penalty),
        "turnover_penalty_value": float(median_mad_config.turnover_penalty * turnover_l1),
        "portfolio_beta_vs_spy": (portfolio_beta),
        "maximum_position_adv_fraction": float(position_adv_fractions.max()),
        "positions": int((solved_weights > portfolio_config.weight_tolerance).sum()),
        "weight_sum": float(solved_weights.sum()),
        "maximum_weight": float(solved_weights.max()),
        "maximum_sector_weight": float(sector_weights.max()),
    }

    return (
        solved_weights,
        diagnostics,
    )


def build_median_mad_portfolios(
    final_signal: pd.DataFrame,
    market_daily: pd.DataFrame,
    risk_estimates: pd.DataFrame,
    *,
    portfolio_config: PortfolioOptimizerConfig,
    median_mad_config: MedianMadConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build historical median-MAD portfolios through time."""
    portfolio_config.validate()
    median_mad_config.validate()

    signal = _prepare_signal(final_signal)

    portfolio_blocks = []
    diagnostic_rows = []

    previous_weights: pd.Series | None = None

    random_generator = np.random.default_rng(median_mad_config.seed)

    for (
        as_of_date,
        date_signal,
    ) in signal.groupby(
        "as_of_date",
        sort=True,
    ):
        normalized_date = pd.Timestamp(as_of_date).normalize()

        candidates = _select_candidates(
            date_signal,
            config=portfolio_config,
        )

        tickers = candidates["ticker"].astype(str).tolist()

        return_window = _build_daily_return_window(
            market_daily,
            as_of_date=normalized_date,
            tickers=tickers,
            config=median_mad_config,
        )

        candidate_risk = _prepare_candidate_risk(
            risk_estimates,
            as_of_date=normalized_date,
            tickers=tickers,
        )

        optimization_seed = int(
            random_generator.integers(
                0,
                1_000_000_000,
            )
        )

        (
            solved_weights,
            diagnostics,
        ) = _solve_median_mad_portfolio(
            candidates,
            return_window,
            candidate_risk,
            previous_weights=previous_weights,
            portfolio_config=portfolio_config,
            median_mad_config=median_mad_config,
            seed=optimization_seed,
        )

        if previous_weights is None:
            previous_candidate_weights = np.zeros(
                len(candidates),
                dtype=float,
            )
        else:
            previous_candidate_weights = (
                previous_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
            )

        candidates = candidates.copy()

        candidates["previous_weight"] = previous_candidate_weights

        candidates["weight"] = solved_weights

        candidates["method"] = "median_mad_de"

        candidates["beta_vs_spy"] = candidate_risk["beta_vs_spy"].to_numpy(dtype=float)

        candidates["average_dollar_volume"] = candidate_risk["average_dollar_volume"].to_numpy(
            dtype=float
        )

        candidates["position_adv_fraction"] = (
            portfolio_config.reference_portfolio_value
            * solved_weights
            / candidates["average_dollar_volume"].to_numpy(dtype=float)
        )

        portfolio_blocks.append(
            candidates.loc[
                :,
                [
                    "as_of_date",
                    "ticker",
                    "sector",
                    "method",
                    "rank",
                    "percentile_score",
                    "previous_weight",
                    "beta_vs_spy",
                    "average_dollar_volume",
                    "position_adv_fraction",
                    "weight",
                ],
            ]
        )

        diagnostic_rows.append(
            {
                "as_of_date": (normalized_date),
                **diagnostics,
            }
        )

        previous_weights = pd.Series(
            solved_weights,
            index=tickers,
            dtype=float,
        )

    weights = (
        pd.concat(
            portfolio_blocks,
            ignore_index=True,
        )
        .sort_values(
            [
                "as_of_date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    diagnostics = pd.DataFrame(diagnostic_rows).sort_values("as_of_date").reset_index(drop=True)

    return (
        weights,
        diagnostics,
    )


def validate_median_mad_diagnostics(
    diagnostics: pd.DataFrame,
    *,
    portfolio_config: PortfolioOptimizerConfig,
    median_mad_config: MedianMadConfig,
) -> pd.DataFrame:
    """Audit median-MAD portfolio diagnostics."""
    checks = [
        (
            "finite_objective",
            int((~np.isfinite(diagnostics["objective_value"].to_numpy(dtype=float))).sum()),
            "Median-MAD objective values must be finite.",
        ),
        (
            "finite_median",
            int((~np.isfinite(diagnostics["median_daily_return"].to_numpy(dtype=float))).sum()),
            "Median return estimates must be finite.",
        ),
        (
            "finite_mad",
            int((~np.isfinite(diagnostics["mad_daily"].to_numpy(dtype=float))).sum()),
            "MAD estimates must be finite.",
        ),
        (
            "minimum_observations",
            int(diagnostics["observation_count"].lt(median_mad_config.minimum_observations).sum()),
            "Every optimization must use enough historical observations.",
        ),
        (
            "fully_invested",
            int(
                diagnostics["weight_sum"]
                .sub(1.0)
                .abs()
                .gt(
                    max(
                        portfolio_config.weight_tolerance,
                        1e-6,
                    )
                )
                .sum()
            ),
            "Median-MAD portfolios must remain fully invested.",
        ),
        (
            "security_weight_limit",
            int(
                diagnostics["maximum_weight"]
                .gt(portfolio_config.max_security_weight + portfolio_config.weight_tolerance)
                .sum()
            ),
            "Median-MAD positions must respect the security weight limit.",
        ),
        (
            "sector_weight_limit",
            int(
                diagnostics["maximum_sector_weight"]
                .gt(portfolio_config.max_sector_weight + portfolio_config.weight_tolerance)
                .sum()
            ),
            "Median-MAD sector weights must respect the sector limit.",
        ),
        (
            "portfolio_beta_lower_limit",
            int(
                diagnostics["portfolio_beta_vs_spy"]
                .lt(portfolio_config.minimum_portfolio_beta - portfolio_config.weight_tolerance)
                .sum()
            ),
            "Median-MAD portfolio beta must remain above its lower bound.",
        ),
        (
            "portfolio_beta_upper_limit",
            int(
                diagnostics["portfolio_beta_vs_spy"]
                .gt(portfolio_config.maximum_portfolio_beta + portfolio_config.weight_tolerance)
                .sum()
            ),
            "Median-MAD portfolio beta must remain below its upper bound.",
        ),
        (
            "position_liquidity_limit",
            int(
                diagnostics["maximum_position_adv_fraction"]
                .gt(portfolio_config.max_position_adv_fraction + portfolio_config.weight_tolerance)
                .sum()
            ),
            "Median-MAD positions must respect the liquidity limit.",
        ),
        (
            "nonnegative_turnover",
            int(diagnostics["turnover_l1"].lt(-portfolio_config.weight_tolerance).sum()),
            "Median-MAD turnover must be non-negative.",
        ),
    ]

    return pd.DataFrame(
        [
            {
                "check": name,
                "status": ("PASS" if violations == 0 else "FAIL"),
                "violations": violations,
                "description": description,
            }
            for (
                name,
                violations,
                description,
            ) in checks
        ]
    )
