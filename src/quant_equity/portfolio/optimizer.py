"""Risk-aware portfolio optimization with turnover control."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd


class PortfolioOptimizationError(ValueError):
    """Raised when target portfolio optimization cannot be completed."""


@dataclass(frozen=True)
class PortfolioOptimizerConfig:
    """Configuration for alpha-risk-turnover portfolio optimization."""

    candidate_count: int = 25
    annualized_alpha_scale: float = 0.10
    risk_aversion: float = 0.50
    turnover_penalty: float = 0.01
    max_security_weight: float = 0.05
    max_sector_weight: float = 0.25
    weight_tolerance: float = 1e-8
    solver_tolerance: float = 1e-9
    constraint_margin: float = 1e-6
    minimum_portfolio_beta: float = 0.85
    maximum_portfolio_beta: float = 1.15
    reference_portfolio_value: float = 1_000_000.0
    max_position_adv_fraction: float = 0.01

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> PortfolioOptimizerConfig:
        """Build optimizer configuration from project settings."""
        config = cls(
            candidate_count=int(
                values.get(
                    "candidate_count",
                    25,
                )
            ),
            annualized_alpha_scale=float(
                values.get(
                    "annualized_alpha_scale",
                    0.10,
                )
            ),
            risk_aversion=float(
                values.get(
                    "risk_aversion",
                    0.50,
                )
            ),
            turnover_penalty=float(
                values.get(
                    "turnover_penalty",
                    0.01,
                )
            ),
            max_security_weight=float(
                values.get(
                    "max_security_weight",
                    0.05,
                )
            ),
            max_sector_weight=float(
                values.get(
                    "max_sector_weight",
                    0.25,
                )
            ),
            weight_tolerance=float(
                values.get(
                    "weight_tolerance",
                    1e-8,
                )
            ),
            solver_tolerance=float(
                values.get(
                    "solver_tolerance",
                    1e-9,
                )
            ),
            constraint_margin=float(
                values.get(
                    "constraint_margin",
                    1e-6,
                )
            ),
            minimum_portfolio_beta=float(
                values.get(
                    "minimum_portfolio_beta",
                    0.85,
                )
            ),
            maximum_portfolio_beta=float(
                values.get(
                    "maximum_portfolio_beta",
                    1.15,
                )
            ),
            reference_portfolio_value=float(
                values.get(
                    "reference_portfolio_value",
                    1_000_000.0,
                )
            ),
            max_position_adv_fraction=float(
                values.get(
                    "max_position_adv_fraction",
                    0.01,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate optimizer settings."""
        if self.candidate_count < 1:
            raise PortfolioOptimizationError("candidate_count must be positive.")

        if self.annualized_alpha_scale <= 0.0:
            raise PortfolioOptimizationError("annualized_alpha_scale must be positive.")

        if self.risk_aversion < 0.0:
            raise PortfolioOptimizationError("risk_aversion cannot be negative.")

        if self.turnover_penalty < 0.0:
            raise PortfolioOptimizationError("turnover_penalty cannot be negative.")

        if not 0.0 < self.max_security_weight <= 1.0:
            raise PortfolioOptimizationError("max_security_weight must be in (0, 1].")

        if not 0.0 < self.max_sector_weight <= 1.0:
            raise PortfolioOptimizationError("max_sector_weight must be in (0, 1].")

        if self.candidate_count * self.max_security_weight < 1.0 - self.weight_tolerance:
            raise PortfolioOptimizationError(
                "Candidate capacity cannot support a fully invested portfolio."
            )

        if self.weight_tolerance <= 0.0:
            raise PortfolioOptimizationError("weight_tolerance must be positive.")

        if self.solver_tolerance <= 0.0:
            raise PortfolioOptimizationError("solver_tolerance must be positive.")

        if self.constraint_margin < 0.0:
            raise PortfolioOptimizationError("constraint_margin cannot be negative.")

        if self.constraint_margin >= self.max_security_weight:
            raise PortfolioOptimizationError(
                "constraint_margin must be smaller than max_security_weight."
            )

        if self.constraint_margin >= self.max_sector_weight:
            raise PortfolioOptimizationError(
                "constraint_margin must be smaller than max_sector_weight."
            )

        if self.minimum_portfolio_beta > self.maximum_portfolio_beta:
            raise PortfolioOptimizationError(
                "minimum_portfolio_beta cannot exceed maximum_portfolio_beta."
            )

        if self.reference_portfolio_value <= 0.0:
            raise PortfolioOptimizationError("reference_portfolio_value must be positive.")

        if not (0.0 < self.max_position_adv_fraction <= 1.0):
            raise PortfolioOptimizationError("max_position_adv_fraction must be in (0, 1].")


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    """Require expected columns."""
    missing = sorted(set(columns).difference(data.columns))

    if missing:
        raise PortfolioOptimizationError(
            f"{dataset_name} is missing columns: " + ", ".join(missing) + "."
        )


def _prepare_signal(
    final_signal: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize the final alpha signal."""
    _require_columns(
        final_signal,
        (
            "as_of_date",
            "ticker",
            "sector",
            "rank",
            "percentile_score",
        ),
        dataset_name="final alpha signal",
    )

    signal = final_signal.loc[
        :,
        [
            "as_of_date",
            "ticker",
            "sector",
            "rank",
            "percentile_score",
        ],
    ].copy()

    signal["as_of_date"] = pd.to_datetime(signal["as_of_date"]).dt.normalize()

    signal["ticker"] = signal["ticker"].astype(str)

    signal["sector"] = signal["sector"].astype(str)

    signal["rank"] = pd.to_numeric(
        signal["rank"],
        errors="coerce",
    )

    signal["percentile_score"] = pd.to_numeric(
        signal["percentile_score"],
        errors="coerce",
    )

    if signal.duplicated(
        [
            "as_of_date",
            "ticker",
        ]
    ).any():
        raise PortfolioOptimizationError("Final alpha signal contains duplicate keys.")

    if (
        signal[
            [
                "rank",
                "percentile_score",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise PortfolioOptimizationError("Final alpha signal contains invalid ranking values.")

    return signal


def _select_candidates(
    date_signal: pd.DataFrame,
    *,
    config: PortfolioOptimizerConfig,
) -> pd.DataFrame:
    """Select a diversified candidate set from the alpha ranking."""
    equal_weight = 1.0 / float(config.candidate_count)

    maximum_names_per_sector = int(np.floor((config.max_sector_weight + 1e-12) / equal_weight))

    if maximum_names_per_sector < 1:
        raise PortfolioOptimizationError("Sector constraint cannot support candidate selection.")

    ordered = date_signal.sort_values(
        [
            "rank",
            "ticker",
        ]
    )

    selected_indices: list[int] = []
    sector_counts: dict[str, int] = {}

    for index, row in ordered.iterrows():
        sector = str(row["sector"])

        count = sector_counts.get(
            sector,
            0,
        )

        if count >= maximum_names_per_sector:
            continue

        selected_indices.append(index)

        sector_counts[sector] = count + 1

        if len(selected_indices) == config.candidate_count:
            break

    if len(selected_indices) != config.candidate_count:
        raise PortfolioOptimizationError(
            "Unable to select enough candidates under the sector diversification rule."
        )

    return (
        date_signal.loc[selected_indices]
        .sort_values(
            [
                "rank",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )


def _resolve_covariance_schema(
    covariance: pd.DataFrame,
) -> tuple[str, str, str]:
    """Resolve the stored long-format covariance schema."""
    ticker_pairs = (
        (
            "ticker_a",
            "ticker_b",
        ),
        (
            "ticker_i",
            "ticker_j",
        ),
        (
            "row_ticker",
            "column_ticker",
        ),
        (
            "ticker_x",
            "ticker_y",
        ),
        (
            "asset_i",
            "asset_j",
        ),
        (
            "ticker",
            "other_ticker",
        ),
    )

    value_columns = (
        "covariance",
        "annualized_covariance",
        "covariance_value",
    )

    ticker_columns = None

    for first, second in ticker_pairs:
        if first in covariance.columns and second in covariance.columns:
            ticker_columns = (
                first,
                second,
            )
            break

    if ticker_columns is None:
        raise PortfolioOptimizationError(
            "Unable to identify covariance ticker columns. "
            f"Available columns: {list(covariance.columns)}"
        )

    covariance_column = next(
        (column for column in value_columns if column in covariance.columns),
        None,
    )

    if covariance_column is None:
        raise PortfolioOptimizationError(
            "Unable to identify covariance value column. "
            f"Available columns: {list(covariance.columns)}"
        )

    return (
        ticker_columns[0],
        ticker_columns[1],
        covariance_column,
    )


def _resolve_risk_schema(
    risk_estimates: pd.DataFrame,
) -> tuple[str, str]:
    """Resolve beta and liquidity columns from stored risk estimates."""
    beta_columns = (
        "beta_vs_spy",
        "rolling_beta_vs_spy",
        "market_beta",
        "beta",
    )

    adv_columns = (
        "average_dollar_volume",
        "average_dollar_volume_20d",
        "adv_20d",
        "dollar_volume_20d",
        "adv",
    )

    beta_column = next(
        (column for column in beta_columns if column in risk_estimates.columns),
        None,
    )

    if beta_column is None:
        raise PortfolioOptimizationError(
            "Unable to identify beta column in risk estimates. "
            f"Available columns: {list(risk_estimates.columns)}"
        )

    adv_column = next(
        (column for column in adv_columns if column in risk_estimates.columns),
        None,
    )

    if adv_column is None:
        raise PortfolioOptimizationError(
            "Unable to identify Average Dollar Volume column "
            "in risk estimates. "
            f"Available columns: {list(risk_estimates.columns)}"
        )

    return (
        beta_column,
        adv_column,
    )


def _prepare_candidate_risk(
    risk_estimates: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    tickers: list[str],
) -> pd.DataFrame:
    """Load beta and liquidity estimates for portfolio candidates."""
    _require_columns(
        risk_estimates,
        (
            "as_of_date",
            "ticker",
        ),
        dataset_name="risk estimates",
    )

    (
        beta_column,
        adv_column,
    ) = _resolve_risk_schema(risk_estimates)

    risk = risk_estimates.loc[
        :,
        [
            "as_of_date",
            "ticker",
            beta_column,
            adv_column,
        ],
    ].copy()

    risk["as_of_date"] = pd.to_datetime(risk["as_of_date"]).dt.normalize()

    risk["ticker"] = risk["ticker"].astype(str)

    risk = risk.loc[risk["as_of_date"].eq(as_of_date) & risk["ticker"].isin(tickers)].copy()

    if risk.duplicated("ticker").any():
        raise PortfolioOptimizationError(
            f"Risk estimates contain duplicate tickers for {as_of_date.date()}."
        )

    risk = risk.set_index("ticker").reindex(tickers)

    if (
        risk[
            [
                beta_column,
                adv_column,
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise PortfolioOptimizationError(f"Risk estimates are incomplete for {as_of_date.date()}.")

    risk[beta_column] = pd.to_numeric(
        risk[beta_column],
        errors="coerce",
    )

    risk[adv_column] = pd.to_numeric(
        risk[adv_column],
        errors="coerce",
    )

    if (~np.isfinite(risk[beta_column].to_numpy(dtype=float))).any():
        raise PortfolioOptimizationError("Candidate beta estimates must be finite.")

    if (~np.isfinite(risk[adv_column].to_numpy(dtype=float))).any():
        raise PortfolioOptimizationError("Candidate liquidity estimates must be finite.")

    if (risk[adv_column] <= 0.0).any():
        raise PortfolioOptimizationError("Average Dollar Volume must be positive.")

    return pd.DataFrame(
        {
            "beta_vs_spy": (risk[beta_column].to_numpy(dtype=float)),
            "average_dollar_volume": (risk[adv_column].to_numpy(dtype=float)),
        },
        index=tickers,
    )


def _build_covariance_matrix(
    covariance: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    tickers: list[str],
) -> np.ndarray:
    """Build an ordered covariance matrix for one portfolio date."""
    _require_columns(
        covariance,
        ("as_of_date",),
        dataset_name="covariance estimates",
    )

    (
        first_ticker_column,
        second_ticker_column,
        covariance_column,
    ) = _resolve_covariance_schema(covariance)

    covariance_data = covariance.copy()

    covariance_data["as_of_date"] = pd.to_datetime(covariance_data["as_of_date"]).dt.normalize()

    date_covariance = covariance_data.loc[covariance_data["as_of_date"].eq(as_of_date)]

    if date_covariance.empty:
        raise PortfolioOptimizationError(
            f"No covariance matrix is available for {as_of_date.date()}."
        )

    matrix = date_covariance.pivot(
        index=first_ticker_column,
        columns=second_ticker_column,
        values=covariance_column,
    ).reindex(
        index=tickers,
        columns=tickers,
    )

    if matrix.isna().any().any():
        missing_pairs = int(matrix.isna().sum().sum())

        raise PortfolioOptimizationError(
            "Covariance matrix is incomplete for "
            f"{as_of_date.date()}: "
            f"{missing_pairs} missing cells."
        )

    values = matrix.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise PortfolioOptimizationError("Covariance matrix contains non-finite values.")

    values = 0.5 * (values + values.T)

    minimum_eigenvalue = float(np.linalg.eigvalsh(values).min())

    if minimum_eigenvalue < -1e-8:
        raise PortfolioOptimizationError("Covariance matrix is not positive semidefinite.")

    if minimum_eigenvalue < 0.0:
        values = values + np.eye(len(values)) * (-minimum_eigenvalue + 1e-10)

    return values


def _calculate_realized_turnover(
    previous_weights: pd.Series | None,
    current_weights: pd.Series,
) -> float:
    """Calculate one-way turnover across the union of holdings."""
    if previous_weights is None:
        return float("nan")

    aligned = pd.concat(
        [
            previous_weights.rename("previous"),
            current_weights.rename("current"),
        ],
        axis=1,
    ).fillna(0.0)

    return float(0.5 * (aligned["current"] - aligned["previous"]).abs().sum())


def _solve_portfolio(
    candidates: pd.DataFrame,
    covariance_matrix: np.ndarray,
    candidate_risk: pd.DataFrame,
    *,
    previous_weights: pd.Series | None,
    config: PortfolioOptimizerConfig,
) -> tuple[
    np.ndarray,
    dict[str, float | str],
]:
    """Solve one alpha-risk-turnover portfolio."""
    tickers = candidates["ticker"].astype(str).tolist()

    alpha = (
        candidates["percentile_score"].to_numpy(dtype=float) - 0.5
    ) * config.annualized_alpha_scale

    effective_security_limit = config.max_security_weight - config.constraint_margin

    effective_sector_limit = config.max_sector_weight - config.constraint_margin

    beta = candidate_risk["beta_vs_spy"].to_numpy(dtype=float)

    average_dollar_volume = candidate_risk["average_dollar_volume"].to_numpy(dtype=float)

    liquidity_weight_limits = (
        config.max_position_adv_fraction * average_dollar_volume / config.reference_portfolio_value
    )

    maximum_feasible_weights = np.minimum(
        effective_security_limit,
        liquidity_weight_limits,
    )

    if maximum_feasible_weights.sum() < 1.0 - config.weight_tolerance:
        raise PortfolioOptimizationError(
            "Security and liquidity constraints cannot support a fully invested portfolio."
        )

    weights = cp.Variable(len(candidates))

    constraints = [
        weights >= 0.0,
        weights <= effective_security_limit,
        cp.sum(weights) == 1.0,
    ]

    constraints.extend(
        [
            beta @ weights >= config.minimum_portfolio_beta,
            beta @ weights <= config.maximum_portfolio_beta,
            weights <= liquidity_weight_limits,
        ]
    )

    sectors = candidates["sector"].astype(str).to_numpy()

    for sector in sorted(set(sectors)):
        indices = np.where(sectors == sector)[0]

        constraints.append(cp.sum(weights[indices]) <= effective_sector_limit)

    predicted_alpha = alpha @ weights

    predicted_variance = cp.quad_form(
        weights,
        covariance_matrix,
    )

    if previous_weights is None:
        turnover_expression = 0.0
        previous_current = np.zeros(
            len(tickers),
            dtype=float,
        )
    else:
        previous_current = previous_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)

        dropped_weight = float(previous_weights.loc[~previous_weights.index.isin(tickers)].sum())

        turnover_expression = 0.5 * (cp.norm1(weights - previous_current) + dropped_weight)

    objective = cp.Maximize(
        predicted_alpha
        - (config.risk_aversion * predicted_variance)
        - (config.turnover_penalty * turnover_expression)
    )

    problem = cp.Problem(
        objective,
        constraints,
    )

    problem.solve(
        solver=cp.OSQP,
        eps_abs=config.solver_tolerance,
        eps_rel=config.solver_tolerance,
        max_iter=100_000,
    )

    if problem.status not in {
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    }:
        raise PortfolioOptimizationError(
            f"Portfolio optimization failed with status {problem.status}."
        )

    solved_weights = np.asarray(
        weights.value,
        dtype=float,
    ).reshape(-1)

    solved_weights[np.abs(solved_weights) < config.weight_tolerance] = 0.0

    solved_weights = np.clip(
        solved_weights,
        0.0,
        None,
    )

    portfolio_beta = float(beta @ solved_weights)

    position_adv_fractions = (
        config.reference_portfolio_value * solved_weights / average_dollar_volume
    )

    maximum_position_adv_fraction = float(position_adv_fractions.max())

    total_weight = float(solved_weights.sum())

    if total_weight <= 0.0:
        raise PortfolioOptimizationError("Optimizer produced zero total weight.")

    if abs(total_weight - 1.0) > max(
        10.0 * config.solver_tolerance,
        config.weight_tolerance,
    ):
        raise PortfolioOptimizationError(
            "Optimizer produced a portfolio whose "
            "weights do not sum sufficiently close to one: "
            f"{total_weight:.12f}."
        )

    if solved_weights.max() > config.max_security_weight + config.weight_tolerance:
        raise PortfolioOptimizationError("Solved portfolio violates the maximum security weight.")

    solved_sector_weights = (
        candidates.assign(solved_weight=solved_weights).groupby("sector")["solved_weight"].sum()
    )

    if solved_sector_weights.max() > config.max_sector_weight + config.weight_tolerance:
        raise PortfolioOptimizationError("Solved portfolio violates the maximum sector weight.")

    current_series = pd.Series(
        solved_weights,
        index=tickers,
        dtype=float,
    )

    one_way_turnover = _calculate_realized_turnover(
        previous_weights,
        current_series,
    )

    variance_value = float(solved_weights @ covariance_matrix @ solved_weights)

    alpha_value = float(alpha @ solved_weights)

    diagnostics: dict[
        str,
        float | str,
    ] = {
        "solver_status": str(problem.status),
        "objective_value": float(problem.value),
        "predicted_alpha_proxy": (alpha_value),
        "predicted_variance": (variance_value),
        "predicted_volatility": float(
            np.sqrt(
                max(
                    variance_value,
                    0.0,
                )
            )
        ),
        "one_way_turnover": (one_way_turnover),
        "portfolio_beta_vs_spy": (portfolio_beta),
        "maximum_position_adv_fraction": (maximum_position_adv_fraction),
    }

    return (
        solved_weights,
        diagnostics,
    )


def build_alpha_risk_turnover_portfolios(
    final_signal: pd.DataFrame,
    covariance: pd.DataFrame,
    risk_estimates: pd.DataFrame,
    *,
    config: PortfolioOptimizerConfig | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build optimized portfolios through time."""
    if config is None:
        config = PortfolioOptimizerConfig()

    config.validate()

    signal = _prepare_signal(final_signal)

    portfolio_blocks: list[pd.DataFrame] = []

    diagnostic_rows: list[dict[str, Any]] = []

    previous_weights: pd.Series | None = None

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
            config=config,
        )

        tickers = candidates["ticker"].astype(str).tolist()

        covariance_matrix = _build_covariance_matrix(
            covariance,
            as_of_date=normalized_date,
            tickers=tickers,
        )

        candidate_risk = _prepare_candidate_risk(
            risk_estimates,
            as_of_date=normalized_date,
            tickers=tickers,
        )

        (
            solved_weights,
            optimizer_diagnostics,
        ) = _solve_portfolio(
            candidates,
            covariance_matrix,
            candidate_risk,
            previous_weights=previous_weights,
            config=config,
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

        candidates["annualized_alpha_proxy"] = (
            candidates["percentile_score"] - 0.5
        ) * config.annualized_alpha_scale

        candidates["previous_weight"] = previous_candidate_weights

        candidates["raw_score_weight"] = np.nan

        candidates["weight"] = solved_weights

        candidates["method"] = "alpha_risk_turnover"

        candidates["beta_vs_spy"] = candidate_risk["beta_vs_spy"].to_numpy(dtype=float)

        candidates["average_dollar_volume"] = candidate_risk["average_dollar_volume"].to_numpy(
            dtype=float
        )

        candidates["position_adv_fraction"] = (
            config.reference_portfolio_value
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
                    "raw_score_weight",
                    "annualized_alpha_proxy",
                    "previous_weight",
                    "weight",
                    "beta_vs_spy",
                    "average_dollar_volume",
                    "position_adv_fraction",
                ],
            ]
        )

        sector_weights = candidates.assign(weight=solved_weights).groupby("sector")["weight"].sum()

        diagnostic_rows.append(
            {
                "as_of_date": (normalized_date),
                **optimizer_diagnostics,
                "positions": int((solved_weights > config.weight_tolerance).sum()),
                "maximum_weight": float(solved_weights.max()),
                "maximum_sector_weight": float(sector_weights.max()),
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


def validate_optimizer_diagnostics(
    diagnostics: pd.DataFrame,
    *,
    config: PortfolioOptimizerConfig,
) -> pd.DataFrame:
    """Audit beta and liquidity constraints of optimized portfolios."""
    _require_columns(
        diagnostics,
        (
            "portfolio_beta_vs_spy",
            "maximum_position_adv_fraction",
        ),
        dataset_name="optimizer diagnostics",
    )

    checks = [
        (
            "portfolio_beta_lower_limit",
            int(
                diagnostics["portfolio_beta_vs_spy"]
                .lt(config.minimum_portfolio_beta - config.weight_tolerance)
                .sum()
            ),
            ("Optimized portfolio beta must remain above its lower bound."),
        ),
        (
            "portfolio_beta_upper_limit",
            int(
                diagnostics["portfolio_beta_vs_spy"]
                .gt(config.maximum_portfolio_beta + config.weight_tolerance)
                .sum()
            ),
            ("Optimized portfolio beta must remain below its upper bound."),
        ),
        (
            "position_liquidity_limit",
            int(
                diagnostics["maximum_position_adv_fraction"]
                .gt(config.max_position_adv_fraction + config.weight_tolerance)
                .sum()
            ),
            ("No target position may exceed the configured share of Average Dollar Volume."),
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
