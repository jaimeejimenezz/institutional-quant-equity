"""Transparent constrained portfolio-construction baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd


class BaselinePortfolioConstructionError(ValueError):
    """Raised when baseline target weights cannot be constructed."""


@dataclass(frozen=True)
class BaselinePortfolioConfig:
    """Configuration for baseline portfolio construction."""

    candidate_count: int = 25
    equal_weight_positions: int = 25
    max_security_weight: float = 0.05
    max_sector_weight: float = 0.25
    minimum_positions: int = 20
    weight_tolerance: float = 1e-8

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> BaselinePortfolioConfig:
        """Build configuration from project settings."""
        return cls(
            candidate_count=int(
                values.get(
                    "candidate_count",
                    25,
                )
            ),
            equal_weight_positions=int(
                values.get(
                    "equal_weight_positions",
                    25,
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
            minimum_positions=int(
                values.get(
                    "minimum_positions",
                    20,
                )
            ),
            weight_tolerance=float(
                values.get(
                    "weight_tolerance",
                    1e-8,
                )
            ),
        )

    def validate(self) -> None:
        """Validate portfolio-construction settings."""
        if self.candidate_count < 1:
            raise BaselinePortfolioConstructionError("candidate_count must be positive.")

        if self.equal_weight_positions < 1:
            raise BaselinePortfolioConstructionError("equal_weight_positions must be positive.")

        if self.minimum_positions < 1:
            raise BaselinePortfolioConstructionError("minimum_positions must be positive.")

        if self.equal_weight_positions > self.candidate_count:
            raise BaselinePortfolioConstructionError(
                "equal_weight_positions cannot exceed candidate_count."
            )

        if self.minimum_positions > self.candidate_count:
            raise BaselinePortfolioConstructionError(
                "minimum_positions cannot exceed candidate_count."
            )

        if not (0.0 < self.max_security_weight <= 1.0):
            raise BaselinePortfolioConstructionError(
                "max_security_weight must be between zero and one."
            )

        if not (0.0 < self.max_sector_weight <= 1.0):
            raise BaselinePortfolioConstructionError(
                "max_sector_weight must be between zero and one."
            )

        if self.candidate_count * self.max_security_weight < 1.0 - self.weight_tolerance:
            raise BaselinePortfolioConstructionError(
                "candidate_count and max_security_weight cannot support a fully invested portfolio."
            )

        if 1.0 / self.equal_weight_positions > self.max_security_weight + self.weight_tolerance:
            raise BaselinePortfolioConstructionError(
                "Equal-weight positions would violate the maximum security weight."
            )

        if self.weight_tolerance <= 0.0:
            raise BaselinePortfolioConstructionError("weight_tolerance must be positive.")


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    """Require expected columns."""
    missing = sorted(set(columns).difference(data.columns))

    if missing:
        raise BaselinePortfolioConstructionError(
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
        raise BaselinePortfolioConstructionError("Final alpha signal contains duplicate keys.")

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
        raise BaselinePortfolioConstructionError(
            "Final alpha signal contains missing ranking values."
        )

    return signal


def _select_sector_aware_candidates(
    date_signal: pd.DataFrame,
    *,
    count: int,
    max_sector_weight: float,
) -> pd.DataFrame:
    """Select highest-ranked names while controlling sector concentration."""
    equal_weight = 1.0 / float(count)

    maximum_names_per_sector = int(np.floor((max_sector_weight + 1e-12) / equal_weight))

    if maximum_names_per_sector < 1:
        raise BaselinePortfolioConstructionError(
            "Sector limit cannot accommodate even one equal-weight position."
        )

    ordered = date_signal.sort_values(
        [
            "rank",
            "ticker",
        ]
    )

    sector_counts: dict[
        str,
        int,
    ] = {}

    selected_indices: list[int] = []

    for index, row in ordered.iterrows():
        sector = str(row["sector"])

        current_count = sector_counts.get(
            sector,
            0,
        )

        if current_count >= maximum_names_per_sector:
            continue

        selected_indices.append(index)

        sector_counts[sector] = current_count + 1

        if len(selected_indices) == count:
            break

    if len(selected_indices) != count:
        raise BaselinePortfolioConstructionError(
            "Unable to construct enough positions under the sector constraint."
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


def build_equal_weight_portfolios(
    final_signal: pd.DataFrame,
    *,
    config: BaselinePortfolioConfig | None = None,
) -> pd.DataFrame:
    """Build sector-controlled equal-weight target portfolios."""
    if config is None:
        config = BaselinePortfolioConfig()

    config.validate()

    signal = _prepare_signal(final_signal)

    blocks = []

    for (
        _as_of_date,
        date_signal,
    ) in signal.groupby(
        "as_of_date",
        sort=True,
    ):
        selected = _select_sector_aware_candidates(
            date_signal,
            count=(config.equal_weight_positions),
            max_sector_weight=(config.max_sector_weight),
        )

        selected["weight"] = 1.0 / config.equal_weight_positions

        selected["raw_score_weight"] = selected["weight"]

        selected["method"] = "top_n_equal_weight"

        blocks.append(selected)

    result = pd.concat(
        blocks,
        ignore_index=True,
    )

    return result.loc[
        :,
        [
            "as_of_date",
            "ticker",
            "sector",
            "method",
            "rank",
            "percentile_score",
            "raw_score_weight",
            "weight",
        ],
    ]


def _solve_score_weights(
    candidates: pd.DataFrame,
    *,
    config: BaselinePortfolioConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Project score-proportional weights onto portfolio constraints."""
    scores = candidates["percentile_score"].to_numpy(dtype=float)

    scores = np.clip(
        scores,
        0.0,
        None,
    )

    if scores.sum() <= 0.0:
        raw_weights = np.full(
            len(candidates),
            1.0 / len(candidates),
            dtype=float,
        )
    else:
        raw_weights = scores / scores.sum()

    weights = cp.Variable(len(candidates))

    constraints = [
        weights >= 0.0,
        weights <= config.max_security_weight,
        cp.sum(weights) == 1.0,
    ]

    sectors = candidates["sector"].astype(str).to_numpy()

    for sector in sorted(set(sectors)):
        mask = np.where(sectors == sector)[0]

        constraints.append(cp.sum(weights[mask]) <= config.max_sector_weight)

    objective = cp.Minimize(cp.sum_squares(weights - raw_weights))

    problem = cp.Problem(
        objective,
        constraints,
    )

    problem.solve()

    if problem.status not in {
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    }:
        raise BaselinePortfolioConstructionError(
            f"Score-weight projection failed with status {problem.status}."
        )

    solved_weights = np.asarray(
        weights.value,
        dtype=float,
    ).reshape(-1)

    solved_weights[np.abs(solved_weights) < config.weight_tolerance] = 0.0

    if (solved_weights < -config.weight_tolerance).any():
        raise BaselinePortfolioConstructionError(
            "Score-weight projection produced negative weights."
        )

    solved_weights = np.clip(
        solved_weights,
        0.0,
        None,
    )

    total_weight = float(solved_weights.sum())

    if total_weight <= 0.0:
        raise BaselinePortfolioConstructionError(
            "Score-weight projection produced zero total weight."
        )

    solved_weights = solved_weights / total_weight

    return (
        raw_weights,
        solved_weights,
    )


def build_score_weighted_portfolios(
    final_signal: pd.DataFrame,
    *,
    config: BaselinePortfolioConfig | None = None,
) -> pd.DataFrame:
    """Build constrained portfolios tilted toward stronger alpha scores."""
    if config is None:
        config = BaselinePortfolioConfig()

    config.validate()

    signal = _prepare_signal(final_signal)

    blocks = []

    for (
        _as_of_date,
        date_signal,
    ) in signal.groupby(
        "as_of_date",
        sort=True,
    ):
        candidates = _select_sector_aware_candidates(
            date_signal,
            count=(config.candidate_count),
            max_sector_weight=(config.max_sector_weight),
        )

        (
            raw_weights,
            solved_weights,
        ) = _solve_score_weights(
            candidates,
            config=config,
        )

        candidates["raw_score_weight"] = raw_weights

        candidates["weight"] = solved_weights

        candidates["method"] = "score_weighted"

        blocks.append(candidates)

    result = pd.concat(
        blocks,
        ignore_index=True,
    )

    return result.loc[
        :,
        [
            "as_of_date",
            "ticker",
            "sector",
            "method",
            "rank",
            "percentile_score",
            "raw_score_weight",
            "weight",
        ],
    ]


def compute_portfolio_diagnostics(
    weights: pd.DataFrame,
    *,
    weight_tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Summarize portfolio constraints, concentration and turnover."""
    _require_columns(
        weights,
        (
            "as_of_date",
            "ticker",
            "sector",
            "method",
            "weight",
        ),
        dataset_name="target weights",
    )

    data = weights.copy()

    data["as_of_date"] = pd.to_datetime(data["as_of_date"]).dt.normalize()

    rows = []

    for (
        method,
        method_data,
    ) in data.groupby(
        "method",
        sort=True,
    ):
        previous_weights: pd.Series | None = None

        for (
            as_of_date,
            date_data,
        ) in method_data.groupby(
            "as_of_date",
            sort=True,
        ):
            current_weights = date_data.set_index("ticker")["weight"]

            sector_weights = date_data.groupby("sector")["weight"].sum()

            hhi = float(np.square(current_weights.to_numpy(dtype=float)).sum())

            if previous_weights is None:
                one_way_turnover = np.nan

                two_way_turnover = np.nan
            else:
                aligned = pd.concat(
                    [
                        previous_weights.rename("previous"),
                        current_weights.rename("current"),
                    ],
                    axis=1,
                ).fillna(0.0)

                two_way_turnover = float((aligned["current"] - aligned["previous"]).abs().sum())

                one_way_turnover = 0.5 * two_way_turnover

            positive_weights = current_weights.loc[current_weights > weight_tolerance]

            if positive_weights.empty:
                minimum_positive_weight = np.nan
            else:
                minimum_positive_weight = float(positive_weights.min())

            rows.append(
                {
                    "as_of_date": pd.Timestamp(as_of_date),
                    "method": method,
                    "positions": int((current_weights > weight_tolerance).sum()),
                    "weight_sum": float(current_weights.sum()),
                    "maximum_weight": float(current_weights.max()),
                    "minimum_positive_weight": (minimum_positive_weight),
                    "maximum_sector_weight": float(sector_weights.max()),
                    "concentration_hhi": (hhi),
                    "effective_positions": float(1.0 / hhi),
                    "one_way_turnover": (one_way_turnover),
                    "two_way_turnover": (two_way_turnover),
                }
            )

            previous_weights = current_weights

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "method",
                "as_of_date",
            ]
        )
        .reset_index(drop=True)
    )


def validate_baseline_portfolios(
    weights: pd.DataFrame,
    *,
    config: BaselinePortfolioConfig | None = None,
) -> pd.DataFrame:
    """Audit baseline portfolio constraints."""
    if config is None:
        config = BaselinePortfolioConfig()

    config.validate()

    diagnostics = compute_portfolio_diagnostics(
        weights,
        weight_tolerance=(config.weight_tolerance),
    )

    duplicate_violations = int(
        weights.duplicated(
            [
                "method",
                "as_of_date",
                "ticker",
            ]
        ).sum()
    )

    checks = [
        (
            "unique_weight_keys",
            duplicate_violations,
            ("Target weights must have unique method-date-ticker keys."),
        ),
        (
            "fully_invested",
            int((np.abs(diagnostics["weight_sum"] - 1.0) > config.weight_tolerance).sum()),
            ("Every portfolio must sum to one."),
        ),
        (
            "long_only",
            int(weights["weight"].lt(-config.weight_tolerance).sum()),
            ("Portfolio weights must be long-only."),
        ),
        (
            "security_weight_limit",
            int(
                diagnostics["maximum_weight"]
                .gt(config.max_security_weight + config.weight_tolerance)
                .sum()
            ),
            ("No security may exceed its maximum weight."),
        ),
        (
            "sector_weight_limit",
            int(
                diagnostics["maximum_sector_weight"]
                .gt(config.max_sector_weight + config.weight_tolerance)
                .sum()
            ),
            ("No sector may exceed its maximum portfolio weight."),
        ),
        (
            "minimum_positions",
            int(diagnostics["positions"].lt(config.minimum_positions).sum()),
            ("Every portfolio must contain enough active positions."),
        ),
        (
            "finite_weights",
            int((~np.isfinite(weights["weight"].to_numpy(dtype=float))).sum()),
            ("All target weights must be finite."),
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
