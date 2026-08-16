"""Point-in-time covariance estimation for cross-sectional portfolios."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class CovarianceEstimateError(ValueError):
    """Raised when covariance estimates cannot be constructed."""


@dataclass(frozen=True)
class CovarianceConfig:
    """Configuration for rolling covariance estimation."""

    window_sessions: int = 252
    minimum_observations: int = 126
    annualization_factor: int = 252
    method: str = "ledoit_wolf"

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> CovarianceConfig:
        """Build covariance configuration from project settings."""
        return cls(
            window_sessions=int(
                values.get(
                    "covariance_window_sessions",
                    252,
                )
            ),
            minimum_observations=int(
                values.get(
                    "covariance_minimum_observations",
                    126,
                )
            ),
            annualization_factor=int(
                values.get(
                    "annualization_factor",
                    252,
                )
            ),
            method=str(
                values.get(
                    "covariance_method",
                    "ledoit_wolf",
                )
            ),
        )

    def validate(self) -> None:
        """Validate covariance configuration."""
        if self.window_sessions < 2:
            raise CovarianceEstimateError("window_sessions must exceed one.")

        if self.minimum_observations < 2:
            raise CovarianceEstimateError("minimum_observations must exceed one.")

        if self.minimum_observations > self.window_sessions:
            raise CovarianceEstimateError("minimum_observations cannot exceed window_sessions.")

        if self.annualization_factor < 1:
            raise CovarianceEstimateError("annualization_factor must be positive.")

        if self.method != "ledoit_wolf":
            raise CovarianceEstimateError(
                "Only ledoit_wolf covariance estimation is currently supported."
            )


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    """Require columns to exist."""
    missing = sorted(set(columns).difference(data.columns))

    if missing:
        raise CovarianceEstimateError(
            f"{dataset_name} is missing columns: " + ", ".join(missing) + "."
        )


def _prepare_return_panel(
    market_data: pd.DataFrame,
    *,
    analysis_end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build a clean daily return panel up to the required final date."""
    _require_columns(
        market_data,
        (
            "date",
            "ticker",
            "adjusted_close",
        ),
        dataset_name="market data",
    )

    data = market_data.loc[
        :,
        [
            "date",
            "ticker",
            "adjusted_close",
        ],
    ].copy()

    data["date"] = pd.to_datetime(data["date"]).dt.normalize()

    data["ticker"] = data["ticker"].astype(str)

    data = data.loc[data["date"].le(analysis_end_date)].copy()

    data["adjusted_close"] = pd.to_numeric(
        data["adjusted_close"],
        errors="coerce",
    )

    if data.duplicated(
        [
            "date",
            "ticker",
        ]
    ).any():
        raise CovarianceEstimateError("Market data contain duplicate date-ticker rows.")

    if (
        data[
            [
                "date",
                "ticker",
                "adjusted_close",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise CovarianceEstimateError("Relevant market history contains missing required values.")

    if data["adjusted_close"].le(0.0).any():
        raise CovarianceEstimateError(
            "Relevant market history contains non-positive adjusted prices."
        )

    data = data.sort_values(
        [
            "ticker",
            "date",
        ]
    ).reset_index(drop=True)

    data["daily_return"] = data.groupby(
        "ticker",
        sort=False,
    )["adjusted_close"].pct_change(fill_method=None)

    return data.pivot(
        index="date",
        columns="ticker",
        values="daily_return",
    ).sort_index()


def _covariance_to_correlation(
    covariance: np.ndarray,
) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix."""
    variances = np.diag(covariance)

    if not np.isfinite(variances).all() or (variances <= 0.0).any():
        raise CovarianceEstimateError("Covariance matrix contains invalid variances.")

    standard_deviations = np.sqrt(variances)

    denominator = np.outer(
        standard_deviations,
        standard_deviations,
    )

    correlation = covariance / denominator

    correlation = np.clip(
        correlation,
        -1.0,
        1.0,
    )

    np.fill_diagonal(
        correlation,
        1.0,
    )

    return correlation


def build_covariance_matrices(
    market_data: pd.DataFrame,
    signal_universe: pd.DataFrame,
    *,
    config: CovarianceConfig | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build rolling shrinkage covariance matrices for all signal dates."""
    if config is None:
        config = CovarianceConfig()

    config.validate()

    _require_columns(
        signal_universe,
        (
            "as_of_date",
            "ticker",
        ),
        dataset_name="signal universe",
    )

    signal = signal_universe.loc[
        :,
        [
            "as_of_date",
            "ticker",
        ],
    ].copy()

    signal["as_of_date"] = pd.to_datetime(signal["as_of_date"]).dt.normalize()

    signal["ticker"] = signal["ticker"].astype(str)

    if signal.duplicated(
        [
            "as_of_date",
            "ticker",
        ]
    ).any():
        raise CovarianceEstimateError("Signal universe contains duplicate date-ticker rows.")

    analysis_end_date = signal["as_of_date"].max()

    return_panel = _prepare_return_panel(
        market_data,
        analysis_end_date=(analysis_end_date),
    )

    matrix_rows: list[dict[str, Any]] = []

    diagnostic_rows: list[dict[str, Any]] = []

    for (
        as_of_date,
        date_signal,
    ) in signal.groupby(
        "as_of_date",
        sort=True,
    ):
        tickers = sorted(date_signal["ticker"].unique())

        missing_tickers = sorted(set(tickers).difference(return_panel.columns))

        if missing_tickers:
            raise CovarianceEstimateError(
                "Return history is missing tickers at "
                f"{as_of_date.date()}: " + ", ".join(missing_tickers) + "."
            )

        available = return_panel.loc[
            return_panel.index <= as_of_date,
            tickers,
        ]

        complete_history = available.dropna(how="any").tail(config.window_sessions).copy()

        observation_count = len(complete_history)

        if observation_count < config.minimum_observations:
            raise CovarianceEstimateError(
                "Insufficient joint return history at "
                f"{as_of_date.date()}: "
                f"{observation_count} observations."
            )

        values = complete_history.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise CovarianceEstimateError("Return window contains non-finite values.")

        estimator = LedoitWolf(assume_centered=False)

        estimator.fit(values)

        daily_covariance = estimator.covariance_

        annualized_covariance = daily_covariance * config.annualization_factor

        correlation = _covariance_to_correlation(annualized_covariance)

        if not np.allclose(
            annualized_covariance,
            annualized_covariance.T,
            atol=1e-12,
        ):
            raise CovarianceEstimateError("Estimated covariance matrix is not symmetric.")

        eigenvalues = np.linalg.eigvalsh(annualized_covariance)

        minimum_eigenvalue = float(eigenvalues.min())

        maximum_eigenvalue = float(eigenvalues.max())

        if minimum_eigenvalue < -1e-10:
            raise CovarianceEstimateError(
                "Estimated covariance matrix is not positive semidefinite."
            )

        sample_covariance = (
            np.cov(
                values,
                rowvar=False,
                ddof=1,
            )
            * config.annualization_factor
        )

        sample_condition_number = float(np.linalg.cond(sample_covariance))

        shrinkage_condition_number = float(np.linalg.cond(annualized_covariance))

        off_diagonal_mask = ~np.eye(
            len(tickers),
            dtype=bool,
        )

        off_diagonal_correlations = correlation[off_diagonal_mask]

        diagnostic_rows.append(
            {
                "as_of_date": (pd.Timestamp(as_of_date)),
                "assets": len(tickers),
                "observations": (observation_count),
                "window_start_date": (pd.Timestamp(complete_history.index.min())),
                "window_end_date": (pd.Timestamp(complete_history.index.max())),
                "shrinkage": float(estimator.shrinkage_),
                "minimum_eigenvalue": (minimum_eigenvalue),
                "maximum_eigenvalue": (maximum_eigenvalue),
                "sample_condition_number": (sample_condition_number),
                "shrinkage_condition_number": (shrinkage_condition_number),
                "mean_pairwise_correlation": float(off_diagonal_correlations.mean()),
                "median_pairwise_correlation": float(np.median(off_diagonal_correlations)),
                "maximum_pairwise_correlation": float(off_diagonal_correlations.max()),
                "minimum_pairwise_correlation": float(off_diagonal_correlations.min()),
            }
        )

        for row_number, ticker_a in enumerate(tickers):
            for column_number, ticker_b in enumerate(tickers):
                matrix_rows.append(
                    {
                        "as_of_date": (pd.Timestamp(as_of_date)),
                        "ticker_a": (ticker_a),
                        "ticker_b": (ticker_b),
                        "annualized_covariance": float(
                            annualized_covariance[
                                row_number,
                                column_number,
                            ]
                        ),
                        "correlation": float(
                            correlation[
                                row_number,
                                column_number,
                            ]
                        ),
                    }
                )

    matrices = (
        pd.DataFrame(matrix_rows)
        .sort_values(
            [
                "as_of_date",
                "ticker_a",
                "ticker_b",
            ]
        )
        .reset_index(drop=True)
    )

    diagnostics = pd.DataFrame(diagnostic_rows).sort_values("as_of_date").reset_index(drop=True)

    return (
        matrices,
        diagnostics,
    )


def validate_covariance_matrices(
    matrices: pd.DataFrame,
    diagnostics: pd.DataFrame,
    signal_universe: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Run deterministic readiness checks on covariance artifacts."""
    _require_columns(
        matrices,
        (
            "as_of_date",
            "ticker_a",
            "ticker_b",
            "annualized_covariance",
            "correlation",
        ),
        dataset_name="covariance matrices",
    )

    _require_columns(
        diagnostics,
        (
            "as_of_date",
            "assets",
            "observations",
            "window_start_date",
            "window_end_date",
            "shrinkage",
            "minimum_eigenvalue",
        ),
        dataset_name="covariance diagnostics",
    )

    matrices = matrices.copy()
    diagnostics = diagnostics.copy()
    signal = signal_universe.copy()

    matrices["as_of_date"] = pd.to_datetime(matrices["as_of_date"]).dt.normalize()

    diagnostics["as_of_date"] = pd.to_datetime(diagnostics["as_of_date"]).dt.normalize()

    signal["as_of_date"] = pd.to_datetime(signal["as_of_date"]).dt.normalize()

    diagnostics["window_start_date"] = pd.to_datetime(
        diagnostics["window_start_date"]
    ).dt.normalize()

    diagnostics["window_end_date"] = pd.to_datetime(diagnostics["window_end_date"]).dt.normalize()

    duplicate_violations = int(
        matrices.duplicated(
            [
                "as_of_date",
                "ticker_a",
                "ticker_b",
            ]
        ).sum()
    )

    expected_dates = set(signal["as_of_date"].unique())

    observed_dates = set(matrices["as_of_date"].unique())

    missing_date_violations = len(expected_dates.symmetric_difference(observed_dates))

    matrix_size_violations = 0
    symmetry_violations = 0
    diagonal_violations = 0
    ticker_coverage_violations = 0

    for (
        as_of_date,
        matrix_data,
    ) in matrices.groupby(
        "as_of_date",
        sort=True,
    ):
        expected_tickers = sorted(
            signal.loc[
                signal["as_of_date"].eq(as_of_date),
                "ticker",
            ].unique()
        )

        expected_size = len(expected_tickers) ** 2

        if len(matrix_data) != expected_size:
            matrix_size_violations += 1

        observed_a = set(matrix_data["ticker_a"])

        observed_b = set(matrix_data["ticker_b"])

        if observed_a != set(expected_tickers) or observed_b != set(expected_tickers):
            ticker_coverage_violations += 1

        covariance_wide = matrix_data.pivot(
            index="ticker_a",
            columns="ticker_b",
            values="annualized_covariance",
        ).reindex(
            index=expected_tickers,
            columns=expected_tickers,
        )

        values = covariance_wide.to_numpy(dtype=float)

        if not np.allclose(
            values,
            values.T,
            atol=tolerance,
        ):
            symmetry_violations += 1

        if (np.diag(values) < -tolerance).any():
            diagonal_violations += 1

    checks = [
        (
            "unique_matrix_keys",
            duplicate_violations,
            ("Every date and ticker pair must appear exactly once."),
        ),
        (
            "signal_date_coverage",
            missing_date_violations,
            ("Covariance dates must exactly match the final alpha signal dates."),
        ),
        (
            "matrix_dimensions",
            matrix_size_violations,
            ("Every covariance matrix must contain the complete cross-product of assets."),
        ),
        (
            "ticker_coverage",
            ticker_coverage_violations,
            ("Every matrix must contain exactly the signal universe for that date."),
        ),
        (
            "matrix_symmetry",
            symmetry_violations,
            ("Every covariance matrix must be symmetric."),
        ),
        (
            "non_negative_variances",
            diagonal_violations,
            ("Covariance diagonals must contain non-negative variances."),
        ),
        (
            "positive_semidefinite",
            int(diagnostics["minimum_eigenvalue"].lt(-tolerance).sum()),
            ("Shrinkage covariance matrices must be positive semidefinite."),
        ),
        (
            "valid_shrinkage",
            int((diagnostics["shrinkage"].lt(0.0) | diagnostics["shrinkage"].gt(1.0)).sum()),
            ("Ledoit-Wolf shrinkage must remain between zero and one."),
        ),
        (
            "point_in_time_windows",
            int(diagnostics["window_end_date"].gt(diagnostics["as_of_date"]).sum()),
            ("Every covariance estimation window must end on or before as_of_date."),
        ),
        (
            "finite_covariances",
            int((~np.isfinite(matrices["annualized_covariance"].to_numpy(dtype=float))).sum()),
            ("All covariance estimates must be finite."),
        ),
        (
            "valid_correlations",
            int(matrices["correlation"].abs().gt(1.0 + tolerance).sum()),
            ("All correlations must remain between -1 and 1."),
        ),
    ]

    return pd.DataFrame(
        [
            {
                "check": name,
                "status": ("PASS" if violations == 0 else "FAIL"),
                "violations": (violations),
                "description": (description),
            }
            for (
                name,
                violations,
                description,
            ) in checks
        ]
    )
