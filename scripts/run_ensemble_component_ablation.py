"""Evaluate final-alpha ensemble component ablations through portfolio backtests."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_equity.backtest import (
    ExecutionCostConfig,
    MVPBacktestConfig,
    PerformanceEvaluationConfig,
    build_buy_and_hold_benchmark,
    evaluate_performance,
    run_mvp_backtest,
)
from quant_equity.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    load_config,
)
from quant_equity.logging_config import configure_logging
from quant_equity.models import (
    EnsembleConfig,
    build_ablation_candidates,
    build_component_scores,
    build_final_alpha_signal,
)
from quant_equity.portfolio import (
    BaselinePortfolioConfig,
    build_score_weighted_portfolios,
    compute_portfolio_diagnostics,
)

PREDICTIONS_PATH = PROCESSED_DATA_DIR / "predictions_oos_all_models.parquet"

FINAL_SIGNAL_PATH = PROCESSED_DATA_DIR / "final_alpha_signal.parquet"

VALIDATION_WEIGHTS_PATH = REPORTS_DIR / "tables" / "ensemble_validation_weights.csv"

MARKET_DATA_PATH = PROCESSED_DATA_DIR / "market_daily.parquet"

RISK_ESTIMATES_PATH = PROCESSED_DATA_DIR / "risk_estimates.parquet"

SPY_DATA_PATH = PROCESSED_DATA_DIR / "benchmark_spy_daily.parquet"

ABLATION_SIGNALS_PATH = PROCESSED_DATA_DIR / "robustness_ensemble_ablation_signals.parquet"

ABLATION_WEIGHTS_PATH = PROCESSED_DATA_DIR / "robustness_ensemble_ablation_weights.parquet"

TABLES_DIR = REPORTS_DIR / "tables"

RESULTS_PATH = TABLES_DIR / "robustness_ensemble_component_ablation.csv"

DIAGNOSTICS_PATH = TABLES_DIR / "robustness_ensemble_component_ablation_diagnostics.csv"

CHECKS_PATH = TABLES_DIR / "robustness_ensemble_component_ablation_checks.csv"

REPORT_PATH = REPORTS_DIR / "robustness" / "ensemble_component_ablation.md"

BASELINE_SCENARIO = "full_ensemble"

EXPECTED_ABLATIONS = {
    "without_composite",
    "without_elastic_net",
    "without_lightgbm_ranker",
}

EXPECTED_SCENARIOS = {
    BASELINE_SCENARIO,
    *EXPECTED_ABLATIONS,
}

EXPECTED_CROSS_SECTION_SIZE = 50

CANDIDATE_COUNT = 25
MAX_SECURITY_WEIGHT = 0.05
MAX_SECTOR_WEIGHT = 0.25
MINIMUM_POSITIONS = 20
WEIGHT_TOLERANCE = 1.0e-8

ENSEMBLE_WEIGHT_COLUMNS = (
    "composite_weight",
    "elastic_net_weight",
    "lightgbm_ranker_weight",
)


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


def _write_parquet(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write one Parquet table."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    data.to_parquet(
        path,
        index=False,
    )


def _normalize_fold_ids(
    data: pd.DataFrame,
    *,
    dataset_name: str,
) -> pd.DataFrame:
    """Normalize fold identifiers so persisted artifacts merge reliably."""
    if "fold_id" not in data.columns:
        raise ValueError(f"{dataset_name} is missing fold_id.")

    result = data.copy()

    raw = result["fold_id"]

    if raw.isna().any():
        raise ValueError(f"{dataset_name} contains missing fold_id values.")

    numeric = pd.to_numeric(
        raw,
        errors="coerce",
    )

    if numeric.notna().all():
        rounded = np.rint(numeric.to_numpy(dtype=float))

        if not np.allclose(
            numeric.to_numpy(dtype=float),
            rounded,
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError(f"{dataset_name} contains non-integral numeric fold_id values.")

        result["fold_id"] = pd.Series(
            rounded.astype(np.int64),
            index=result.index,
        ).astype("string")
    else:
        result["fold_id"] = raw.astype("string").str.strip()

    if result["fold_id"].isna().any():
        raise ValueError(f"{dataset_name} contains invalid fold_id values after normalization.")

    return result


def _validate_fold_id_alignment(
    predictions: pd.DataFrame,
    validation_weights: pd.DataFrame,
) -> None:
    """Require predictions and validation weights to reference identical folds."""
    prediction_folds = set(predictions["fold_id"].astype(str))

    validation_folds = set(validation_weights["fold_id"].astype(str))

    if prediction_folds != validation_folds:
        missing_in_weights = sorted(prediction_folds.difference(validation_folds))

        missing_in_predictions = sorted(validation_folds.difference(prediction_folds))

        raise ValueError(
            "Fold identifiers remain misaligned after normalization. "
            f"Missing in validation weights: {missing_in_weights[:5]}; "
            f"missing in predictions: {missing_in_predictions[:5]}."
        )


def _validation_weights_from_stored_signal(
    stored_final_signal: pd.DataFrame,
) -> pd.DataFrame:
    """Recover the exact fold weights persisted inside the final signal."""
    required = {
        "fold_id",
        *ENSEMBLE_WEIGHT_COLUMNS,
    }

    missing = sorted(required.difference(stored_final_signal.columns))

    if missing:
        raise ValueError(
            "Stored final alpha signal is missing ensemble weights: " + ", ".join(missing) + "."
        )

    data = stored_final_signal.loc[
        :,
        [
            "fold_id",
            *ENSEMBLE_WEIGHT_COLUMNS,
        ],
    ].copy()

    for column in ENSEMBLE_WEIGHT_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if data[list(ENSEMBLE_WEIGHT_COLUMNS)].isna().any().any():
        raise ValueError("Stored final alpha signal contains invalid ensemble weights.")

    within_fold_unique = data.groupby("fold_id")[list(ENSEMBLE_WEIGHT_COLUMNS)].nunique(
        dropna=False
    )

    if within_fold_unique.gt(1).any().any():
        raise ValueError("Stored final alpha signal has inconsistent weights within a fold.")

    weights = (
        data.groupby(
            "fold_id",
            as_index=False,
        )[list(ENSEMBLE_WEIGHT_COLUMNS)]
        .first()
        .sort_values("fold_id")
        .reset_index(drop=True)
    )

    weight_sum = weights[list(ENSEMBLE_WEIGHT_COLUMNS)].sum(axis=1)

    if not np.allclose(
        weight_sum.to_numpy(dtype=float),
        1.0,
        atol=1.0e-12,
        rtol=1.0e-10,
    ):
        raise ValueError("Stored ensemble weights do not sum to one.")

    return weights


def _audit_csv_weights(
    stored_weights: pd.DataFrame,
    csv_weights: pd.DataFrame,
) -> None:
    """Require the CSV report to agree numerically with stored production weights."""
    required = {
        "fold_id",
        *ENSEMBLE_WEIGHT_COLUMNS,
    }

    missing = sorted(required.difference(csv_weights.columns))

    if missing:
        raise ValueError(
            "Ensemble validation-weight CSV is missing columns: " + ", ".join(missing) + "."
        )

    csv = csv_weights.loc[
        :,
        [
            "fold_id",
            *ENSEMBLE_WEIGHT_COLUMNS,
        ],
    ].copy()

    for column in ENSEMBLE_WEIGHT_COLUMNS:
        csv[column] = pd.to_numeric(
            csv[column],
            errors="coerce",
        )

    if csv[list(ENSEMBLE_WEIGHT_COLUMNS)].isna().any().any():
        raise ValueError("Ensemble validation-weight CSV contains invalid weights.")

    if csv["fold_id"].duplicated().any():
        raise ValueError("Ensemble validation-weight CSV contains duplicate fold identifiers.")

    comparison = stored_weights.merge(
        csv,
        on="fold_id",
        how="outer",
        suffixes=(
            "_stored",
            "_csv",
        ),
        indicator=True,
        validate="one_to_one",
    )

    if not comparison["_merge"].eq("both").all():
        raise ValueError(
            "Stored production weights and validation-weight CSV do not reference identical folds."
        )

    for column in ENSEMBLE_WEIGHT_COLUMNS:
        stored_values = comparison[f"{column}_stored"].to_numpy(dtype=float)

        csv_values = comparison[f"{column}_csv"].to_numpy(dtype=float)

        if not np.allclose(
            stored_values,
            csv_values,
            atol=1.0e-12,
            rtol=1.0e-10,
        ):
            raise ValueError(f"Stored production weights differ materially from CSV {column}.")


def _portfolio_config() -> BaselinePortfolioConfig:
    """Return the frozen score-weighted portfolio configuration."""
    config = BaselinePortfolioConfig(
        candidate_count=CANDIDATE_COUNT,
        equal_weight_positions=CANDIDATE_COUNT,
        max_security_weight=MAX_SECURITY_WEIGHT,
        max_sector_weight=MAX_SECTOR_WEIGHT,
        minimum_positions=MINIMUM_POSITIONS,
        weight_tolerance=WEIGHT_TOLERANCE,
    )
    config.validate()
    return config


def _cross_section_percentile(
    values: pd.Series,
) -> pd.Series:
    """Map a cross-section to scores between zero and one."""
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).astype(float)

    if numeric.isna().any():
        raise ValueError("Ablation predictions contain missing values.")

    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Ablation predictions contain non-finite values.")

    if numeric.nunique() < 2:
        return pd.Series(
            0.5,
            index=numeric.index,
            dtype=float,
        )

    rank = numeric.rank(
        method="average",
        ascending=True,
    )

    return (rank - 1.0) / (len(numeric) - 1.0)


def _rank_signal(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create percentile scores and deterministic ranks for one scenario."""
    result = data.copy()

    result["percentile_score"] = result.groupby("as_of_date")["raw_prediction"].transform(
        _cross_section_percentile
    )

    blocks = []

    for _, month in result.groupby(
        "as_of_date",
        sort=True,
    ):
        month = month.sort_values(
            [
                "raw_prediction",
                "ticker",
            ],
            ascending=[
                False,
                True,
            ],
        ).copy()

        month["rank"] = np.arange(
            1,
            len(month) + 1,
            dtype=int,
        )

        blocks.append(month)

    return (
        pd.concat(
            blocks,
            ignore_index=True,
        )
        .sort_values(
            [
                "as_of_date",
                "rank",
            ]
        )
        .reset_index(drop=True)
    )


def _build_ablation_signals(
    predictions: pd.DataFrame,
    validation_weights: pd.DataFrame,
    stored_final_signal: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct baseline and one-component-removed final signals."""
    ensemble_config = EnsembleConfig(expected_cross_section_size=(EXPECTED_CROSS_SECTION_SIZE))

    component_scores = build_component_scores(
        predictions,
        config=ensemble_config,
    )

    reconstructed_baseline = build_final_alpha_signal(
        component_scores,
        validation_weights,
    )

    stored = stored_final_signal.copy()

    stored["as_of_date"] = pd.to_datetime(stored["as_of_date"]).dt.normalize()

    reconstructed_baseline["as_of_date"] = pd.to_datetime(
        reconstructed_baseline["as_of_date"]
    ).dt.normalize()

    comparison = reconstructed_baseline.loc[
        :,
        [
            "as_of_date",
            "ticker",
            "raw_prediction",
            "percentile_score",
            "rank",
        ],
    ].merge(
        stored.loc[
            :,
            [
                "as_of_date",
                "ticker",
                "raw_prediction",
                "percentile_score",
                "rank",
            ],
        ],
        on=[
            "as_of_date",
            "ticker",
        ],
        suffixes=(
            "_reconstructed",
            "_stored",
        ),
        how="inner",
        validate="one_to_one",
    )

    if len(comparison) != len(stored):
        raise ValueError("Reconstructed baseline does not align with stored final signal.")

    for column in (
        "raw_prediction",
        "percentile_score",
        "rank",
    ):
        left = comparison[f"{column}_reconstructed"].to_numpy(dtype=float)

        right = comparison[f"{column}_stored"].to_numpy(dtype=float)

        if not np.allclose(
            left,
            right,
            atol=1.0e-12,
            rtol=1.0e-10,
        ):
            raise ValueError(f"Reconstructed baseline differs from stored {column}.")

    baseline = reconstructed_baseline.loc[
        :,
        [
            "fold_id",
            "as_of_date",
            "ticker",
            "sector",
            "raw_prediction",
            "percentile_score",
            "rank",
        ],
    ].copy()

    baseline["scenario"] = BASELINE_SCENARIO

    ablation_candidates = build_ablation_candidates(
        component_scores,
        validation_weights,
    )

    observed_ablations = set(ablation_candidates["model_name"].unique())

    if observed_ablations != EXPECTED_ABLATIONS:
        raise ValueError(
            f"Unexpected ensemble ablation candidates. Received {sorted(observed_ablations)}."
        )

    blocks = [
        baseline,
    ]

    for scenario, group in ablation_candidates.groupby(
        "model_name",
        sort=True,
    ):
        signal = group.loc[
            :,
            [
                "fold_id",
                "as_of_date",
                "ticker",
                "sector",
                "prediction",
            ],
        ].rename(
            columns={
                "prediction": "raw_prediction",
            }
        )

        signal = _rank_signal(signal)

        signal["scenario"] = str(scenario)

        blocks.append(signal)

    combined = (
        pd.concat(
            blocks,
            ignore_index=True,
        )
        .sort_values(
            [
                "scenario",
                "as_of_date",
                "rank",
            ]
        )
        .reset_index(drop=True)
    )

    return combined


def _build_portfolios(
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """Construct one score-weighted portfolio for every ablation signal."""
    config = _portfolio_config()

    blocks = []

    for scenario, signal in signals.groupby(
        "scenario",
        sort=True,
    ):
        weights = build_score_weighted_portfolios(
            signal,
            config=config,
        ).copy()

        weights["method"] = str(scenario)

        blocks.append(weights)

    return (
        pd.concat(
            blocks,
            ignore_index=True,
        )
        .sort_values(
            [
                "method",
                "as_of_date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )


def _prepare_targets(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Convert portfolio weights to backtest targets."""
    return (
        weights.loc[
            :,
            [
                "as_of_date",
                "method",
                "ticker",
                "sector",
                "weight",
            ],
        ]
        .rename(
            columns={
                "method": "strategy_name",
                "weight": "target_weight",
            }
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


def _strategy_performance(
    summary: pd.DataFrame,
    *,
    benchmark_name: str,
) -> pd.DataFrame:
    """Return only portfolio-strategy rows."""
    return (
        summary.loc[~summary["strategy_name"].eq(benchmark_name)]
        .sort_values("strategy_name")
        .reset_index(drop=True)
    )


def _build_results(
    performance: pd.DataFrame,
    execution_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Combine performance and portfolio-construction diagnostics."""
    construction = (
        diagnostics.groupby(
            "method",
            as_index=False,
        )
        .agg(
            mean_positions=(
                "positions",
                "mean",
            ),
            maximum_weight=(
                "maximum_weight",
                "max",
            ),
            maximum_sector_weight=(
                "maximum_sector_weight",
                "max",
            ),
            mean_effective_positions=(
                "effective_positions",
                "mean",
            ),
            construction_one_way_turnover=(
                "one_way_turnover",
                "mean",
            ),
        )
        .rename(
            columns={
                "method": "strategy_name",
            }
        )
    )

    result = (
        performance.loc[
            :,
            [
                "strategy_name",
                "final_portfolio_value",
                "total_return",
                "cagr",
                "annualized_volatility",
                "sharpe_ratio",
                "sortino_ratio",
                "maximum_drawdown",
                "beta_vs_spy",
                "annualized_alpha_vs_spy",
                "excess_cagr",
            ],
        ]
        .merge(
            execution_summary.loc[
                :,
                [
                    "strategy_name",
                    "total_transaction_cost",
                    "total_traded_notional",
                    "mean_two_way_turnover",
                    "mean_one_way_turnover",
                ],
            ],
            on="strategy_name",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            construction,
            on="strategy_name",
            how="inner",
            validate="one_to_one",
        )
    )

    result["effective_cost_bps"] = np.where(
        result["total_traded_notional"].gt(0.0),
        (result["total_transaction_cost"] / result["total_traded_notional"] * 10_000.0),
        0.0,
    )

    baseline = result.loc[result["strategy_name"].eq(BASELINE_SCENARIO)]

    if len(baseline) != 1:
        raise ValueError("Ablation results require exactly one full-ensemble baseline.")

    baseline_cagr = float(baseline["cagr"].iloc[0])

    baseline_sharpe = float(baseline["sharpe_ratio"].iloc[0])

    baseline_excess = float(baseline["excess_cagr"].iloc[0])

    result["cagr_difference_vs_full"] = result["cagr"] - baseline_cagr

    result["sharpe_difference_vs_full"] = result["sharpe_ratio"] - baseline_sharpe

    result["excess_cagr_difference_vs_full"] = result["excess_cagr"] - baseline_excess

    result["is_baseline"] = result["strategy_name"].eq(BASELINE_SCENARIO)

    return result.sort_values(
        [
            "is_baseline",
            "strategy_name",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)


def _build_checks(
    signals: pd.DataFrame,
    weights: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Audit ensemble-component ablation outputs."""
    scenarios = set(signals["scenario"].unique())

    dates_per_scenario = signals.groupby("scenario")["as_of_date"].nunique()

    cross_sections = signals.groupby(
        [
            "scenario",
            "as_of_date",
        ]
    )["ticker"].nunique()

    weight_sums = weights.groupby(
        [
            "method",
            "as_of_date",
        ]
    )["weight"].sum()

    sector_weights = weights.groupby(
        [
            "method",
            "as_of_date",
            "sector",
        ],
        as_index=False,
    )["weight"].sum()

    max_weights = weights.groupby("method")["weight"].max()

    key_metrics = results[
        [
            "cagr",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "excess_cagr",
            "effective_cost_bps",
        ]
    ].to_numpy(dtype=float)

    checks = [
        (
            "expected_scenarios",
            int(scenarios != EXPECTED_SCENARIOS),
            "The analysis must contain the full ensemble and all three one-component ablations.",
        ),
        (
            "expected_oos_dates",
            int(dates_per_scenario.ne(77).sum()),
            "Every signal scenario must contain all 77 frozen OOS dates.",
        ),
        (
            "complete_cross_sections",
            int(cross_sections.ne(EXPECTED_CROSS_SECTION_SIZE).sum()),
            "Every signal scenario must retain the full 50-stock cross-section.",
        ),
        (
            "fully_invested",
            int(weight_sums.sub(1.0).abs().gt(1.0e-6).sum()),
            "Every ablation portfolio must sum to one.",
        ),
        (
            "long_only",
            int(weights["weight"].lt(-WEIGHT_TOLERANCE).sum()),
            "Ablation portfolios must remain long-only.",
        ),
        (
            "security_cap",
            int(max_weights.gt(MAX_SECURITY_WEIGHT + 1.0e-6).sum()),
            "Ablation portfolios must respect the 5% security cap.",
        ),
        (
            "sector_cap",
            int(sector_weights["weight"].gt(MAX_SECTOR_WEIGHT + 1.0e-6).sum()),
            "Ablation portfolios must respect the 25% sector cap.",
        ),
        (
            "finite_performance",
            int((~np.isfinite(key_metrics)).sum()),
            "Key ablation performance metrics must remain finite.",
        ),
        (
            "positive_final_values",
            int(results["final_portfolio_value"].le(0.0).sum()),
            "Every ablation backtest must retain positive final value.",
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
    results: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    """Build the ensemble-component ablation report."""
    display = results.loc[
        :,
        [
            "strategy_name",
            "is_baseline",
            "cagr",
            "cagr_difference_vs_full",
            "sharpe_ratio",
            "sharpe_difference_vs_full",
            "maximum_drawdown",
            "excess_cagr",
            "excess_cagr_difference_vs_full",
            "mean_one_way_turnover",
            "maximum_sector_weight",
            "effective_cost_bps",
        ],
    ]

    return "\n".join(
        [
            "# Ensemble Component Ablation",
            "",
            "## Methodology",
            "",
            (
                "- Uses the frozen out-of-sample predictions and the original "
                "fold-specific validation weights."
            ),
            (
                "- One ensemble component is removed at a time and the two "
                "remaining validation weights are renormalized."
            ),
            ("- Tested removals: technical composite, Elastic Net and LightGBM Ranker."),
            (
                "- Each ablated signal is converted to a monthly percentile "
                "ranking and then passed through the same score-weighted "
                "portfolio construction and advanced execution-cost model."
            ),
            (
                "- No test-period outcome is used to choose alternative weights "
                "or re-tune the remaining ensemble."
            ),
            (
                "- This experiment directly covers the no-LightGBM ablation. "
                "Feature-family ablations such as no fundamentals and no momentum "
                "require separate walk-forward retraining and are evaluated later."
            ),
            "",
            "## Results",
            "",
            _to_markdown(display),
            "",
            "## Readiness checks",
            "",
            _to_markdown(checks),
            "",
        ]
    )


def main() -> None:
    """Run portfolio-level ensemble component ablation analysis."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    required_paths = (
        PREDICTIONS_PATH,
        FINAL_SIGNAL_PATH,
        VALIDATION_WEIGHTS_PATH,
        MARKET_DATA_PATH,
        RISK_ESTIMATES_PATH,
        SPY_DATA_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    project_config = load_config()

    backtest_config = MVPBacktestConfig.from_mapping(
        project_config.get(
            "mvp_backtest",
            {},
        )
    )

    execution_config = ExecutionCostConfig.from_mapping(
        project_config.get(
            "execution_costs",
            {},
        )
    )

    evaluation_config = PerformanceEvaluationConfig.from_mapping(
        project_config.get(
            "mvp_performance_evaluation",
            {},
        )
    )

    predictions = _normalize_fold_ids(
        pd.read_parquet(PREDICTIONS_PATH),
        dataset_name="OOS predictions",
    )

    stored_final_signal = _normalize_fold_ids(
        pd.read_parquet(FINAL_SIGNAL_PATH),
        dataset_name="Stored final alpha signal",
    )

    csv_validation_weights = _normalize_fold_ids(
        pd.read_csv(VALIDATION_WEIGHTS_PATH),
        dataset_name="Ensemble validation weights",
    )

    validation_weights = _validation_weights_from_stored_signal(stored_final_signal)

    _audit_csv_weights(
        validation_weights,
        csv_validation_weights,
    )

    _validate_fold_id_alignment(
        predictions,
        validation_weights,
    )

    market_data = pd.read_parquet(MARKET_DATA_PATH)

    risk_estimates = pd.read_parquet(RISK_ESTIMATES_PATH)

    spy_data = pd.read_parquet(SPY_DATA_PATH)

    logger.info("Reconstructing frozen ensemble with production-persisted fold weights.")

    signals = _build_ablation_signals(
        predictions,
        validation_weights,
        stored_final_signal,
    )

    logger.info("Building score-weighted ablation portfolios.")

    weights = _build_portfolios(signals)

    diagnostics = compute_portfolio_diagnostics(
        weights,
        weight_tolerance=WEIGHT_TOLERANCE,
    )

    targets = _prepare_targets(weights)

    net_config = replace(
        backtest_config,
        transaction_cost_bps=0.0,
    )

    logger.info("Running ensemble-component ablation backtests.")

    outputs = run_mvp_backtest(
        targets,
        market_data,
        config=net_config,
        risk_estimates=risk_estimates,
        execution_cost_config=execution_config,
    )

    start_date = pd.to_datetime(outputs.daily_performance["date"]).min()

    end_date = pd.to_datetime(outputs.daily_performance["date"]).max()

    benchmark = build_buy_and_hold_benchmark(
        spy_data,
        strategy_name=(evaluation_config.benchmark_name),
        ticker=(evaluation_config.benchmark_ticker),
        start_date=start_date,
        end_date=end_date,
        initial_capital=(net_config.initial_capital),
        transaction_cost_bps=(execution_config.linear_cost_bps),
    )

    evaluation = evaluate_performance(
        outputs.daily_performance,
        benchmark,
        initial_capital=(net_config.initial_capital),
        config=evaluation_config,
    )

    performance = _strategy_performance(
        evaluation.performance_summary,
        benchmark_name=(evaluation_config.benchmark_name),
    )

    results = _build_results(
        performance,
        outputs.execution_summary,
        diagnostics,
    )

    checks = _build_checks(
        signals,
        weights,
        results,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    _write_parquet(
        signals,
        ABLATION_SIGNALS_PATH,
    )

    _write_parquet(
        weights,
        ABLATION_WEIGHTS_PATH,
    )

    _write_csv(
        results,
        RESULTS_PATH,
    )

    _write_csv(
        diagnostics,
        DIAGNOSTICS_PATH,
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
            results,
            checks,
        ),
        encoding="utf-8",
    )

    if failed_checks:
        raise ValueError(
            f"Ensemble-component ablation validation failed with {failed_checks} failed checks."
        )

    logger.info("Ensemble-component ablation analysis completed.")

    print()

    print("Institutional Quant Equity Research Platform")

    print("Ensemble component ablation")

    print("------------------------------------------------")

    print(f"scenarios: {results['strategy_name'].nunique()}")

    print()

    print("Ablation results:")

    print(
        results.loc[
            :,
            [
                "strategy_name",
                "is_baseline",
                "cagr",
                "cagr_difference_vs_full",
                "sharpe_ratio",
                "sharpe_difference_vs_full",
                "maximum_drawdown",
                "excess_cagr",
                "excess_cagr_difference_vs_full",
                "mean_one_way_turnover",
                "maximum_sector_weight",
                "effective_cost_bps",
            ],
        ].to_string(index=False)
    )

    print()

    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()

    print(f"Signals: {ABLATION_SIGNALS_PATH}")

    print(f"Weights: {ABLATION_WEIGHTS_PATH}")

    print(f"Results table: {RESULTS_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
