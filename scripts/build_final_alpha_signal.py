"""Build, evaluate and audit the final cross-sectional alpha signal."""

from __future__ import annotations

import logging

import pandas as pd

from quant_equity.config import (
    PROJECT_ROOT,
    REPORTS_DIR,
)
from quant_equity.logging_config import (
    configure_logging,
)
from quant_equity.models import (
    EnsembleConfig,
    build_ablation_candidates,
    build_component_scores,
    build_ensemble_candidates,
    build_final_alpha_signal,
    build_validation_weights,
    compute_component_correlations,
    compute_sector_signal_diagnostics,
    evaluate_model_predictions,
)

PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_panel.parquet"

FOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "walk_forward_folds.parquet"

ALL_MODEL_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "predictions_oos_all_models.parquet"
)

ELASTIC_HYPERPARAMETER_PATH = REPORTS_DIR / "tables" / "regularized_linear_hyperparameters.csv"

RANKER_HYPERPARAMETER_PATH = REPORTS_DIR / "tables" / "lightgbm_ranker_hyperparameters.csv"

FINAL_SIGNAL_PATH = PROJECT_ROOT / "data" / "processed" / "final_alpha_signal.parquet"

WEIGHTS_PATH = REPORTS_DIR / "tables" / "ensemble_validation_weights.csv"

CANDIDATE_MONTHLY_PATH = REPORTS_DIR / "tables" / "ensemble_candidate_monthly_metrics.csv"

CANDIDATE_SUMMARY_PATH = REPORTS_DIR / "tables" / "ensemble_candidate_summary.csv"

CORRELATION_PATH = REPORTS_DIR / "tables" / "ensemble_signal_correlations.csv"

ABLATION_PATH = REPORTS_DIR / "tables" / "ensemble_ablation_summary.csv"

SECTOR_DIAGNOSTICS_PATH = REPORTS_DIR / "tables" / "ensemble_sector_diagnostics.csv"

REPORT_PATH = REPORTS_DIR / "models" / "ensemble_signal.md"


def write_report(
    *,
    candidate_summary: pd.DataFrame,
    validation_weights: pd.DataFrame,
    correlations: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    sector_diagnostics: pd.DataFrame,
    final_signal: pd.DataFrame,
) -> None:
    """Write the consolidated alpha ensemble report."""
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weight_summary = validation_weights[
        [
            "composite_weight",
            "elastic_net_weight",
            "lightgbm_ranker_weight",
        ]
    ].describe()

    sector_summary = (
        sector_diagnostics.groupby(
            "sector",
            as_index=False,
        )
        .agg(
            mean_percentile_score=(
                "mean_percentile_score",
                "mean",
            ),
            mean_top_group_share=(
                "top_group_share",
                "mean",
            ),
        )
        .sort_values(
            "mean_percentile_score",
            ascending=False,
        )
    )

    lines = [
        "# Final Alpha Ensemble",
        "",
        "## Objective",
        "",
        (
            "Combine economically distinct cross-sectional "
            "signals into one stable monthly alpha ranking."
        ),
        "",
        "## Components",
        "",
        ("- Technical equal-weight composite"),
        ("- Elastic Net"),
        ("- LightGBM Ranker"),
        "",
        "## Weighting policy",
        "",
        (
            "Weights are calculated independently for every "
            "walk-forward fold using validation IC only."
        ),
        "",
        ("No test-period return is used to determine ensemble weights."),
        "",
        (
            "Validation weights are shrunk toward equal "
            "weights to reduce model-selection instability."
        ),
        "",
        "## Candidate ensemble comparison",
        "",
        "```text",
        candidate_summary.to_string(index=False),
        "```",
        "",
        "## Validation-weight distribution",
        "",
        "```text",
        weight_summary.to_string(),
        "```",
        "",
        "## Component correlations",
        "",
        "```text",
        correlations.to_string(index=False),
        "```",
        "",
        "## Marginal contribution diagnostics",
        "",
        "```text",
        ablation_summary.to_string(index=False),
        "```",
        "",
        "## Sector diagnostics",
        "",
        "```text",
        sector_summary.to_string(index=False),
        "```",
        "",
        "## Final signal coverage",
        "",
        "```text",
        (f"rows: {len(final_signal)}"),
        (f"dates: {final_signal['as_of_date'].nunique()}"),
        (f"tickers: {final_signal['ticker'].nunique()}"),
        "```",
        "",
        "## Downstream contract",
        "",
        (
            "The final alpha artifact intentionally excludes "
            "future realized targets. Portfolio construction "
            "must consume only the stored alpha score, ranking "
            "and contemporaneous metadata."
        ),
        "",
        "**FINAL ALPHA SIGNAL READY**",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Build and audit the final alpha ensemble."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    required_paths = (
        PANEL_PATH,
        FOLDS_PATH,
        ALL_MODEL_PREDICTIONS_PATH,
        ELASTIC_HYPERPARAMETER_PATH,
        RANKER_HYPERPARAMETER_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    panel = pd.read_parquet(PANEL_PATH)

    folds = pd.read_parquet(FOLDS_PATH)

    predictions = pd.read_parquet(ALL_MODEL_PREDICTIONS_PATH)

    elastic_hyperparameters = pd.read_csv(ELASTIC_HYPERPARAMETER_PATH)

    ranking_hyperparameters = pd.read_csv(RANKER_HYPERPARAMETER_PATH)

    config = EnsembleConfig()

    validation_weights = build_validation_weights(
        panel,
        folds,
        elastic_hyperparameters,
        ranking_hyperparameters,
        config=config,
    )

    component_scores = build_component_scores(
        predictions,
        config=config,
    )

    candidate_predictions = build_ensemble_candidates(
        component_scores,
        validation_weights,
    )

    (
        candidate_monthly,
        candidate_summary,
    ) = evaluate_model_predictions(candidate_predictions)

    final_signal = build_final_alpha_signal(
        component_scores,
        validation_weights,
    )

    correlations = compute_component_correlations(component_scores)

    ablation_predictions = build_ablation_candidates(
        component_scores,
        validation_weights,
    )

    (
        _,
        ablation_summary,
    ) = evaluate_model_predictions(ablation_predictions)

    sector_diagnostics = compute_sector_signal_diagnostics(final_signal)

    expected_rows = folds["test_rows"].sum()

    if len(final_signal) != expected_rows:
        raise ValueError("Final alpha coverage does not match the frozen OOS test universe.")

    duplicate_keys = int(
        final_signal.duplicated(
            [
                "as_of_date",
                "ticker",
            ]
        ).sum()
    )

    if duplicate_keys:
        raise ValueError("Final alpha signal contains duplicate keys.")

    FINAL_SIGNAL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    WEIGHTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_signal.to_parquet(
        FINAL_SIGNAL_PATH,
        index=False,
    )

    validation_weights.to_csv(
        WEIGHTS_PATH,
        index=False,
    )

    candidate_monthly.to_csv(
        CANDIDATE_MONTHLY_PATH,
        index=False,
    )

    candidate_summary.to_csv(
        CANDIDATE_SUMMARY_PATH,
        index=False,
    )

    correlations.to_csv(
        CORRELATION_PATH,
        index=False,
    )

    ablation_summary.to_csv(
        ABLATION_PATH,
        index=False,
    )

    sector_diagnostics.to_csv(
        SECTOR_DIAGNOSTICS_PATH,
        index=False,
    )

    write_report(
        candidate_summary=(candidate_summary),
        validation_weights=(validation_weights),
        correlations=correlations,
        ablation_summary=(ablation_summary),
        sector_diagnostics=(sector_diagnostics),
        final_signal=(final_signal),
    )

    logger.info("Final alpha ensemble completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("Final alpha ensemble")
    print("------------------------------------------------")

    print("Component models: 3")

    print(f"Folds: {folds['fold_id'].nunique()}")

    print(f"OOS dates: {final_signal['as_of_date'].nunique()}")

    print(f"Final signal rows: {len(final_signal)}")

    print(f"Duplicate signal keys: {duplicate_keys}")

    print(f"Candidate ensembles: {candidate_summary['model_name'].nunique()}")

    print()
    print("Candidate comparison:")

    print(candidate_summary.to_string(index=False))

    print()
    print("Average validation weights:")

    print(
        validation_weights[
            [
                "composite_weight",
                "elastic_net_weight",
                "lightgbm_ranker_weight",
            ]
        ]
        .mean()
        .to_string()
    )

    print()
    print("Component correlations:")

    print(correlations.to_string(index=False))

    print()
    print("Ablation diagnostics:")

    print(ablation_summary.to_string(index=False))

    print()
    print(f"Final alpha signal: {FINAL_SIGNAL_PATH}")

    print(f"Validation weights: {WEIGHTS_PATH}")

    print(f"Candidate summary: {CANDIDATE_SUMMARY_PATH}")

    print(f"Signal correlations: {CORRELATION_PATH}")

    print(f"Ablation summary: {ABLATION_PATH}")

    print(f"Sector diagnostics: {SECTOR_DIAGNOSTICS_PATH}")

    print(f"Report: {REPORT_PATH}")

    print()
    print("Final alpha signal: READY")


if __name__ == "__main__":
    main()
