"""Research utilities for quantitative equity signals."""

from quant_equity.research.technical_factors import (
    TechnicalFactorResearchConfig,
    TechnicalFactorResearchError,
    TechnicalFactorResearchResult,
    build_factor_research_panel,
    calculate_ic_summary,
    calculate_monthly_information_coefficients,
    calculate_quintile_research,
    calculate_selected_quantile_turnover,
    run_technical_factor_research,
)

__all__ = [
    "TechnicalFactorResearchConfig",
    "TechnicalFactorResearchError",
    "TechnicalFactorResearchResult",
    "build_factor_research_panel",
    "calculate_ic_summary",
    "calculate_monthly_information_coefficients",
    "calculate_quintile_research",
    "calculate_selected_quantile_turnover",
    "run_technical_factor_research",
    "LinearModelEvaluationConfig",
    "LinearModelEvaluationError",
    "LinearModelEvaluationOutputs",
    "calculate_monthly_model_metrics",
    "calculate_monthly_quintile_returns",
    "calculate_monthly_ranking_turnover",
    "evaluate_linear_model_predictions",
    "prepare_prediction_rankings",
    "summarize_coefficient_stability",
    "summarize_metrics_by_year",
    "summarize_model_metrics",
    "summarize_quintile_returns",
    "summarize_ranking_turnover",
]

from quant_equity.research.linear_model_evaluation import (
    LinearModelEvaluationConfig,
    LinearModelEvaluationError,
    LinearModelEvaluationOutputs,
    calculate_monthly_model_metrics,
    calculate_monthly_quintile_returns,
    calculate_monthly_ranking_turnover,
    evaluate_linear_model_predictions,
    prepare_prediction_rankings,
    summarize_coefficient_stability,
    summarize_metrics_by_year,
    summarize_model_metrics,
    summarize_quintile_returns,
    summarize_ranking_turnover,
)
