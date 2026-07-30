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
]
