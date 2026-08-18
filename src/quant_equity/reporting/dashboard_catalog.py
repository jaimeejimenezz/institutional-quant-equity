from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DashboardSource:
    source_id: str
    relative_path: str
    area: str
    description: str
    required_columns: tuple[str, ...]


DASHBOARD_SOURCES: dict[str, DashboardSource] = {
    "performance_net_daily": DashboardSource(
        source_id="performance_net_daily",
        relative_path="data/processed/backtest_all_methods_net_daily.parquet",
        area="overview",
        description="Net daily realized portfolio performance for all construction methods.",
        required_columns=(
            "date",
            "strategy_name",
            "portfolio_value",
            "daily_return",
            "transaction_cost",
            "one_way_turnover",
        ),
    ),
    "performance_gross_daily": DashboardSource(
        source_id="performance_gross_daily",
        relative_path="data/processed/backtest_all_methods_gross_daily.parquet",
        area="overview",
        description="Gross daily realized portfolio performance for all construction methods.",
        required_columns=("date", "strategy_name", "portfolio_value", "daily_return"),
    ),
    "performance_summary": DashboardSource(
        source_id="performance_summary",
        relative_path="reports/tables/all_methods_gross_net_comparison.csv",
        area="overview",
        description="Canonical gross-versus-net strategy comparison.",
        required_columns=(
            "strategy_name",
            "net_cagr",
            "net_sharpe_ratio",
            "net_sortino_ratio",
            "net_maximum_drawdown",
            "net_beta_vs_spy",
            "net_annualized_alpha_vs_spy",
            "total_transaction_cost",
        ),
    ),
    "benchmark_spy": DashboardSource(
        source_id="benchmark_spy",
        relative_path="data/processed/benchmark_spy_daily.parquet",
        area="overview",
        description="Raw SPY benchmark market series.",
        required_columns=("date", "ticker", "adjusted_close"),
    ),
    "alpha_signal": DashboardSource(
        source_id="alpha_signal",
        relative_path="data/processed/final_alpha_signal.parquet",
        area="alpha",
        description="Frozen full-model out-of-sample alpha signal.",
        required_columns=(
            "as_of_date",
            "ticker",
            "sector",
            "percentile_score",
            "rank",
            "composite_contribution",
            "elastic_net_contribution",
            "lightgbm_ranker_contribution",
        ),
    ),
    "modeling_panel": DashboardSource(
        source_id="modeling_panel",
        relative_path="data/processed/modeling_panel.parquet",
        area="alpha",
        description="Master point-in-time modeling panel used for feature drill-down.",
        required_columns=("as_of_date", "ticker", "sector", "sample_role"),
    ),
    "security_risk": DashboardSource(
        source_id="security_risk",
        relative_path="data/processed/risk_estimates.parquet",
        area="risk",
        description="Security-level ex-ante risk and liquidity estimates.",
        required_columns=(
            "as_of_date",
            "ticker",
            "sector",
            "annualized_volatility",
            "annualized_downside_volatility",
            "beta_vs_spy",
            "average_dollar_volume",
        ),
    ),
    "target_weights": DashboardSource(
        source_id="target_weights",
        relative_path="data/processed/target_weights_all_methods.parquet",
        area="portfolio",
        description="Canonical rebalance target weights for all portfolio methods.",
        required_columns=(
            "as_of_date",
            "ticker",
            "sector",
            "method",
            "weight",
            "previous_weight",
            "beta_vs_spy",
            "average_dollar_volume",
        ),
    ),
    "positions_daily": DashboardSource(
        source_id="positions_daily",
        relative_path="data/processed/positions_all_methods_net.parquet",
        area="portfolio",
        description="Realized daily holdings, weights, and drift.",
        required_columns=(
            "date",
            "strategy_name",
            "ticker",
            "sector",
            "market_value",
            "actual_weight",
            "target_weight",
            "weight_drift",
        ),
    ),
    "portfolio_diagnostics": DashboardSource(
        source_id="portfolio_diagnostics",
        relative_path="reports/tables/all_method_portfolio_diagnostics.csv",
        area="portfolio",
        description="Portfolio concentration, sector exposure, and turnover diagnostics.",
        required_columns=(
            "as_of_date",
            "method",
            "positions",
            "maximum_weight",
            "maximum_sector_weight",
            "effective_positions",
            "one_way_turnover",
        ),
    ),
    "portfolio_risk": DashboardSource(
        source_id="portfolio_risk",
        relative_path="reports/tables/all_method_risk_summary.csv",
        area="risk",
        description="Portfolio-level predicted risk, beta, concentration, and liquidity.",
        required_columns=(
            "as_of_date",
            "method",
            "predicted_volatility",
            "portfolio_beta_vs_spy",
            "effective_positions",
            "maximum_sector_weight",
            "maximum_liquidation_days",
        ),
    ),
    "risk_contributions": DashboardSource(
        source_id="risk_contributions",
        relative_path="reports/tables/reference_portfolio_risk_contributions.csv",
        area="risk",
        description="Security-level marginal and component risk contributions.",
        required_columns=(
            "as_of_date",
            "ticker",
            "sector",
            "weight",
            "risk_contribution_share",
            "liquidation_days",
        ),
    ),
    "covariance_diagnostics": DashboardSource(
        source_id="covariance_diagnostics",
        relative_path="reports/tables/covariance_diagnostics.csv",
        area="risk",
        description="Rolling shrinkage covariance diagnostics.",
        required_columns=(
            "as_of_date",
            "assets",
            "observations",
            "shrinkage",
            "shrinkage_condition_number",
            "mean_pairwise_correlation",
        ),
    ),
    "trades": DashboardSource(
        source_id="trades",
        relative_path="data/processed/trades_all_methods_net.parquet",
        area="execution",
        description="Executed rebalance trades and modeled implementation costs.",
        required_columns=(
            "signal_date",
            "execution_date",
            "strategy_name",
            "ticker",
            "side",
            "trade_notional",
            "total_execution_cost",
            "effective_cost_bps",
            "order_adv_fraction",
        ),
    ),
    "execution_summary": DashboardSource(
        source_id="execution_summary",
        relative_path="reports/tables/all_methods_execution_summary.csv",
        area="execution",
        description="Strategy-level realized execution summary.",
        required_columns=(
            "strategy_name",
            "start_date",
            "end_date",
            "rebalances",
            "final_portfolio_value",
            "total_transaction_cost",
            "mean_one_way_turnover",
        ),
    ),
    "execution_cost_components": DashboardSource(
        source_id="execution_cost_components",
        relative_path="reports/tables/all_methods_execution_cost_components.csv",
        area="execution",
        description="Commission, spread, slippage, and market-impact decomposition.",
        required_columns=(
            "strategy_name",
            "commission_cost",
            "spread_cost",
            "slippage_cost",
            "market_impact_cost",
            "total_execution_cost",
            "effective_cost_bps",
        ),
    ),
    "cost_sensitivity": DashboardSource(
        source_id="cost_sensitivity",
        relative_path="reports/tables/transaction_cost_sensitivity.csv",
        area="execution",
        description="Performance sensitivity to transaction-cost assumptions.",
        required_columns=(
            "scenario",
            "strategy_name",
            "cagr",
            "sharpe_ratio",
            "maximum_drawdown",
            "total_transaction_cost",
        ),
    ),
    "capacity": DashboardSource(
        source_id="capacity",
        relative_path="reports/tables/capacity_analysis.csv",
        area="execution",
        description="Capital-capacity and market-impact sensitivity.",
        required_columns=(
            "capital",
            "strategy_name",
            "net_cagr",
            "net_sharpe_ratio",
            "effective_cost_bps",
            "maximum_order_adv_fraction",
        ),
    ),
    "model_summary": DashboardSource(
        source_id="model_summary",
        relative_path="reports/tables/model_comparison_summary.csv",
        area="research",
        description="Out-of-sample model comparison.",
        required_columns=(
            "model_name",
            "months",
            "mean_ic",
            "annualized_ic_ir",
            "positive_ic_ratio",
            "mean_top_bottom_spread",
            "mean_top_quintile_precision",
        ),
    ),
    "model_monthly": DashboardSource(
        source_id="model_monthly",
        relative_path="reports/tables/model_comparison_monthly_metrics.csv",
        area="research",
        description="Monthly out-of-sample model diagnostics.",
        required_columns=(
            "model_name",
            "as_of_date",
            "ic",
            "top_bottom_spread",
            "top_quintile_precision",
            "top_quintile_turnover",
        ),
    ),
    "model_yearly": DashboardSource(
        source_id="model_yearly",
        relative_path="reports/tables/model_yearly_stability.csv",
        area="research",
        description="Annual model stability diagnostics.",
        required_columns=("model_name", "year", "mean_ic", "positive_ic_ratio"),
    ),
    "model_sector": DashboardSource(
        source_id="model_sector",
        relative_path="reports/tables/model_sector_stability.csv",
        area="research",
        description="Sector-level model stability diagnostics.",
        required_columns=("model_name", "sector", "mean_sector_ic", "positive_ic_ratio"),
    ),
    "feature_importance": DashboardSource(
        source_id="feature_importance",
        relative_path="reports/tables/lightgbm_ranker_feature_importance_summary.csv",
        area="research",
        description="Aggregated LightGBM ranker feature importance.",
        required_columns=("feature", "mean_gain", "mean_gain_share", "folds_used"),
    ),
    "ensemble_summary": DashboardSource(
        source_id="ensemble_summary",
        relative_path="reports/tables/ensemble_candidate_summary.csv",
        area="research",
        description="Candidate ensemble comparison.",
        required_columns=("model_name", "mean_ic", "annualized_ic_ir"),
    ),
    "robustness_inventory": DashboardSource(
        source_id="robustness_inventory",
        relative_path="reports/tables/robustness_evaluation_check_inventory.csv",
        area="robustness",
        description="Final robustness check-suite inventory.",
        required_columns=(
            "suite",
            "category",
            "artifact_exists",
            "checks",
            "passed_checks",
            "failed_checks",
            "suite_status",
        ),
    ),
    "robustness_coverage": DashboardSource(
        source_id="robustness_coverage",
        relative_path="reports/tables/robustness_evaluation_coverage.csv",
        area="robustness",
        description="Final robustness coverage and documented limitations.",
        required_columns=("dimension", "category", "status", "note"),
    ),
    "bootstrap_strategy": DashboardSource(
        source_id="bootstrap_strategy",
        relative_path="reports/tables/bootstrap_strategy_summary.csv",
        area="robustness",
        description="Monthly portfolio-return bootstrap evidence.",
        required_columns=(
            "strategy_name",
            "observed_annualized_return",
            "observed_sharpe",
            "annualized_return_ci_lower",
            "annualized_return_ci_upper",
            "sharpe_ci_lower",
            "sharpe_ci_upper",
        ),
    ),
    "signal_bootstrap": DashboardSource(
        source_id="signal_bootstrap",
        relative_path="reports/tables/robustness_final_signal_bootstrap.csv",
        area="robustness",
        description="Block-bootstrap evidence for the frozen final alpha signal.",
        required_columns=(
            "months",
            "observed_mean_ic",
            "mean_ic_ci_lower",
            "mean_ic_ci_upper",
            "probability_mean_ic_positive",
            "observed_mean_top_bottom_spread",
        ),
    ),
    "feature_family_ablation": DashboardSource(
        source_id="feature_family_ablation",
        relative_path=("reports/tables/feature_family_ablation/official_predictive_comparison.csv"),
        area="robustness",
        description="Official predictive feature-family ablation.",
        required_columns=(
            "scenario",
            "model_name",
            "mean_ic",
            "annualized_ic_ir",
            "mean_top_bottom_spread",
        ),
    ),
    "economic_ablation": DashboardSource(
        source_id="economic_ablation",
        relative_path="reports/tables/feature_family_ablation/economic_comparison.csv",
        area="robustness",
        description="Economic feature-family ablation under the frozen construction rule.",
        required_columns=(
            "strategy_name",
            "cagr",
            "sharpe_ratio",
            "maximum_drawdown",
            "cagr_difference_vs_full",
        ),
    ),
    "construction_ablation": DashboardSource(
        source_id="construction_ablation",
        relative_path="reports/tables/robustness_portfolio_construction_ablation.csv",
        area="robustness",
        description="Portfolio-construction ablation evidence.",
        required_columns=(
            "strategy_name",
            "experiment",
            "cagr",
            "sharpe_ratio",
            "mean_one_way_turnover",
        ),
    ),
    "horizon_sensitivity": DashboardSource(
        source_id="horizon_sensitivity",
        relative_path="reports/tables/robustness_prediction_horizon_summary.csv",
        area="robustness",
        description="10/21/42-session predictive-horizon sensitivity.",
        required_columns=("horizon_sessions", "mean_ic", "annualized_ic_ir"),
    ),
    "rebalance_sensitivity": DashboardSource(
        source_id="rebalance_sensitivity",
        relative_path="reports/tables/robustness_rebalance_frequency.csv",
        area="robustness",
        description="Monthly-versus-quarterly rebalance sensitivity.",
        required_columns=(
            "strategy_name",
            "monthly_cagr",
            "quarterly_cagr",
            "quarterly_minus_monthly_cagr",
        ),
    ),
    "rolling_window_sensitivity": DashboardSource(
        source_id="rolling_window_sensitivity",
        relative_path="reports/tables/robustness_rolling_window_summary.csv",
        area="robustness",
        description="Rolling evaluation-window robustness.",
        required_columns=("window_months", "strategy_name", "mean_cagr", "mean_sharpe"),
    ),
    "universe_exclusions": DashboardSource(
        source_id="universe_exclusions",
        relative_path="reports/tables/robustness_universe_exclusion_results.csv",
        area="robustness",
        description="Frozen-universe exclusion robustness.",
        required_columns=(
            "strategy_name",
            "scenario_type",
            "excluded_group",
            "cagr",
            "sharpe_ratio",
        ),
    ),
    "regime_performance": DashboardSource(
        source_id="regime_performance",
        relative_path="reports/tables/robustness_regime_performance.csv",
        area="robustness",
        description="Market-regime strategy performance.",
        required_columns=(
            "regime",
            "strategy_name",
            "start_date",
            "end_date",
            "cagr",
            "sharpe_ratio",
            "maximum_drawdown",
        ),
    ),
    "leakage_checks": DashboardSource(
        source_id="leakage_checks",
        relative_path="reports/tables/modeling_panel_leakage_checks.csv",
        area="data_quality",
        description="Modeling-panel leakage controls.",
        required_columns=("check", "status", "violations", "description"),
    ),
    "panel_readiness": DashboardSource(
        source_id="panel_readiness",
        relative_path="reports/tables/modeling_panel_readiness_checks.csv",
        area="data_quality",
        description="Modeling-panel readiness controls.",
        required_columns=("check", "status", "violations", "description"),
    ),
    "walk_forward_readiness": DashboardSource(
        source_id="walk_forward_readiness",
        relative_path="reports/tables/walk_forward_readiness_checks.csv",
        area="data_quality",
        description="Walk-forward readiness controls.",
        required_columns=("check", "status", "violations", "description"),
    ),
    "risk_checks": DashboardSource(
        source_id="risk_checks",
        relative_path="reports/tables/risk_estimate_checks.csv",
        area="data_quality",
        description="Security-level risk-estimate controls.",
        required_columns=("check", "status", "violations", "description"),
    ),
    "covariance_checks": DashboardSource(
        source_id="covariance_checks",
        relative_path="reports/tables/covariance_checks.csv",
        area="data_quality",
        description="Covariance-model controls.",
        required_columns=("check", "status", "violations", "description"),
    ),
    "portfolio_checks": DashboardSource(
        source_id="portfolio_checks",
        relative_path="reports/tables/all_method_portfolio_checks.csv",
        area="data_quality",
        description="Portfolio-construction controls.",
        required_columns=("check", "status", "violations", "description"),
    ),
    "execution_checks": DashboardSource(
        source_id="execution_checks",
        relative_path="reports/tables/all_methods_execution_checks.csv",
        area="data_quality",
        description="Execution-engine controls.",
        required_columns=("check", "status", "violations", "description"),
    ),
}

STRATEGY_ORDER: tuple[str, ...] = (
    "score_weighted",
    "top_n_equal_weight",
    "median_mad_de",
    "alpha_risk_turnover",
    "cvar",
)

STRATEGY_LABELS: dict[str, str] = {
    "score_weighted": "Score Weighted",
    "top_n_equal_weight": "Top-N Equal Weight",
    "median_mad_de": "Median-MAD DE",
    "alpha_risk_turnover": "Alpha-Risk-Turnover",
    "cvar": "CVaR",
}

DEFAULT_STRATEGY = "score_weighted"


def get_source(source_id: str) -> DashboardSource:
    try:
        return DASHBOARD_SOURCES[source_id]
    except KeyError as exc:
        raise KeyError(f"Unknown dashboard source: {source_id}") from exc


def source_path(source_id: str) -> Path:
    return PROJECT_ROOT / get_source(source_id).relative_path


def sources_for_area(area: str) -> tuple[DashboardSource, ...]:
    return tuple(source for source in DASHBOARD_SOURCES.values() if source.area == area)


def strategy_label(strategy_name: str) -> str:
    return STRATEGY_LABELS.get(strategy_name, strategy_name.replace("_", " ").title())
