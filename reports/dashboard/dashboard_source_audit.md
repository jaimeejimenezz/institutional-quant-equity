# Dashboard source audit

Read-only inventory of tabular artifacts available to the presentation layer.

- Tabular artifacts inspected: **336**
- Artifacts with read/schema errors: **0**
- Candidate dashboard sources: **181**

## Candidate sources by dashboard area

### overview

- `data/processed/backtest_all_methods_gross_daily.parquet` — 8050 rows × 19 cols; dates 2020-01-31T00:00:00 → 2026-06-30T00:00:00
- `data/processed/backtest_all_methods_net_daily.parquet` — 8050 rows × 19 cols; dates 2020-01-31T00:00:00 → 2026-06-30T00:00:00
- `data/processed/benchmark_spy_daily.parquet` — 3159 rows × 8 cols; dates 2014-01-02T00:00:00 → 2026-07-27T00:00:00
- `reports/tables/all_methods_gross_net_comparison.csv` — 5 rows × 35 cols; dates 1970-01-01T00:00:00.000000077 → 1970-01-01T00:00:00.000000077
- `reports/tables/mvp_performance_summary.csv` — 6 rows × 27 cols; dates 1970-01-01T00:00:00 → 2026-06-30T00:00:00

### stock_ranking

- `data/processed/final_alpha_signal.parquet` — 3850 rows × 14 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/modeling_panel.parquet` — 7550 rows × 108 cols; dates 2014-01-31T00:00:00 → 2026-07-27T00:00:00
- `data/processed/risk_estimates.parquet` — 3850 rows × 18 cols; dates 2019-02-01T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_fundamentals_final_alpha_signal.parquet` — 3850 rows × 14 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_momentum_final_alpha_signal.parquet` — 3850 rows × 14 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/technical_modeling_panel.parquet` — 7450 rows × 22 cols; dates 2014-01-31T00:00:00 → 2026-06-30T00:00:00
- `reports/tables/modeling_panel_data_dictionary.csv` — 108 rows × 18 cols; dates 2014-01-31T00:00:00 → 2026-07-27T00:00:00
- `reports/tables/modeling_panel_feature_coverage.csv` — 91 rows × 11 cols; dates 2014-01-31T00:00:00 → 2026-07-27T00:00:00
- `reports/tables/modeling_panel_leakage_checks.csv` — 21 rows × 4 cols
- `reports/tables/modeling_panel_readiness_checks.csv` — 17 rows × 4 cols

### portfolio

- `data/processed/positions_all_methods_net.parquet` — 186898 rows × 12 cols; dates 2020-01-31T00:00:00 → 2026-06-30T00:00:00
- `data/processed/target_weights_all_methods.parquet` — 9625 rows × 14 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/all_method_portfolio_diagnostics.csv` — 385 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/baseline_portfolio_diagnostics.csv` — 154 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/baseline_portfolio_risk_summary.csv` — 154 rows × 18 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/core_portfolio_diagnostics.csv` — 231 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/core_portfolio_risk_summary.csv` — 231 rows × 18 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/cvar_portfolio_diagnostics.csv` — 77 rows × 18 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/feature_family_ablation/economic_portfolio_diagnostics.csv` — 231 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/four_method_portfolio_diagnostics.csv` — 308 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/median_mad_portfolio_diagnostics.csv` — 77 rows × 23 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/portfolio_risk_checks.csv` — 10 rows × 4 cols
- `reports/tables/reference_portfolio_risk_contributions.csv` — 1540 rows × 13 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/reference_portfolio_risk_summary.csv` — 77 rows × 17 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00

### model_research

- `data/processed/predictions_oos_model_comparison.parquet` — 23100 rows × 13 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_fundamentals_ensemble_ablation_candidates.parquet` — 11550 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_fundamentals_ensemble_candidates.parquet` — 11550 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_fundamentals_ensemble_predictions.parquet` — 11550 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_momentum_ensemble_ablation_candidates.parquet` — 11550 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_momentum_ensemble_candidates.parquet` — 11550 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_momentum_ensemble_predictions.parquet` — 11550 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness_ensemble_ablation_signals.parquet` — 15400 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness_ensemble_ablation_weights.parquet` — 7700 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/ensemble_ablation_summary.csv` — 3 rows × 11 cols
- `reports/tables/ensemble_candidate_monthly_metrics.csv` — 231 rows × 12 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/ensemble_candidate_summary.csv` — 3 rows × 11 cols
- `reports/tables/ensemble_sector_diagnostics.csv` — 847 rows × 6 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/ensemble_signal_correlations.csv` — 3 rows × 7 cols
- `reports/tables/ensemble_validation_weights.csv` — 77 rows × 9 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/factor_ic.csv` — 19 rows × 11 cols
- `reports/tables/factor_ic_by_sector.csv` — 144 rows × 8 cols
- `reports/tables/factor_ic_by_sector_monthly.csv` — 15048 rows × 5 cols; dates 2014-01-31T00:00:00 → 2019-12-31T00:00:00
- `reports/tables/factor_ic_by_subperiod.csv` — 36 rows × 11 cols; dates 2016-12-31T00:00:00 → 2019-12-31T00:00:00
- `reports/tables/factor_ic_by_year.csv` — 107 rows × 11 cols; dates 2014-12-31T00:00:00 → 2019-12-31T00:00:00
- `reports/tables/factor_ic_monthly.csv` — 1368 rows × 4 cols; dates 2014-01-31T00:00:00 → 2019-12-31T00:00:00
- `reports/tables/factor_quintiles.csv` — 90 rows × 7 cols
- `reports/tables/factor_quintiles_monthly.csv` — 6185 rows × 7 cols; dates 2014-01-31T00:00:00 → 2019-12-31T00:00:00
- `reports/tables/feature_family_ablation/no_fundamentals_ensemble_checks.csv` — 14 rows × 4 cols
- `reports/tables/feature_family_ablation/no_fundamentals_lightgbm_ranker_feature_importance.csv` — 1463 rows × 6 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/feature_family_ablation/no_momentum_ensemble_checks.csv` — 14 rows × 4 cols
- `reports/tables/feature_family_ablation/no_momentum_lightgbm_ranker_feature_importance.csv` — 6545 rows × 6 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/lightgbm_feature_importance.csv` — 7007 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/lightgbm_feature_importance_summary.csv` — 91 rows × 6 cols
- `reports/tables/lightgbm_ranker_feature_importance.csv` — 7007 rows × 6 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/lightgbm_ranker_feature_importance_summary.csv` — 91 rows × 6 cols
- `reports/tables/linear_model_yearly.csv` — 35 rows × 12 cols
- `reports/tables/model_comparison_monthly_metrics.csv` — 539 rows × 12 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/model_comparison_summary.csv` — 7 rows × 11 cols
- `reports/tables/model_sector_stability.csv` — 48 rows × 7 cols
- `reports/tables/model_yearly_stability.csv` — 42 rows × 10 cols
- `reports/tables/robustness_ensemble_component_ablation.csv` — 4 rows × 25 cols
- `reports/tables/robustness_ensemble_component_ablation_checks.csv` — 9 rows × 4 cols
- `reports/tables/robustness_ensemble_component_ablation_diagnostics.csv` — 308 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00

### risk

- `data/processed/risk_estimates.parquet` — 3850 rows × 18 cols; dates 2019-02-01T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/all_method_risk_summary.csv` — 385 rows × 18 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/baseline_portfolio_risk_summary.csv` — 154 rows × 18 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/core_portfolio_risk_summary.csv` — 231 rows × 18 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/covariance_diagnostics.csv` — 77 rows × 14 cols; dates 2019-02-01T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/four_method_risk_summary.csv` — 308 rows × 18 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/portfolio_risk_checks.csv` — 10 rows × 4 cols
- `reports/tables/reference_portfolio_risk_contributions.csv` — 1540 rows × 13 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/reference_portfolio_risk_summary.csv` — 77 rows × 17 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00

### trades_costs

- `data/processed/trades_all_methods_net.parquet` — 10804 rows × 26 cols; dates 2020-01-31T00:00:00 → 2026-06-01T00:00:00
- `reports/tables/all_methods_execution_checks.csv` — 8 rows × 4 cols
- `reports/tables/all_methods_execution_cost_components.csv` — 5 rows × 10 cols
- `reports/tables/all_methods_execution_summary.csv` — 5 rows × 17 cols; dates 1970-01-01T00:00:00.000000077 → 2026-06-30T00:00:00
- `reports/tables/all_methods_gross_net_comparison.csv` — 5 rows × 35 cols; dates 1970-01-01T00:00:00.000000077 → 1970-01-01T00:00:00.000000077
- `reports/tables/capacity_analysis.csv` — 20 rows × 28 cols
- `reports/tables/capacity_checks.csv` — 8 rows × 4 cols
- `reports/tables/capacity_cost_components.csv` — 20 rows × 13 cols
- `reports/tables/feature_family_ablation/economic_execution_summary.csv` — 3 rows × 17 cols; dates 1970-01-01T00:00:00.000000077 → 2026-06-30T00:00:00
- `reports/tables/mvp_execution_schedule.csv` — 77 rows × 4 cols; dates 2020-01-31T00:00:00 → 2026-06-30T00:00:00
- `reports/tables/mvp_execution_summary.csv` — 5 rows × 17 cols; dates 1970-01-01T00:00:00.000000077 → 2026-06-30T00:00:00
- `reports/tables/transaction_cost_sensitivity.csv` — 30 rows × 19 cols
- `reports/tables/transaction_cost_sensitivity_checks.csv` — 7 rows × 4 cols

### robustness

- `data/processed/robustness/feature_family_ablation/feature_family_economic_combined_daily.parquet` — 6440 rows × 19 cols; dates 2020-01-31T00:00:00 → 2026-06-30T00:00:00
- `data/processed/robustness/feature_family_ablation/feature_family_economic_daily_performance.parquet` — 4830 rows × 19 cols; dates 2020-01-31T00:00:00 → 2026-06-30T00:00:00
- `data/processed/robustness/feature_family_ablation/feature_family_economic_drawdowns.parquet` — 6440 rows × 4 cols; dates 2020-02-03T00:00:00 → 2026-06-30T00:00:00
- `data/processed/robustness/feature_family_ablation/feature_family_economic_signals.parquet` — 11550 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/feature_family_economic_weights.parquet` — 5775 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_fundamentals_component_scores.parquet` — 3850 rows × 12 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_fundamentals_ensemble_ablation_candidates.parquet` — 11550 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_fundamentals_ensemble_candidates.parquet` — 11550 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_fundamentals_ensemble_predictions.parquet` — 11550 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_fundamentals_final_alpha_signal.parquet` — 3850 rows × 14 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_fundamentals_lightgbm_ranker_predictions.parquet` — 3850 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_fundamentals_regularized_linear_predictions.parquet` — 7700 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_momentum_component_scores.parquet` — 3850 rows × 12 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_momentum_ensemble_ablation_candidates.parquet` — 11550 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_momentum_ensemble_candidates.parquet` — 11550 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_momentum_ensemble_predictions.parquet` — 11550 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_momentum_final_alpha_signal.parquet` — 3850 rows × 14 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_momentum_lightgbm_ranker_predictions.parquet` — 3850 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness/feature_family_ablation/no_momentum_regularized_linear_predictions.parquet` — 7700 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness_ensemble_ablation_signals.parquet` — 15400 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness_ensemble_ablation_weights.parquet` — 7700 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness_portfolio_construction_ablation_weights.parquet` — 7700 rows × 13 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness_portfolio_parameter_weights.parquet` — 17325 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `data/processed/robustness_universe_exclusion_weights.parquet` — 25025 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/bootstrap_monthly_returns.csv` — 77 rows × 7 cols
- `reports/tables/bootstrap_pairwise_comparison.csv` — 10 rows × 10 cols
- `reports/tables/bootstrap_rank_stability.csv` — 5 rows × 3 cols
- `reports/tables/bootstrap_robustness_checks.csv` — 8 rows × 4 cols
- `reports/tables/bootstrap_strategy_summary.csv` — 5 rows × 25 cols
- `reports/tables/ensemble_ablation_summary.csv` — 3 rows × 11 cols
- `reports/tables/feature_family_ablation/economic_bootstrap_checks.csv` — 6 rows × 4 cols
- `reports/tables/feature_family_ablation/economic_checks.csv` — 11 rows × 4 cols
- `reports/tables/feature_family_ablation/economic_comparison.csv` — 3 rows × 25 cols
- `reports/tables/feature_family_ablation/economic_cost_deltas_vs_full.csv` — 2 rows × 9 cols
- `reports/tables/feature_family_ablation/economic_execution_summary.csv` — 3 rows × 17 cols; dates 1970-01-01T00:00:00.000000077 → 2026-06-30T00:00:00
- `reports/tables/feature_family_ablation/economic_monthly_returns.csv` — 308 rows × 3 cols
- `reports/tables/feature_family_ablation/economic_paired_bootstrap.csv` — 2 rows × 17 cols; dates 1970-01-01T00:00:00 → 1970-01-01T00:00:00
- `reports/tables/feature_family_ablation/economic_portfolio_diagnostics.csv` — 231 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/feature_family_ablation/economic_yearly_stability.csv` — 14 rows × 9 cols; dates 1970-01-01T00:00:00 → 1970-01-01T00:00:00
- `reports/tables/feature_family_ablation/economic_yearly_summary.csv` — 28 rows × 9 cols
- `reports/tables/feature_family_ablation/no_fundamentals_ensemble_checks.csv` — 14 rows × 4 cols
- `reports/tables/feature_family_ablation/no_fundamentals_feature_columns.csv` — 19 rows × 1 cols
- `reports/tables/feature_family_ablation/no_fundamentals_lightgbm_ranker_feature_importance.csv` — 1463 rows × 6 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/feature_family_ablation/no_fundamentals_lightgbm_ranker_hyperparameters.csv` — 308 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/feature_family_ablation/no_fundamentals_regularized_linear_coefficients.csv` — 2926 rows × 9 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/feature_family_ablation/no_fundamentals_regularized_linear_hyperparameters.csv` — 1001 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/feature_family_ablation/no_fundamentals_retraining_checks.csv` — 9 rows × 4 cols
- `reports/tables/feature_family_ablation/no_fundamentals_validation_weights.csv` — 77 rows × 9 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/feature_family_ablation/no_momentum_ensemble_checks.csv` — 14 rows × 4 cols
- `reports/tables/feature_family_ablation/no_momentum_feature_columns.csv` — 85 rows × 1 cols
- `reports/tables/feature_family_ablation/no_momentum_lightgbm_ranker_feature_importance.csv` — 6545 rows × 6 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/feature_family_ablation/no_momentum_lightgbm_ranker_hyperparameters.csv` — 308 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/feature_family_ablation/no_momentum_regularized_linear_coefficients.csv` — 13090 rows × 9 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/feature_family_ablation/no_momentum_regularized_linear_hyperparameters.csv` — 1001 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/feature_family_ablation/no_momentum_retraining_checks.csv` — 9 rows × 4 cols
- `reports/tables/feature_family_ablation/no_momentum_validation_weights.csv` — 77 rows × 9 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/feature_family_ablation/official_predictive_comparison.csv` — 3 rows × 13 cols
- `reports/tables/feature_family_ablation/official_predictive_deltas_vs_full.csv` — 2 rows × 8 cols
- `reports/tables/feature_family_ablation/official_predictive_frozen_full_checks.csv` — 5 rows × 6 cols
- `reports/tables/feature_family_ablation/official_predictive_full_candidate_match.csv` — 3 rows × 8 cols
- `reports/tables/feature_family_ablation/official_predictive_key_checks.csv` — 3 rows × 3 cols
- `reports/tables/feature_family_ablation/official_predictive_monthly_metrics.csv` — 231 rows × 13 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/feature_family_ablation/official_predictive_yearly_metrics.csv` — 21 rows × 8 cols
- `reports/tables/feature_family_ablation/predictive_comparison.csv` — 3 rows × 10 cols; dates 1970-01-01T00:00:00.000000077 → 1970-01-01T00:00:00.000000077
- `reports/tables/feature_family_ablation/predictive_deltas_vs_full.csv` — 2 rows × 8 cols
- `reports/tables/feature_family_ablation/predictive_formula_checks.csv` — 5 rows × 6 cols
- `reports/tables/feature_family_ablation/predictive_key_checks.csv` — 3 rows × 3 cols
- `reports/tables/feature_family_ablation/predictive_monthly_metrics.csv` — 231 rows × 6 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/feature_family_ablation/predictive_yearly_metrics.csv` — 21 rows × 8 cols
- `reports/tables/robustness_conditional_months.csv` — 20 rows × 10 cols
- `reports/tables/robustness_ensemble_component_ablation.csv` — 4 rows × 25 cols
- `reports/tables/robustness_ensemble_component_ablation_checks.csv` — 9 rows × 4 cols
- `reports/tables/robustness_ensemble_component_ablation_diagnostics.csv` — 308 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/robustness_evaluation_check_inventory.csv` — 17 rows × 8 cols
- `reports/tables/robustness_evaluation_coverage.csv` — 13 rows × 4 cols
- `reports/tables/robustness_feature_family_contract.csv` — 91 rows × 9 cols
- `reports/tables/robustness_feature_family_contract_checks.csv` — 11 rows × 4 cols
- `reports/tables/robustness_final_signal_bootstrap.csv` — 1 rows × 21 cols
- `reports/tables/robustness_final_signal_checks.csv` — 8 rows × 4 cols
- `reports/tables/robustness_final_signal_monthly.csv` — 77 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/robustness_final_signal_sector.csv` — 8 rows × 8 cols
- `reports/tables/robustness_final_signal_yearly.csv` — 7 rows × 10 cols
- `reports/tables/robustness_portfolio_construction_ablation.csv` — 4 rows × 36 cols
- `reports/tables/robustness_portfolio_construction_ablation_checks.csv` — 10 rows × 4 cols
- `reports/tables/robustness_portfolio_construction_ablation_diagnostics.csv` — 308 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/robustness_portfolio_construction_optimizer_diagnostics.csv` — 154 rows × 13 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/robustness_portfolio_parameter_checks.csv` — 9 rows × 4 cols
- `reports/tables/robustness_portfolio_parameter_diagnostics.csv` — 693 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/robustness_portfolio_parameter_sensitivity.csv` — 9 rows × 30 cols
- `reports/tables/robustness_prediction_horizon_checks.csv` — 8 rows × 4 cols
- `reports/tables/robustness_prediction_horizon_monthly.csv` — 228 rows × 8 cols; dates 2020-01-31T00:00:00 → 2026-04-30T00:00:00
- `reports/tables/robustness_prediction_horizon_summary.csv` — 3 rows × 21 cols
- `reports/tables/robustness_prediction_horizon_yearly.csv` — 21 rows × 8 cols
- `reports/tables/robustness_rebalance_frequency.csv` — 5 rows × 37 cols; dates 1970-01-01T00:00:00.000000025 → 1970-01-01T00:00:00.000000073
- `reports/tables/robustness_rebalance_frequency_checks.csv` — 10 rows × 4 cols
- `reports/tables/robustness_regime_performance.csv` — 30 rows × 31 cols; dates 1970-01-01T00:00:00 → 2026-06-30T00:00:00
- `reports/tables/robustness_rolling_window_checks.csv` — 8 rows × 4 cols
- `reports/tables/robustness_rolling_window_results.csv` — 810 rows × 31 cols; dates 1970-01-01T00:00:00 → 2026-06-30T00:00:00
- `reports/tables/robustness_rolling_window_summary.csv` — 15 rows × 15 cols
- `reports/tables/robustness_temporal_checks.csv` — 8 rows × 4 cols
- `reports/tables/robustness_universe_exclusion_checks.csv` — 10 rows × 4 cols
- `reports/tables/robustness_universe_exclusion_diagnostics.csv` — 1001 rows × 11 cols; dates 2020-01-31T00:00:00 → 2026-05-29T00:00:00
- `reports/tables/robustness_universe_exclusion_results.csv` — 13 rows × 31 cols
- `reports/tables/robustness_yearly_performance.csv` — 35 rows × 31 cols; dates 1970-01-01T00:00:00 → 2026-06-30T00:00:00

### data_quality

- `reports/tables/all_method_portfolio_checks.csv` — 27 rows × 5 cols
- `reports/tables/all_methods_execution_checks.csv` — 8 rows × 4 cols
- `reports/tables/baseline_portfolio_checks.csv` — 7 rows × 4 cols
- `reports/tables/bootstrap_robustness_checks.csv` — 8 rows × 4 cols
- `reports/tables/capacity_checks.csv` — 8 rows × 4 cols
- `reports/tables/core_portfolio_checks.csv` — 10 rows × 4 cols
- `reports/tables/covariance_checks.csv` — 11 rows × 4 cols
- `reports/tables/feature_family_ablation/economic_bootstrap_checks.csv` — 6 rows × 4 cols
- `reports/tables/feature_family_ablation/economic_checks.csv` — 11 rows × 4 cols
- `reports/tables/feature_family_ablation/no_fundamentals_ensemble_checks.csv` — 14 rows × 4 cols
- `reports/tables/feature_family_ablation/no_fundamentals_retraining_checks.csv` — 9 rows × 4 cols
- `reports/tables/feature_family_ablation/no_momentum_ensemble_checks.csv` — 14 rows × 4 cols
- `reports/tables/feature_family_ablation/no_momentum_retraining_checks.csv` — 9 rows × 4 cols
- `reports/tables/feature_family_ablation/official_predictive_frozen_full_checks.csv` — 5 rows × 6 cols
- `reports/tables/feature_family_ablation/official_predictive_key_checks.csv` — 3 rows × 3 cols
- `reports/tables/feature_family_ablation/predictive_formula_checks.csv` — 5 rows × 6 cols
- `reports/tables/feature_family_ablation/predictive_key_checks.csv` — 3 rows × 3 cols
- `reports/tables/four_method_portfolio_checks.csv` — 16 rows × 5 cols
- `reports/tables/fundamental_base_coverage.csv` — 17 rows × 5 cols; dates 2026-07-27T00:00:00 → 2026-07-27T00:00:00
- `reports/tables/fundamental_canonical_coverage.csv` — 18 rows × 7 cols; dates 2009-05-07T00:00:00 → 2026-08-10T00:00:00
- `reports/tables/fundamental_growth_factor_coverage.csv` — 7 rows × 5 cols; dates 2026-07-27T00:00:00 → 2026-07-27T00:00:00
- `reports/tables/fundamental_pit_coverage.csv` — 42 rows × 8 cols; dates 1970-01-01T00:00:00.000000151 → 2026-07-27T00:00:00
- `reports/tables/fundamental_processed_feature_coverage.csv` — 48 rows × 6 cols; dates 2026-07-27T00:00:00 → 2026-07-27T00:00:00
- `reports/tables/fundamental_raw_factor_coverage.csv` — 17 rows × 5 cols; dates 2026-07-27T00:00:00 → 2026-07-27T00:00:00
- `reports/tables/fundamental_ttm_coverage.csv` — 7 rows × 2 cols
- `reports/tables/modeling_panel_feature_coverage.csv` — 91 rows × 11 cols; dates 2014-01-31T00:00:00 → 2026-07-27T00:00:00
- `reports/tables/modeling_panel_leakage_checks.csv` — 21 rows × 4 cols
- `reports/tables/modeling_panel_readiness_checks.csv` — 17 rows × 4 cols
- `reports/tables/portfolio_risk_checks.csv` — 10 rows × 4 cols
- `reports/tables/risk_estimate_checks.csv` — 12 rows × 4 cols
- `reports/tables/robustness_ensemble_component_ablation_checks.csv` — 9 rows × 4 cols
- `reports/tables/robustness_evaluation_check_inventory.csv` — 17 rows × 8 cols
- `reports/tables/robustness_evaluation_coverage.csv` — 13 rows × 4 cols
- `reports/tables/robustness_feature_family_contract_checks.csv` — 11 rows × 4 cols
- `reports/tables/robustness_final_signal_checks.csv` — 8 rows × 4 cols
- `reports/tables/robustness_portfolio_construction_ablation_checks.csv` — 10 rows × 4 cols
- `reports/tables/robustness_portfolio_parameter_checks.csv` — 9 rows × 4 cols
- `reports/tables/robustness_prediction_horizon_checks.csv` — 8 rows × 4 cols
- `reports/tables/robustness_rebalance_frequency_checks.csv` — 10 rows × 4 cols
- `reports/tables/robustness_rolling_window_checks.csv` — 8 rows × 4 cols
- `reports/tables/robustness_temporal_checks.csv` — 8 rows × 4 cols
- `reports/tables/robustness_universe_exclusion_checks.csv` — 10 rows × 4 cols
- `reports/tables/transaction_cost_sensitivity_checks.csv` — 7 rows × 4 cols
- `reports/tables/walk_forward_readiness_checks.csv` — 19 rows × 4 cols
