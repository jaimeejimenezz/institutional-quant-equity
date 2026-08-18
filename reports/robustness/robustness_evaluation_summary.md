# Robustness Evaluation Summary

## Status

**COMPLETE WITH DOCUMENTED LIMITATION**

## Validation inventory

| category | suite | suite_status | checks | passed_checks | failed_checks | artifact |
| --- | --- | --- | --- | --- | --- | --- |
| temporal | temporal_robustness | PASS | 8 | 8 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\robustness_temporal_checks.csv |
| parameters | transaction_cost_sensitivity | PASS | 7 | 7 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\transaction_cost_sensitivity_checks.csv |
| statistics | monthly_portfolio_bootstrap | PASS | 8 | 8 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\bootstrap_robustness_checks.csv |
| statistics | final_signal_statistics | PASS | 8 | 8 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\robustness_final_signal_checks.csv |
| parameters | portfolio_parameter_sensitivity | PASS | 9 | 9 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\robustness_portfolio_parameter_checks.csv |
| parameters | rebalance_frequency_sensitivity | PASS | 10 | 10 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\robustness_rebalance_frequency_checks.csv |
| parameters | rolling_evaluation_windows | PASS | 8 | 8 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\robustness_rolling_window_checks.csv |
| parameters | prediction_horizon_robustness | PASS | 8 | 8 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\robustness_prediction_horizon_checks.csv |
| universe | universe_exclusion_robustness | PASS | 10 | 10 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\robustness_universe_exclusion_checks.csv |
| ablation | ensemble_component_ablation | PASS | 9 | 9 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\robustness_ensemble_component_ablation_checks.csv |
| ablation | portfolio_construction_ablation | PASS | 10 | 10 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\robustness_portfolio_construction_ablation_checks.csv |
| ablation | feature_family_contract | PASS | 11 | 11 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\robustness_feature_family_contract_checks.csv |
| ablation | feature_family_predictive_keys | PASS | 3 | 3 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\feature_family_ablation\official_predictive_key_checks.csv |
| ablation | feature_family_predictive_full_reference | PASS | 5 | 5 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\feature_family_ablation\official_predictive_frozen_full_checks.csv |
| ablation | feature_family_predictive_candidate_match | PASS | 1 | 1 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\feature_family_ablation\official_predictive_full_candidate_match.csv |
| ablation | feature_family_economic_backtest | PASS | 11 | 11 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\feature_family_ablation\economic_checks.csv |
| ablation | feature_family_economic_bootstrap | PASS | 6 | 6 | 0 | C:\Users\Usuario\Documents\PROYECTO_FINTECH\institutional-quant-equity\reports\tables\feature_family_ablation\economic_bootstrap_checks.csv |

## Coverage

| dimension | category | status | note |
| --- | --- | --- | --- |
| calendar_years_and_market_regimes | temporal | COMPLETE | Calendar years, COVID-related periods, up/down markets and high-volatility conditions. |
| transaction_cost_assumptions | parameters | COMPLETE | Linear cost scenarios plus the liquidity-dependent execution model. |
| top_n_and_security_caps | parameters | COMPLETE | Top-N and maximum-security-weight sensitivity. |
| rebalance_frequency | parameters | COMPLETE | Monthly versus calendar-quarter rebalancing. |
| evaluation_windows | parameters | COMPLETE | Overlapping 12-, 24- and 36-month evaluation windows. |
| return_horizons | parameters | COMPLETE | Frozen signal evaluated at 10-, 21- and 42-session realized horizons. |
| frozen_universe_exclusions | universe | COMPLETE | Technology, leave-one-sector-out and largest-liquidity exclusions. |
| expanded_universe | universe | DEFERRED_LIMITATION | A genuine expanded-universe experiment requires adding securities and rebuilding the upstream point-in-time pipeline. |
| portfolio_return_bootstrap | statistics | COMPLETE | Paired monthly bootstrap for the portfolio methods. |
| signal_ic_spread_sector_statistics | statistics | COMPLETE | IC, spread, yearly and sector stability diagnostics. |
| no_fundamentals_and_no_momentum | ablation | COMPLETE | Walk-forward feature-family ablations with predictive and economic comparison. |
| no_lightgbm | ablation | COMPLETE | LightGBM Ranker removed from the frozen ensemble. |
| no_optimization_no_sector_control_no_turnover_penalty | ablation | COMPLETE | Non-optimized reference, sector-cap ablation and turnover-penalty ablation. |

## Feature-family interpretation

- `no_fundamentals`: observed economic CAGR delta +0.0223; paired annualized geometric-return delta +0.0222, 95% CI [-0.0027, +0.0490], P(delta > 0)=0.9591; monthly win frequency=0.5844; the two-sided 95% interval includes zero; calendar-year blocks won: 7/7; turnover delta: +0.0020.
- `no_momentum`: observed economic CAGR delta +0.0077; paired annualized geometric-return delta +0.0077, 95% CI [-0.0100, +0.0268], P(delta > 0)=0.7917; monthly win frequency=0.5714; the two-sided 95% interval includes zero; calendar-year blocks won: 5/7; turnover delta: -0.0218.
- Ablation outcomes are diagnostics, not permission to retune the frozen production specification using the same out-of-sample period. Any specification change should be validated on new untouched data.

## Ensemble-component ablation

| strategy_name | final_portfolio_value | total_return | cagr | annualized_volatility | sharpe_ratio | sortino_ratio | maximum_drawdown | beta_vs_spy | annualized_alpha_vs_spy | excess_cagr | total_transaction_cost | total_traded_notional | mean_two_way_turnover | mean_one_way_turnover | mean_positions | maximum_weight | maximum_sector_weight | mean_effective_positions | construction_one_way_turnover | effective_cost_bps | cagr_difference_vs_full | sharpe_difference_vs_full | excess_cagr_difference_vs_full | is_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_ensemble | 4040172.810593 | 3.040173 | 0.244270 | 0.224438 | 1.086881 | 1.555182 | -0.343498 | 1.062284 | 0.067231 | 0.087729 | 42012.903422 | 81775981.914862 | 0.504809 | 0.259021 | 25.000000 | 0.050000 | 0.250000 | 24.176025 | 0.233225 | 5.137560 | 0.000000 | 0.000000 | 0.000000 | True |
| without_composite | 3767327.810214 | 2.767328 | 0.230727 | 0.223664 | 1.040972 | 1.477862 | -0.346518 | 1.059024 | 0.056665 | 0.074185 | 39425.251556 | 76762350.768538 | 0.490511 | 0.251868 | 25.000000 | 0.050000 | 0.250000 | 24.182651 | 0.229540 | 5.136014 | -0.013543 | -0.045909 | -0.013543 | False |
| without_elastic_net | 4024251.708922 | 3.024252 | 0.243502 | 0.224962 | 1.082188 | 1.541700 | -0.351797 | 1.061970 | 0.066797 | 0.086960 | 49141.215822 | 95579336.901113 | 0.564302 | 0.288783 | 25.000000 | 0.050000 | 0.250000 | 24.170747 | 0.262810 | 5.141406 | -0.000769 | -0.004693 | -0.000769 | False |
| without_lightgbm_ranker | 3960240.283659 | 2.960240 | 0.240385 | 0.224009 | 1.074344 | 1.552682 | -0.350442 | 1.056640 | 0.064895 | 0.083843 | 41731.419605 | 81301964.952344 | 0.517202 | 0.265221 | 25.000000 | 0.050000 | 0.250000 | 24.169639 | 0.238600 | 5.132892 | -0.003886 | -0.012537 | -0.003886 | False |

## Portfolio-construction ablation

| strategy_name | final_portfolio_value | total_return | cagr | annualized_volatility | sharpe_ratio | sortino_ratio | maximum_drawdown | beta_vs_spy | annualized_alpha_vs_spy | excess_cagr | total_transaction_cost | total_traded_notional | mean_two_way_turnover | mean_one_way_turnover | mean_positions | maximum_weight | maximum_sector_weight | mean_effective_positions | construction_one_way_turnover | mean_predicted_alpha_proxy | mean_predicted_volatility | optimizer_mean_one_way_turnover | optimizer_maximum_weight | optimizer_maximum_sector_weight | effective_cost_bps | experiment | is_controlled_baseline | configured_sector_cap | configured_turnover_penalty | baseline_cagr | baseline_sharpe | baseline_turnover | cagr_difference_vs_controlled | sharpe_difference_vs_controlled | turnover_difference_vs_controlled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| score_weighted_sector_controlled | 4040172.810593 | 3.040173 | 0.244270 | 0.224438 | 1.086881 | 1.555182 | -0.343498 | 1.062284 | 0.067231 | 0.087729 | 42012.903422 | 81775981.914862 | 0.504809 | 0.259021 | 25.000000 | 0.050000 | 0.250000 | 24.176025 | 0.233225 |  |  |  |  |  | 5.137560 | sector_control | True | 0.250000 |  | 0.244270 | 1.086881 | 0.259021 | 0.000000 | 0.000000 | 0.000000 |
| score_weighted_no_sector_control | 4044294.604275 | 3.044295 | 0.244469 | 0.224625 | 1.086877 | 1.555087 | -0.342040 | 1.063158 | 0.067288 | 0.087927 | 42149.339639 | 82039852.056168 | 0.505092 | 0.259163 | 25.000000 | 0.050000 | 0.291560 | 24.168716 | 0.233378 |  |  |  |  |  | 5.137666 | sector_control | False | 1.000000 |  | 0.244270 | 1.086881 | 0.259021 | 0.000199 | -0.000004 | 0.000142 |
| alpha_risk_turnover_penalized | 3445326.342035 | 2.445326 | 0.213635 | 0.214395 | 1.011181 | 1.439904 | -0.341795 | 1.003290 | 0.049899 | 0.057093 | 39028.737778 | 75656641.482621 | 0.510578 | 0.261907 | 21.402597 | 0.049999 | 0.249999 | 20.424987 | 0.227074 | 0.028527 | 0.190236 | 0.227074 | 0.049999 | 0.249999 | 5.158666 | turnover_penalty | True | 0.250000 | 0.010000 | 0.213635 | 1.011181 | 0.261907 | 0.000000 | 0.000000 | 0.000000 |
| alpha_risk_turnover_no_penalty | 3346910.346644 | 2.346910 | 0.208143 | 0.213957 | 0.991552 | 1.412107 | -0.345571 | 1.002181 | 0.045441 | 0.051601 | 46739.986023 | 90623016.260685 | 0.607544 | 0.310415 | 21.454545 | 0.049999 | 0.249999 | 20.426919 | 0.278296 | 0.028597 | 0.189298 | 0.278296 | 0.049999 | 0.249999 | 5.157629 | turnover_penalty | False | 0.250000 | 0.000000 | 0.213635 | 1.011181 | 0.261907 | -0.005493 | -0.019629 | 0.048508 |

## Documented limitation

A genuine expanded-universe experiment is not claimed as completed. It requires additional securities and a rebuilt point-in-time upstream data pipeline so that the comparison remains methodologically valid.

## Research interpretation rule

The objective of robustness analysis is not to search for the best-looking historical variant. The frozen specification remains the reference, and alternative specifications are interpreted as evidence about stability, dependence and possible simplification.
