# Ensemble Component Ablation

## Methodology

- Uses the frozen out-of-sample predictions and the original fold-specific validation weights.
- One ensemble component is removed at a time and the two remaining validation weights are renormalized.
- Tested removals: technical composite, Elastic Net and LightGBM Ranker.
- Each ablated signal is converted to a monthly percentile ranking and then passed through the same score-weighted portfolio construction and advanced execution-cost model.
- No test-period outcome is used to choose alternative weights or re-tune the remaining ensemble.
- This experiment directly covers the no-LightGBM ablation. Feature-family ablations such as no fundamentals and no momentum require separate walk-forward retraining and are evaluated later.

## Results

| strategy_name | is_baseline | cagr | cagr_difference_vs_full | sharpe_ratio | sharpe_difference_vs_full | maximum_drawdown | excess_cagr | excess_cagr_difference_vs_full | mean_one_way_turnover | maximum_sector_weight | effective_cost_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_ensemble | True | 0.244270 | 0.000000 | 1.086881 | 0.000000 | -0.343498 | 0.087729 | 0.000000 | 0.259021 | 0.250000 | 5.137560 |
| without_composite | False | 0.230727 | -0.013543 | 1.040972 | -0.045909 | -0.346518 | 0.074185 | -0.013543 | 0.251868 | 0.250000 | 5.136014 |
| without_elastic_net | False | 0.243502 | -0.000769 | 1.082188 | -0.004693 | -0.351797 | 0.086960 | -0.000769 | 0.288783 | 0.250000 | 5.141406 |
| without_lightgbm_ranker | False | 0.240385 | -0.003886 | 1.074344 | -0.012537 | -0.350442 | 0.083843 | -0.003886 | 0.265221 | 0.250000 | 5.132892 |

## Readiness checks

| check | status | violations | description |
| --- | --- | --- | --- |
| expected_scenarios | PASS | 0 | The analysis must contain the full ensemble and all three one-component ablations. |
| expected_oos_dates | PASS | 0 | Every signal scenario must contain all 77 frozen OOS dates. |
| complete_cross_sections | PASS | 0 | Every signal scenario must retain the full 50-stock cross-section. |
| fully_invested | PASS | 0 | Every ablation portfolio must sum to one. |
| long_only | PASS | 0 | Ablation portfolios must remain long-only. |
| security_cap | PASS | 0 | Ablation portfolios must respect the 5% security cap. |
| sector_cap | PASS | 0 | Ablation portfolios must respect the 25% sector cap. |
| finite_performance | PASS | 0 | Key ablation performance metrics must remain finite. |
| positive_final_values | PASS | 0 | Every ablation backtest must retain positive final value. |
