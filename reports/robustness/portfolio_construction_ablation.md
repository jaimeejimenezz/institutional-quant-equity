# Portfolio Construction Ablation

## Methodology

- The final alpha signal, covariance estimates, risk estimates, market data and execution-cost assumptions remain frozen.
- Sector-control ablation compares the score-weighted portfolio with the original 25% sector cap against the same construction with a 100% sector cap.
- Turnover-penalty ablation compares the alpha-risk-turnover optimizer with its original penalty against an otherwise identical optimizer with turnover_penalty = 0.
- No out-of-sample result is used to retune the remaining construction parameters.
- The score-weighted portfolio also serves as the project's non-optimized construction reference when interpreting whether portfolio optimization adds economic value.

## Results

| experiment | strategy_name | is_controlled_baseline | configured_sector_cap | configured_turnover_penalty | cagr | cagr_difference_vs_controlled | sharpe_ratio | sharpe_difference_vs_controlled | maximum_drawdown | excess_cagr | mean_one_way_turnover | turnover_difference_vs_controlled | maximum_sector_weight | total_transaction_cost | effective_cost_bps | mean_predicted_alpha_proxy | mean_predicted_volatility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sector_control | score_weighted_sector_controlled | True | 0.250000 |  | 0.244270 | 0.000000 | 1.086881 | 0.000000 | -0.343498 | 0.087729 | 0.259021 | 0.000000 | 0.250000 | 42012.903422 | 5.137560 |  |  |
| sector_control | score_weighted_no_sector_control | False | 1.000000 |  | 0.244469 | 0.000199 | 1.086877 | -0.000004 | -0.342040 | 0.087927 | 0.259163 | 0.000142 | 0.291560 | 42149.339639 | 5.137666 |  |  |
| turnover_penalty | alpha_risk_turnover_penalized | True | 0.250000 | 0.010000 | 0.213635 | 0.000000 | 1.011181 | 0.000000 | -0.341795 | 0.057093 | 0.261907 | 0.000000 | 0.249999 | 39028.737778 | 5.158666 | 0.028527 | 0.190236 |
| turnover_penalty | alpha_risk_turnover_no_penalty | False | 0.250000 | 0.000000 | 0.208143 | -0.005493 | 0.991552 | -0.019629 | -0.345571 | 0.051601 | 0.310415 | 0.048508 | 0.249999 | 46739.986023 | 5.157629 | 0.028597 | 0.189298 |

## Readiness checks

| check | status | violations | description |
| --- | --- | --- | --- |
| expected_scenarios | PASS | 0 | The analysis must contain all four construction-ablation scenarios. |
| expected_oos_dates | PASS | 0 | Every construction scenario must contain all frozen OOS dates. |
| fully_invested | PASS | 0 | Every construction-ablation portfolio must sum to one. |
| long_only | PASS | 0 | Construction-ablation portfolios must remain long-only. |
| security_caps | PASS | 0 | Every scenario must respect the configured security cap. |
| controlled_sector_caps | PASS | 0 | Sector-controlled scenarios must respect the configured sector cap. |
| finite_performance | PASS | 0 | Key construction-ablation metrics must remain finite. |
| positive_final_values | PASS | 0 | Every construction-ablation backtest must retain positive final value. |
| one_baseline_per_experiment | PASS | 0 | Each construction experiment must contain exactly one controlled baseline. |
| turnover_penalty_removed | PASS | 0 | The turnover-penalty ablation must set the penalty exactly to zero. |
