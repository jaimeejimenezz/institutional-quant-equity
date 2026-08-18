# Feature-Family Economic Ablation

## Methodology

- Uses the frozen FULL final alpha signal and the two walk-forward-retrained feature-family final alpha signals.
- All three signals are passed through the same frozen score-weighted portfolio construction.
- Portfolio construction uses 25 candidates, a 5% security cap, a 25% sector cap and at least 20 positions.
- Backtests use the same project configuration, advanced execution-cost model, risk estimates and SPY benchmark as the existing ensemble-component ablation.
- The fixed transaction-cost field of the MVP engine is set to zero so costs are charged only through the advanced execution-cost model, matching the existing robustness template.
- No test-period result is used to retune signals, portfolio limits, costs or execution assumptions.

## Results

| strategy_name | is_baseline | cagr | cagr_difference_vs_full | sharpe_ratio | sharpe_difference_vs_full | maximum_drawdown | excess_cagr | excess_cagr_difference_vs_full | mean_one_way_turnover | construction_one_way_turnover | maximum_sector_weight | effective_cost_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_ensemble | True | 0.244270 | 0.000000 | 1.086881 | 0.000000 | -0.343498 | 0.087729 | 0.000000 | 0.259021 | 0.233225 | 0.250000 | 5.137560 |
| no_fundamentals | False | 0.266559 | 0.022289 | 1.162630 | 0.075749 | -0.346866 | 0.110018 | 0.022289 | 0.260972 | 0.234478 | 0.250000 | 5.137238 |
| no_momentum | False | 0.251993 | 0.007722 | 1.103858 | 0.016977 | -0.345520 | 0.095451 | 0.007722 | 0.237221 | 0.211636 | 0.250000 | 5.134671 |

## Readiness checks

| check | status | violations | description |
| --- | --- | --- | --- |
| expected_scenarios | PASS | 0 | The analysis must contain FULL, no fundamentals and no momentum. |
| expected_oos_dates | PASS | 0 | Every feature scenario must contain all 77 frozen OOS dates. |
| complete_cross_sections | PASS | 0 | Every feature scenario must retain the full 50-stock cross-section. |
| identical_oos_keys | PASS | 0 | All scenarios must use exactly the same date-ticker OOS keys as FULL. |
| fully_invested | PASS | 0 | Every feature-family portfolio must sum to one. |
| long_only | PASS | 0 | Feature-family portfolios must remain long-only. |
| security_cap | PASS | 0 | Feature-family portfolios must respect the 5% security cap. |
| sector_cap | PASS | 0 | Feature-family portfolios must respect the 25% sector cap. |
| finite_performance | PASS | 0 | Key economic performance metrics must remain finite. |
| positive_final_values | PASS | 0 | Every feature-family backtest must retain positive final value. |
| reference_full_reproduction | PASS | 0 | FULL must reproduce the previously stored ensemble-component-ablation economic baseline. |
