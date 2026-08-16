# Portfolio Parameter Sensitivity

## Methodology

- Top-N sensitivity uses sector-controlled equal-weight portfolios for N = 10, 20, 25, 30 and 40.
- A fixed 5% security cap is mathematically infeasible for Top-10, so each equal-weight scenario uses the larger of 5% and 1/N as its configured security cap.
- Security-cap sensitivity keeps the score-weighted candidate count fixed at 25 and tests 4%, 5%, 7.5% and 10%.
- Sector cap remains fixed at 25% in every scenario.
- All scenarios use the same frozen final alpha signal, same OOS dates, same market data and same advanced execution-cost model.
- Baselines are Top-25 for the Top-N experiment and a 5% security cap for the score-weighted cap experiment.

## Results

| experiment | strategy_name | top_n | configured_security_cap | is_baseline | cagr | cagr_difference_vs_baseline | sharpe_ratio | sharpe_difference_vs_baseline | maximum_drawdown | excess_cagr | mean_one_way_turnover | maximum_weight | maximum_sector_weight | mean_effective_positions | effective_cost_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| security_cap | score_cap_400bps | 25 | 0.040000 | False | 0.232428 | -0.011842 | 1.066914 | -0.019968 | -0.337135 | 0.075886 | 0.241765 | 0.040000 | 0.240000 | 25.000000 | 5.150280 |
| security_cap | score_cap_500bps | 25 | 0.050000 | True | 0.244270 | 0.000000 | 1.086881 | 0.000000 | -0.343498 | 0.087729 | 0.259021 | 0.050000 | 0.250000 | 24.176025 | 5.137560 |
| security_cap | score_cap_750bps | 25 | 0.075000 | False | 0.245691 | 0.001420 | 1.089967 | 0.003086 | -0.344461 | 0.089149 | 0.260495 | 0.054267 | 0.250000 | 24.091012 | 5.137422 |
| security_cap | score_cap_1000bps | 25 | 0.100000 | False | 0.245691 | 0.001420 | 1.089967 | 0.003086 | -0.344461 | 0.089149 | 0.260495 | 0.054267 | 0.250000 | 24.091012 | 5.137422 |
| top_n | top_n_10 | 10 | 0.100000 | False | 0.254753 | 0.022325 | 1.007147 | -0.059767 | -0.378109 | 0.098212 | 0.417083 | 0.100000 | 0.200000 | 10.000000 | 5.265025 |
| top_n | top_n_20 | 20 | 0.050000 | False | 0.241374 | 0.008946 | 1.063077 | -0.003836 | -0.343960 | 0.084832 | 0.296988 | 0.050000 | 0.250000 | 20.000000 | 5.174408 |
| top_n | top_n_25 | 25 | 0.050000 | True | 0.232428 | 0.000000 | 1.066914 | 0.000000 | -0.337135 | 0.075886 | 0.241765 | 0.040000 | 0.240000 | 25.000000 | 5.150280 |
| top_n | top_n_30 | 30 | 0.050000 | False | 0.213448 | -0.018980 | 1.019591 | -0.047322 | -0.339122 | 0.056906 | 0.207652 | 0.033333 | 0.233333 | 30.000000 | 5.123948 |
| top_n | top_n_40 | 40 | 0.050000 | False | 0.197829 | -0.034600 | 0.994196 | -0.072718 | -0.333124 | 0.041287 | 0.136699 | 0.025000 | 0.200000 | 40.000000 | 5.094130 |

## Readiness checks

| check | status | violations | description |
| --- | --- | --- | --- |
| expected_scenarios | PASS | 0 | All Top-N and security-cap scenarios must be present. |
| expected_signal_dates | PASS | 0 | Every scenario must contain all 77 OOS signal dates. |
| fully_invested | PASS | 0 | Every sensitivity portfolio must sum to one. |
| long_only | PASS | 0 | Sensitivity portfolios must remain long-only. |
| security_caps | PASS | 0 | Observed weights must respect each scenario security cap. |
| sector_cap | PASS | 0 | Every sensitivity portfolio must respect the 25% sector cap. |
| finite_performance | PASS | 0 | Key sensitivity performance metrics must be finite. |
| positive_final_value | PASS | 0 | Every sensitivity backtest must retain positive portfolio value. |
| one_baseline_per_experiment | PASS | 0 | Each parameter experiment must contain exactly one frozen baseline. |
