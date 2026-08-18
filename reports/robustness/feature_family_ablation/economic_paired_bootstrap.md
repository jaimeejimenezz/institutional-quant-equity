# Feature-Family Economic Paired Bootstrap

## Methodology

- Uses the 77 common calendar months from the net economic backtests.
- Each bootstrap draw resamples months with replacement and keeps FULL and both feature ablations paired on the same sampled month.
- Uses 10,000 paired bootstrap replications with random seed 42.
- Confidence intervals are two-sided 95% percentile bootstrap intervals.
- Annualized geometric return is calculated from monthly compounding; annualized monthly Sharpe uses sqrt(12) and a zero monthly risk-free rate.
- These bootstrap Sharpe statistics are monthly-return statistics and are intentionally distinct from the daily Sharpe reported by the main performance engine.

## Paired bootstrap summary

| scenario | months | bootstrap_replications | observed_mean_monthly_return_difference | mean_monthly_difference_ci_low | mean_monthly_difference_ci_high | probability_mean_monthly_difference_gt_zero | observed_annualized_geometric_return_difference | annualized_geometric_difference_ci_low | annualized_geometric_difference_ci_high | probability_annualized_geometric_difference_gt_zero | observed_annualized_monthly_sharpe_difference | annualized_monthly_sharpe_difference_ci_low | annualized_monthly_sharpe_difference_ci_high | probability_annualized_monthly_sharpe_difference_gt_zero | candidate_beats_full_month_frequency | candidate_ties_full_month_frequency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_fundamentals | 77 | 10000 | 0.001566 | -0.000163 | 0.003305 | 0.962900 | 0.022171 | -0.002739 | 0.049014 | 0.959100 | 0.067775 | -0.033593 | 0.173161 | 0.908200 | 0.584416 | 0.000000 |
| no_momentum | 77 | 10000 | 0.000609 | -0.000624 | 0.001848 | 0.829800 | 0.007682 | -0.010005 | 0.026793 | 0.791700 | 0.002863 | -0.070683 | 0.077948 | 0.525500 | 0.571429 | 0.000000 |

## Calendar-year stability

| scenario | year | months | full_compounded_return | candidate_compounded_return | compounded_return_difference_vs_full | mean_monthly_return_difference_vs_full | candidate_beats_full_month_frequency | candidate_beats_full_year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_fundamentals | 2020 | 11 | 0.322335 | 0.352360 | 0.030025 | 0.002563 | 0.454545 | True |
| no_fundamentals | 2021 | 12 | 0.442826 | 0.450350 | 0.007524 | 0.000415 | 0.500000 | True |
| no_fundamentals | 2022 | 12 | -0.119135 | -0.108699 | 0.010436 | 0.000909 | 0.500000 | True |
| no_fundamentals | 2023 | 12 | 0.365551 | 0.394462 | 0.028911 | 0.001850 | 0.750000 | True |
| no_fundamentals | 2024 | 12 | 0.326876 | 0.334315 | 0.007439 | 0.000561 | 0.666667 | True |
| no_fundamentals | 2025 | 12 | 0.208576 | 0.245215 | 0.036638 | 0.002530 | 0.583333 | True |
| no_fundamentals | 2026 | 6 | 0.097799 | 0.117288 | 0.019488 | 0.002870 | 0.666667 | True |
| no_momentum | 2020 | 11 | 0.322335 | 0.358093 | 0.035757 | 0.002791 | 0.636364 | True |
| no_momentum | 2021 | 12 | 0.442826 | 0.433352 | -0.009473 | -0.000594 | 0.583333 | False |
| no_momentum | 2022 | 12 | -0.119135 | -0.114748 | 0.004387 | 0.000611 | 0.500000 | True |
| no_momentum | 2023 | 12 | 0.365551 | 0.370397 | 0.004846 | 0.000431 | 0.666667 | True |
| no_momentum | 2024 | 12 | 0.326876 | 0.338050 | 0.011174 | 0.000755 | 0.583333 | True |
| no_momentum | 2025 | 12 | 0.208576 | 0.197924 | -0.010652 | -0.000764 | 0.416667 | False |
| no_momentum | 2026 | 6 | 0.097799 | 0.110377 | 0.012577 | 0.001818 | 0.666667 | True |

## Trading-cost deltas versus FULL

| scenario | total_transaction_cost | transaction_cost_difference_vs_full | total_traded_notional | traded_notional_difference_vs_full | mean_one_way_turnover | turnover_difference_vs_full | effective_cost_bps | effective_cost_bps_difference_vs_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_fundamentals | 44460.465002 | 2447.561579 | 86545456.304811 | 4769474.389949 | 0.260972 | 0.001951 | 5.137238 | -0.000322 |
| no_momentum | 39400.402131 | -2612.501292 | 76734031.842446 | -5041950.072415 | 0.237221 | -0.021800 | 5.134671 | -0.002889 |

## Checks

| check | status | violations | description |
| --- | --- | --- | --- |
| expected_monthly_rows | PASS | 0 | Monthly return table must contain 308 rows. |
| expected_months | PASS | 0 | Paired return panel must contain 77 common months. |
| expected_strategies | PASS | 0 | Paired panel must contain FULL, both feature ablations and SPY. |
| complete_pairs | PASS | 0 | Every strategy must have a return for every paired calendar month. |
| bootstrap_scenarios | PASS | 0 | Bootstrap output must contain both feature ablations. |
| finite_bootstrap_statistics | PASS | 0 | All stored bootstrap statistics must be finite. |
