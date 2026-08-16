# Final Signal Statistical Robustness

## Methodology

- The production-safe final alpha signal is joined back to realized 21-session OOS excess returns only inside this research diagnostic.
- Monthly IC is the Spearman cross-sectional correlation between final percentile score and realized target.
- Top-bottom spread is the mean realized target of the predicted top quintile minus the predicted bottom quintile.
- Confidence intervals use `10,000` circular block-bootstrap replications with `3`-month blocks.
- Sector IC is diagnostic because individual sector cross-sections are much smaller than the full universe.

## Bootstrap summary

| months | bootstrap_replications | block_length_months | confidence_level | observed_mean_ic | mean_ic_ci_lower | mean_ic_ci_upper | probability_mean_ic_positive | observed_annualized_ic_ir | annualized_ic_ir_ci_lower | annualized_ic_ir_ci_upper | observed_positive_ic_ratio | observed_mean_top_bottom_spread | top_bottom_spread_ci_lower | top_bottom_spread_ci_upper | probability_mean_spread_positive | observed_positive_spread_ratio | observed_mean_top_quintile_precision | top_quintile_precision_ci_lower | top_quintile_precision_ci_upper | probability_both_ic_and_spread_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 77 | 10000 | 3 | 0.950000 | 0.040514 | -0.002100 | 0.081543 | 0.968200 | 0.671967 | -0.033321 | 1.411684 | 0.545455 | 0.012308 | 0.003345 | 0.020913 | 0.996800 | 0.597403 | 0.259740 | 0.235065 | 0.285714 | 0.967900 |

## Calendar-year stability

| year | months | mean_ic | median_ic | annualized_ic_ir | positive_ic_ratio | mean_top_bottom_spread | median_top_bottom_spread | positive_spread_ratio | mean_top_quintile_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 12 | 0.080352 | 0.061897 | 1.563804 | 0.666667 | 0.028384 | 0.034256 | 0.750000 | 0.283333 |
| 2021 | 12 | 0.037279 | -0.014166 | 0.490163 | 0.500000 | 0.003348 | -0.000947 | 0.500000 | 0.266667 |
| 2022 | 12 | 0.031906 | 0.001129 | 0.467590 | 0.500000 | 0.009296 | 0.001579 | 0.500000 | 0.216667 |
| 2023 | 12 | -0.000480 | -0.034718 | -0.008811 | 0.416667 | 0.005123 | -0.004409 | 0.500000 | 0.250000 |
| 2024 | 12 | 0.077999 | 0.085090 | 2.024654 | 0.750000 | 0.019983 | 0.024015 | 0.750000 | 0.275000 |
| 2025 | 12 | 0.040684 | 0.002569 | 0.546783 | 0.500000 | 0.009514 | 0.009444 | 0.583333 | 0.300000 |
| 2026 | 5 | -0.018660 | -0.043745 | -0.292573 | 0.400000 | 0.007993 | 0.015113 | 0.600000 | 0.180000 |

## Sector stability

| sector | valid_months | mean_companies_per_month | mean_ic | annualized_ic_ir | positive_ic_ratio |
| --- | --- | --- | --- | --- | --- |
| Industrials | 77 | 5.000000 | 0.122078 | 0.824520 | 0.545455 |
| Financials | 77 | 7.000000 | 0.108071 | 0.715720 | 0.545455 |
| Consumer Staples | 77 | 5.000000 | 0.049351 | 0.313412 | 0.558442 |
| Health Care | 77 | 7.000000 | 0.036178 | 0.292652 | 0.519481 |
| Information Technology | 77 | 8.000000 | 0.026283 | 0.209459 | 0.480519 |
| Consumer Discretionary | 77 | 6.000000 | 0.005566 | 0.041993 | 0.506494 |
| Energy | 77 | 3.000000 | -0.006494 | -0.029819 | 0.519481 |
| Communication Services | 77 | 4.000000 | -0.012987 | -0.086439 | 0.454545 |

## Readiness checks

| check | status | violations | description |
| --- | --- | --- | --- |
| research_rows_match_final_signal | PASS | 0 | The current frozen OOS final-signal research panel should contain 77 dates × 50 securities. |
| expected_oos_months | PASS | 0 | The current frozen OOS evaluation should contain 77 monthly dates. |
| minimum_valid_months | PASS | 0 | Final-signal statistical robustness requires at least 60 valid months. |
| finite_monthly_metrics | PASS | 0 | IC, spread and precision must be finite for every evaluated month. |
| valid_bootstrap_probabilities | PASS | 0 | Bootstrap probabilities must lie between zero and one. |
| bootstrap_interval_order | PASS | 0 | Bootstrap confidence-interval lower bounds must not exceed upper bounds. |
| yearly_coverage | PASS | 0 | The current OOS sample should cover calendar years 2020 through 2026. |
| sector_coverage | PASS | 0 | At least five sectors should have enough monthly cross-sectional observations for sector IC diagnostics. |
