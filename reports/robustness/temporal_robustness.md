# Temporal Robustness Analysis

## Interpretation

This analysis does not change model parameters. It evaluates the same net out-of-sample portfolio returns across calendar years and market regimes.
The available out-of-sample backtest begins in February 2020, so the pre-COVID sample is too short for a strong conclusion and is explicitly flagged.
Up/down market conditions are defined from SPY monthly returns. High-volatility months are the top quartile of SPY monthly realized volatility within the available out-of-sample sample.

## Calendar-year results

| year | strategy_name | trading_days | partial_calendar_year | total_return | annualized_volatility | sharpe_ratio | maximum_drawdown | excess_total_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | alpha_risk_turnover | 232 | True | 0.262819 | 0.374713 | 0.866109 | -0.341795 | 0.085544 |
| 2020 | cvar | 232 | True | 0.298681 | 0.360396 | 0.969993 | -0.326287 | 0.121406 |
| 2020 | median_mad_de | 232 | True | 0.270634 | 0.367668 | 0.893221 | -0.327325 | 0.093359 |
| 2020 | score_weighted | 232 | True | 0.322335 | 0.384044 | 0.984561 | -0.343498 | 0.145061 |
| 2020 | top_n_equal_weight | 232 | True | 0.309581 | 0.377585 | 0.966725 | -0.337135 | 0.132306 |
| 2021 | alpha_risk_turnover | 252 | False | 0.418032 | 0.141627 | 2.538427 | -0.052626 | 0.130744 |
| 2021 | cvar | 252 | False | 0.398149 | 0.148420 | 2.333581 | -0.064415 | 0.110861 |
| 2021 | median_mad_de | 252 | False | 0.424641 | 0.141340 | 2.576238 | -0.057693 | 0.137353 |
| 2021 | score_weighted | 252 | False | 0.442826 | 0.147365 | 2.562975 | -0.054879 | 0.155538 |
| 2021 | top_n_equal_weight | 252 | False | 0.433378 | 0.142525 | 2.598945 | -0.052576 | 0.146090 |
| 2022 | alpha_risk_turnover | 251 | False | -0.054117 | 0.247126 | -0.103003 | -0.209722 | 0.127637 |
| 2022 | cvar | 251 | False | -0.073643 | 0.249394 | -0.183708 | -0.228194 | 0.108111 |
| 2022 | median_mad_de | 251 | False | -0.057616 | 0.241424 | -0.126502 | -0.208140 | 0.124138 |
| 2022 | score_weighted | 251 | False | -0.119135 | 0.265812 | -0.346696 | -0.231467 | 0.062619 |
| 2022 | top_n_equal_weight | 251 | False | -0.111275 | 0.257770 | -0.331030 | -0.227030 | 0.070478 |
| 2023 | alpha_risk_turnover | 250 | False | 0.206654 | 0.140156 | 1.421249 | -0.107740 | -0.055105 |
| 2023 | cvar | 250 | False | 0.202043 | 0.141529 | 1.381563 | -0.109445 | -0.059716 |
| 2023 | median_mad_de | 250 | False | 0.180806 | 0.138018 | 1.282914 | -0.102959 | -0.080953 |
| 2023 | score_weighted | 250 | False | 0.365551 | 0.154034 | 2.116682 | -0.113429 | 0.103793 |
| 2023 | top_n_equal_weight | 250 | False | 0.341777 | 0.150173 | 2.049198 | -0.114030 | 0.080019 |
| 2024 | alpha_risk_turnover | 252 | False | 0.306957 | 0.134862 | 2.053273 | -0.080931 | 0.058092 |
| 2024 | cvar | 252 | False | 0.273477 | 0.134895 | 1.860169 | -0.075855 | 0.024613 |
| 2024 | median_mad_de | 252 | False | 0.330054 | 0.135717 | 2.170322 | -0.085500 | 0.081190 |
| 2024 | score_weighted | 252 | False | 0.326876 | 0.137536 | 2.126035 | -0.077422 | 0.078011 |
| 2024 | top_n_equal_weight | 252 | False | 0.297599 | 0.129634 | 2.075203 | -0.069451 | 0.048734 |
| 2025 | alpha_risk_turnover | 250 | False | 0.157697 | 0.194647 | 0.854808 | -0.203780 | -0.019494 |
| 2025 | cvar | 250 | False | 0.161448 | 0.200877 | 0.850633 | -0.213412 | -0.015743 |
| 2025 | median_mad_de | 250 | False | 0.193069 | 0.199100 | 0.992557 | -0.205335 | 0.015878 |
| 2025 | score_weighted | 250 | False | 0.208576 | 0.204301 | 1.036149 | -0.207272 | 0.031386 |
| 2025 | top_n_equal_weight | 250 | False | 0.204677 | 0.196583 | 1.052513 | -0.203878 | 0.027486 |
| 2026 | alpha_risk_turnover | 123 | True | 0.114108 | 0.135332 | 1.703597 | -0.053370 | 0.013189 |
| 2026 | cvar | 123 | True | 0.099708 | 0.149253 | 1.379222 | -0.071341 | -0.001211 |
| 2026 | median_mad_de | 123 | True | 0.129845 | 0.154040 | 1.700869 | -0.077880 | 0.028927 |
| 2026 | score_weighted | 123 | True | 0.097799 | 0.147586 | 1.368938 | -0.075309 | -0.003120 |
| 2026 | top_n_equal_weight | 123 | True | 0.086213 | 0.143346 | 1.253382 | -0.076919 | -0.014706 |

## Regime results

| regime | strategy_name | trading_days | short_sample_warning | total_return | annualized_volatility | sharpe_ratio | maximum_drawdown | excess_total_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bear_market_2022 | alpha_risk_turnover | 251 | False | -0.054117 | 0.247126 | -0.103003 | -0.209722 | 0.127637 |
| bear_market_2022 | cvar | 251 | False | -0.073643 | 0.249394 | -0.183708 | -0.228194 | 0.108111 |
| bear_market_2022 | median_mad_de | 251 | False | -0.057616 | 0.241424 | -0.126502 | -0.208140 | 0.124138 |
| bear_market_2022 | score_weighted | 251 | False | -0.119135 | 0.265812 | -0.346696 | -0.231467 | 0.062619 |
| bear_market_2022 | top_n_equal_weight | 251 | False | -0.111275 | 0.257770 | -0.331030 | -0.227030 | 0.070478 |
| covid_crash | alpha_risk_turnover | 24 | True | -0.333623 | 0.834726 | -4.653388 | -0.341795 | 0.000380 |
| covid_crash | cvar | 24 | True | -0.321320 | 0.786680 | -4.746810 | -0.326287 | 0.012684 |
| covid_crash | median_mad_de | 24 | True | -0.319535 | 0.815970 | -4.515826 | -0.327325 | 0.014469 |
| covid_crash | score_weighted | 24 | True | -0.333855 | 0.842161 | -4.609984 | -0.343498 | 0.000149 |
| covid_crash | top_n_equal_weight | 24 | True | -0.327996 | 0.830632 | -4.577270 | -0.337135 | 0.006007 |
| covid_recovery_2020_2021 | alpha_risk_turnover | 449 | False | 1.522703 | 0.208105 | 2.601039 | -0.088506 | 0.336504 |
| covid_recovery_2020_2021 | cvar | 449 | False | 1.534772 | 0.208443 | 2.610207 | -0.091524 | 0.348572 |
| covid_recovery_2020_2021 | median_mad_de | 449 | False | 1.491488 | 0.206174 | 2.589498 | -0.082021 | 0.305289 |
| covid_recovery_2020_2021 | score_weighted | 449 | False | 1.670564 | 0.217208 | 2.648408 | -0.092139 | 0.484365 |
| covid_recovery_2020_2021 | top_n_equal_weight | 449 | False | 1.611494 | 0.212298 | 2.645476 | -0.088243 | 0.425295 |
| expansion_2023_2024 | alpha_risk_turnover | 502 | False | 0.577044 | 0.137408 | 1.733573 | -0.107740 | 0.001279 |
| expansion_2023_2024 | cvar | 502 | False | 0.530774 | 0.138112 | 1.617096 | -0.109445 | -0.044991 |
| expansion_2023_2024 | median_mad_de | 502 | False | 0.570536 | 0.136781 | 1.725685 | -0.102959 | -0.005230 |
| expansion_2023_2024 | score_weighted | 502 | False | 0.811917 | 0.145843 | 2.119786 | -0.113429 | 0.236152 |
| expansion_2023_2024 | top_n_equal_weight | 502 | False | 0.741088 | 0.140104 | 2.057742 | -0.114030 | 0.165323 |
| pre_covid_available | alpha_risk_turnover | 11 | True | 0.065224 | 0.146932 | 9.945821 | -0.003541 | 0.024366 |
| pre_covid_available | cvar | 11 | True | 0.055485 | 0.119547 | 10.427462 | -0.003741 | 0.014626 |
| pre_covid_available | median_mad_de | 11 | True | 0.067729 | 0.146417 | 10.350015 | -0.003508 | 0.026870 |
| pre_covid_available | score_weighted | 11 | True | 0.072465 | 0.144000 | 11.230054 | -0.005568 | 0.031607 |
| pre_covid_available | top_n_equal_weight | 11 | True | 0.069627 | 0.136976 | 11.353591 | -0.005952 | 0.028769 |
| recent_2025_2026 | alpha_risk_turnover | 373 | False | 0.289799 | 0.177116 | 1.058878 | -0.203780 | -0.006192 |
| recent_2025_2026 | cvar | 373 | False | 0.277254 | 0.185247 | 0.984674 | -0.213412 | -0.018738 |
| recent_2025_2026 | median_mad_de | 373 | False | 0.347983 | 0.185254 | 1.181343 | -0.205335 | 0.051992 |
| recent_2025_2026 | score_weighted | 373 | False | 0.326774 | 0.187301 | 1.113203 | -0.207272 | 0.030783 |
| recent_2025_2026 | top_n_equal_weight | 373 | False | 0.308535 | 0.180574 | 1.096087 | -0.203878 | 0.012544 |

## Conditional monthly behavior

| dimension | condition | strategy_name | months | mean_monthly_return | median_monthly_return | mean_spy_monthly_return | mean_excess_monthly_return | positive_month_ratio | outperform_spy_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market_direction | down_market_month | alpha_risk_turnover | 27 | -0.036590 | -0.030929 | -0.041071 | 0.004481 | 0.111111 | 0.629630 |
| market_direction | down_market_month | cvar | 27 | -0.039182 | -0.035610 | -0.041071 | 0.001889 | 0.111111 | 0.629630 |
| market_direction | down_market_month | median_mad_de | 27 | -0.035293 | -0.029324 | -0.041071 | 0.005778 | 0.074074 | 0.703704 |
| market_direction | down_market_month | score_weighted | 27 | -0.038549 | -0.031232 | -0.041071 | 0.002522 | 0.111111 | 0.629630 |
| market_direction | down_market_month | top_n_equal_weight | 27 | -0.038455 | -0.032285 | -0.041071 | 0.002616 | 0.111111 | 0.629630 |
| market_direction | up_market_month | alpha_risk_turnover | 50 | 0.046758 | 0.039617 | 0.042770 | 0.003988 | 0.960000 | 0.520000 |
| market_direction | up_market_month | cvar | 50 | 0.047386 | 0.042015 | 0.042770 | 0.004616 | 0.960000 | 0.540000 |
| market_direction | up_market_month | median_mad_de | 50 | 0.046968 | 0.038358 | 0.042770 | 0.004198 | 0.960000 | 0.580000 |
| market_direction | up_market_month | score_weighted | 50 | 0.051314 | 0.042175 | 0.042770 | 0.008543 | 0.980000 | 0.720000 |
| market_direction | up_market_month | top_n_equal_weight | 50 | 0.049898 | 0.041620 | 0.042770 | 0.007128 | 1.000000 | 0.680000 |
| volatility_regime | high_volatility_month | alpha_risk_turnover | 20 | -0.002006 | -0.024803 | -0.009217 | 0.007211 | 0.450000 | 0.750000 |
| volatility_regime | high_volatility_month | cvar | 20 | -0.003385 | -0.026512 | -0.009217 | 0.005832 | 0.400000 | 0.600000 |
| volatility_regime | high_volatility_month | median_mad_de | 20 | -0.001218 | -0.020663 | -0.009217 | 0.007999 | 0.450000 | 0.800000 |
| volatility_regime | high_volatility_month | score_weighted | 20 | -0.002716 | -0.022696 | -0.009217 | 0.006501 | 0.450000 | 0.700000 |
| volatility_regime | high_volatility_month | top_n_equal_weight | 20 | -0.002968 | -0.025380 | -0.009217 | 0.006249 | 0.450000 | 0.700000 |
| volatility_regime | normal_volatility_month | alpha_risk_turnover | 57 | 0.024388 | 0.020964 | 0.021297 | 0.003091 | 0.736842 | 0.491228 |
| volatility_regime | normal_volatility_month | cvar | 57 | 0.024194 | 0.021460 | 0.021297 | 0.002897 | 0.754386 | 0.561404 |
| volatility_regime | normal_volatility_month | median_mad_de | 57 | 0.024910 | 0.027465 | 0.021297 | 0.003613 | 0.719298 | 0.561404 |
| volatility_regime | normal_volatility_month | score_weighted | 57 | 0.027705 | 0.032441 | 0.021297 | 0.006408 | 0.754386 | 0.684211 |
| volatility_regime | normal_volatility_month | top_n_equal_weight | 57 | 0.026596 | 0.027956 | 0.021297 | 0.005299 | 0.771930 | 0.649123 |

## Readiness checks

| check | status | violations | description |
| --- | --- | --- | --- |
| five_methods_per_year | PASS | 0 | Every observed calendar year must contain all five methods. |
| five_methods_per_regime | PASS | 0 | Every temporal regime must contain all five methods. |
| five_methods_per_condition | PASS | 0 | Every monthly market condition must contain all five methods. |
| finite_yearly_metrics | PASS | 0 | Key yearly performance metrics must be finite. |
| finite_regime_metrics | PASS | 0 | Key regime performance metrics must be finite. |
| positive_year_month_counts | PASS | 0 | Conditional analyses must contain at least one month. |
| valid_positive_month_ratios | PASS | 0 | Positive-month ratios must lie between zero and one. |
| valid_outperformance_ratios | PASS | 0 | SPY outperformance ratios must lie between zero and one. |
