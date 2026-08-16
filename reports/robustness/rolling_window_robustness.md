# Rolling Window Robustness

## Methodology

- Uses the frozen net out-of-sample daily return paths with advanced execution costs already applied.
- Evaluates overlapping 12-, 24- and 36-month windows ending at each available calendar month.
- All five methods and SPY are evaluated on identical dates inside each window.
- This test changes only the evaluation window. It does not re-fit models or tune portfolio parameters.
- A later training-window experiment can separately compare expanding and rolling model estimation schemes.

## Rolling-window summary

| window_months | strategy_name | windows | median_cagr | cagr_10th_percentile | minimum_cagr | positive_cagr_window_ratio | median_sharpe | sharpe_10th_percentile | minimum_sharpe | positive_excess_cagr_window_ratio | median_excess_cagr | worst_maximum_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | score_weighted | 66 | 0.282335 | 0.008099 | -0.119580 | 0.909091 | 1.462886 | 0.161072 | -0.346696 | 1.000000 | 0.083208 | -0.343498 |
| 12 | top_n_equal_weight | 66 | 0.267805 | 0.012669 | -0.111693 | 0.909091 | 1.436941 | 0.175768 | -0.331030 | 0.969697 | 0.078558 | -0.337135 |
| 12 | median_mad_de | 66 | 0.233348 | 0.037980 | -0.057839 | 0.939394 | 1.162906 | 0.280879 | -0.126502 | 0.909091 | 0.053308 | -0.327325 |
| 12 | cvar | 66 | 0.208018 | 0.004425 | -0.073926 | 0.924242 | 1.130221 | 0.137268 | -0.183708 | 0.772727 | 0.034180 | -0.326287 |
| 12 | alpha_risk_turnover | 66 | 0.217049 | 0.027431 | -0.054327 | 0.939394 | 1.124483 | 0.230874 | -0.103003 | 0.818182 | 0.050124 | -0.341795 |
| 24 | score_weighted | 54 | 0.243834 | 0.125737 | 0.050868 | 1.000000 | 1.265517 | 0.659141 | 0.335580 | 1.000000 | 0.087347 | -0.343498 |
| 24 | top_n_equal_weight | 54 | 0.230761 | 0.120657 | 0.045200 | 1.000000 | 1.252718 | 0.648922 | 0.313845 | 1.000000 | 0.076580 | -0.337135 |
| 24 | median_mad_de | 54 | 0.224918 | 0.104330 | 0.023313 | 1.000000 | 1.233978 | 0.597440 | 0.215078 | 0.962963 | 0.053707 | -0.327325 |
| 24 | alpha_risk_turnover | 54 | 0.217138 | 0.109801 | 0.033447 | 1.000000 | 1.186687 | 0.615454 | 0.263325 | 0.981481 | 0.044229 | -0.341795 |
| 24 | cvar | 54 | 0.196599 | 0.093755 | 0.015414 | 1.000000 | 1.101133 | 0.549016 | 0.176993 | 0.944444 | 0.030154 | -0.326287 |
| 36 | score_weighted | 42 | 0.221773 | 0.188170 | 0.166039 | 1.000000 | 1.096434 | 0.940104 | 0.879447 | 1.000000 | 0.091529 | -0.343498 |
| 36 | top_n_equal_weight | 42 | 0.210854 | 0.179471 | 0.156808 | 1.000000 | 1.094750 | 0.941710 | 0.866382 | 1.000000 | 0.081509 | -0.337135 |
| 36 | median_mad_de | 42 | 0.190090 | 0.160169 | 0.135041 | 1.000000 | 1.050909 | 0.894285 | 0.811518 | 1.000000 | 0.058861 | -0.327325 |
| 36 | alpha_risk_turnover | 42 | 0.189687 | 0.163167 | 0.138189 | 1.000000 | 1.043026 | 0.904304 | 0.821918 | 0.928571 | 0.062534 | -0.341795 |
| 36 | cvar | 42 | 0.179337 | 0.145498 | 0.116714 | 1.000000 | 0.962168 | 0.828940 | 0.709998 | 0.904762 | 0.043660 | -0.326287 |

## Readiness checks

| check | status | violations | description |
| --- | --- | --- | --- |
| expected_summary_rows | PASS | 0 | Rolling-window summary must contain one row per method and window length. |
| five_methods_per_length | PASS | 0 | Every rolling-window length must contain all five methods. |
| five_methods_per_window | PASS | 0 | Every individual rolling window must contain all five methods. |
| finite_window_metrics | PASS | 0 | Key rolling-window metrics must remain finite. |
| positive_window_lengths | PASS | 0 | Every method and window length must contain at least one rolling window. |
| valid_window_ratios | PASS | 0 | Rolling-window success ratios must lie between zero and one. |
| ordered_windows | PASS | 0 | Every rolling window must start before it ends. |
| positive_trading_days | PASS | 0 | Every rolling window must contain positive trading-day counts. |
