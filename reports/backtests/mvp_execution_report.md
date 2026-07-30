# MVP Daily Execution Backtest — Step 8B

## Execution convention

- Signals are observed after the close of `as_of_date`.
- Trades are executed at the adjusted opening price of the following market session.
- Positions are valued daily using adjusted closing prices.
- Fractional shares are permitted in the MVP.
- Portfolio weights drift naturally between monthly rebalances.
- Transaction cost: `10.00` bps per dollar traded.

## Calendar

| signal_dates | first_signal_date | first_execution_date | last_signal_date | final_backtest_date |
| --- | --- | --- | --- | --- |
| 77 | 2020-01-31 | 2020-02-03 | 2026-05-29 | 2026-06-30 |

## Execution summary

| strategy_name | start_date | end_date | trading_days | rebalances | initial_capital | final_portfolio_value | preliminary_total_return | total_transaction_cost | total_traded_notional | mean_two_way_turnover | mean_one_way_turnover | mean_holdings | minimum_holdings | maximum_holdings | maximum_absolute_cash | maximum_absolute_cash_weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| elastic_net_score_weighted | 2020-02-03 | 2026-06-30 | 1610 | 77 | 1000000.000000 | 4353621.353674 | 3.353621 | 81174.579399 | 81174579.399473 | 0.509077 | 0.261274 | 25.000000 | 25 | 25 | 0.000000 | 0.000000 |
| elastic_net_top20_equal_weight | 2020-02-03 | 2026-06-30 | 1610 | 77 | 1000000.000000 | 4868540.214600 | 3.868540 | 93482.426063 | 93482426.063434 | 0.558561 | 0.286040 | 20.000000 | 20 | 20 | 0.000000 | 0.000000 |
| momentum_3m_top20_equal_weight | 2020-02-03 | 2026-06-30 | 1610 | 77 | 1000000.000000 | 2615746.903777 | 1.615747 | 91593.732369 | 91593732.368792 | 0.666107 | 0.339867 | 20.000000 | 20 | 20 | 0.000000 | 0.000000 |
| ridge_top20_equal_weight | 2020-02-03 | 2026-06-30 | 1610 | 77 | 1000000.000000 | 4377238.067522 | 3.377238 | 103356.085849 | 103356085.848993 | 0.615102 | 0.314339 | 20.000000 | 20 | 20 | 0.000000 | 0.000000 |
| universe_equal_weight | 2020-02-03 | 2026-06-30 | 1610 | 77 | 1000000.000000 | 2984877.591689 | 1.984878 | 8532.951189 | 8532951.189012 | 0.065703 | 0.039365 | 50.000000 | 50 | 50 | 0.000000 | 0.000000 |

## Important interpretation

The preliminary total return shown here is an accounting validation, not the final model selection criterion. Risk-adjusted performance, drawdowns, benchmarks and cost sensitivity are evaluated in Step 8C.
