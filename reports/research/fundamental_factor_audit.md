# Fundamental factor audit

## Step

Step 10E — Final non-predictive audit of fundamental features.

## Dataset

- Rows: 7550
- Rebalance dates: 151
- Companies: 50
- Fundamental factors: 24
- Latest date: 2026-07-27
- Duplicate date-ticker rows: 0

## Z-score audit

- Global centering violations: 0
- Sector centering violations: 0

## Correlation diagnostics

- Factor pairs evaluated: 276
- High-correlation threshold: 0.80
- Highly correlated pairs: 3

High correlation is treated only as a redundancy warning. No factor is removed in Step 10E.

## Coverage summary

```text
                                 factor     family  latest_raw_coverage  overall_global_zscore_coverage  overall_sector_zscore_coverage coverage_tier
                               accruals   Accruals                 0.98                        0.894570                        0.800000          high
         net_income_growth_acceleration     Growth                 1.00                        0.808079                        0.720265          high
                  net_income_growth_yoy     Growth                 1.00                        0.888344                        0.792583          high
operating_cash_flow_growth_acceleration     Growth                 0.92                        0.704238                        0.622384          high
         operating_cash_flow_growth_yoy     Growth                 0.96                        0.798146                        0.710464          high
            revenue_growth_acceleration     Growth                 0.98                        0.787152                        0.702914          high
                     revenue_growth_yoy     Growth                 0.98                        0.865033                        0.772848          high
                       asset_growth_yoy Investment                 1.00                        0.902252                        0.804636          high
                        capex_to_assets Investment                 0.86                        0.810728                        0.709404      moderate
                         debt_to_assets   Leverage                 0.74                        0.672053                        0.528742      moderate
                     net_debt_to_assets   Leverage                 0.74                        0.672053                        0.528742      moderate
                        cash_conversion    Quality                 0.98                        0.894570                        0.800000          high
                           gross_margin    Quality                 0.46                        0.445563                        0.370066       limited
                    gross_profitability    Quality                 0.46                        0.445563                        0.370066       limited
                             net_margin    Quality                 0.98                        0.933377                        0.837219          high
                       operating_margin    Quality                 0.82                        0.763841                        0.646623      moderate
                                    roa    Quality                 1.00                        0.969139                        0.864901          high
                                    roe    Quality                 0.94                        0.934834                        0.830596          high
                          current_ratio   Solvency                 0.88                        0.861722                        0.736159      moderate
                      interest_coverage   Solvency                 0.68                        0.658543                        0.528079       limited
                         book_to_market      Value                 0.96                        0.940795                        0.834172          high
                         earnings_yield      Value                 0.96                        0.928742                        0.824503          high
                              fcf_yield      Value                 0.80                        0.697219                        0.574570      moderate
                            sales_yield      Value                 0.94                        0.902517                        0.802384          high
```

## Highly correlated pairs

```text
        factor_1           factor_2  dates_evaluated  mean_spearman  median_spearman  mean_abs_spearman  max_abs_spearman  same_sign_ratio
operating_margin         net_margin              151       0.898979         0.921854           0.898979          0.975701              1.0
  debt_to_assets net_debt_to_assets              151       0.894136         0.914889           0.894136          0.948832              1.0
             roe                roa              151       0.808850         0.814386           0.808850          0.908639              1.0
```

## Methodological boundary

Step 10E does not use future returns, target_21d, target_21d_excess, model performance or backtest results.

Predictive relevance and final feature selection are deferred to the modeling panel and walk-forward research stages.