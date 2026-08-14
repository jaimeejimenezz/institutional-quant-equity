# Processed fundamental feature report

## Step

Step 10D — Cross-sectional and sector fundamental transformations.

## Summary

- Rows: 7550
- Rebalance dates: 151
- Companies: 50
- Raw fundamental factors: 24
- Global z-score features: 24
- Sector z-score features: 24
- Missingness indicators: 24
- Duplicate date-ticker rows: 0
- Infinite transformed values: 0

## Latest coverage

```text
as_of_date                                  factor       version  companies_available  companies_total  coverage_ratio
2026-07-27                                accruals sector_zscore                   45               50            0.90
2026-07-27                        asset_growth_yoy sector_zscore                   45               50            0.90
2026-07-27                         cash_conversion sector_zscore                   45               50            0.90
2026-07-27          net_income_growth_acceleration sector_zscore                   45               50            0.90
2026-07-27                   net_income_growth_yoy sector_zscore                   45               50            0.90
2026-07-27                                     roa sector_zscore                   45               50            0.90
2026-07-27                              net_margin sector_zscore                   44               50            0.88
2026-07-27          operating_cash_flow_growth_yoy sector_zscore                   44               50            0.88
2026-07-27             revenue_growth_acceleration sector_zscore                   44               50            0.88
2026-07-27                      revenue_growth_yoy sector_zscore                   44               50            0.88
2026-07-27                          book_to_market sector_zscore                   43               50            0.86
2026-07-27                          earnings_yield sector_zscore                   43               50            0.86
2026-07-27 operating_cash_flow_growth_acceleration sector_zscore                   42               50            0.84
2026-07-27                                     roe sector_zscore                   42               50            0.84
2026-07-27                             sales_yield sector_zscore                   42               50            0.84
2026-07-27                         capex_to_assets sector_zscore                   38               50            0.76
2026-07-27                           current_ratio sector_zscore                   38               50            0.76
2026-07-27                        operating_margin sector_zscore                   35               50            0.70
2026-07-27                               fcf_yield sector_zscore                   34               50            0.68
2026-07-27                          debt_to_assets sector_zscore                   30               50            0.60
2026-07-27                      net_debt_to_assets sector_zscore                   30               50            0.60
2026-07-27                       interest_coverage sector_zscore                   28               50            0.56
2026-07-27                            gross_margin sector_zscore                   20               50            0.40
2026-07-27                     gross_profitability sector_zscore                   20               50            0.40
2026-07-27                        asset_growth_yoy        zscore                   50               50            1.00
2026-07-27          net_income_growth_acceleration        zscore                   50               50            1.00
2026-07-27                   net_income_growth_yoy        zscore                   50               50            1.00
2026-07-27                                     roa        zscore                   50               50            1.00
2026-07-27                                accruals        zscore                   49               50            0.98
2026-07-27                         cash_conversion        zscore                   49               50            0.98
2026-07-27                              net_margin        zscore                   49               50            0.98
2026-07-27             revenue_growth_acceleration        zscore                   49               50            0.98
2026-07-27                      revenue_growth_yoy        zscore                   49               50            0.98
2026-07-27                          book_to_market        zscore                   48               50            0.96
2026-07-27                          earnings_yield        zscore                   48               50            0.96
2026-07-27          operating_cash_flow_growth_yoy        zscore                   48               50            0.96
2026-07-27                                     roe        zscore                   47               50            0.94
2026-07-27                             sales_yield        zscore                   47               50            0.94
2026-07-27 operating_cash_flow_growth_acceleration        zscore                   46               50            0.92
2026-07-27                           current_ratio        zscore                   44               50            0.88
2026-07-27                         capex_to_assets        zscore                   43               50            0.86
2026-07-27                        operating_margin        zscore                   41               50            0.82
2026-07-27                               fcf_yield        zscore                   40               50            0.80
2026-07-27                          debt_to_assets        zscore                   37               50            0.74
2026-07-27                      net_debt_to_assets        zscore                   37               50            0.74
2026-07-27                       interest_coverage        zscore                   34               50            0.68
2026-07-27                            gross_margin        zscore                   23               50            0.46
2026-07-27                     gross_profitability        zscore                   23               50            0.46
```

## Methodology

- Winsorization is performed independently within each rebalance date.
- Global z-scores are calculated only using companies in the same rebalance date.
- Sector z-scores are calculated only using companies in the same date and sector.
- Missing raw factors remain missing. No imputation is performed.
- A binary missingness indicator is created for every fundamental factor.
- A factor with zero cross-sectional variation receives a neutral z-score of zero.