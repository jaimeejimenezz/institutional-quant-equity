# Fundamental growth factor report

## Step

Step 10C — Fundamental growth and acceleration.

## Summary

- Rows: 7550
- Rebalance dates: 151
- Companies: 50
- Existing raw factors: 17
- New growth factors: 7
- Total raw fundamental factors: 24
- Infinite factor values: 0

## Latest growth coverage

```text
as_of_date                                  factor  companies_available  companies_total  coverage_ratio
2026-07-27                        asset_growth_yoy                   50               50            1.00
2026-07-27          net_income_growth_acceleration                   50               50            1.00
2026-07-27                   net_income_growth_yoy                   50               50            1.00
2026-07-27             revenue_growth_acceleration                   49               50            0.98
2026-07-27                      revenue_growth_yoy                   49               50            0.98
2026-07-27          operating_cash_flow_growth_yoy                   48               50            0.96
2026-07-27 operating_cash_flow_growth_acceleration                   46               50            0.92
```

## Methodology

- Revenue and asset growth use current / prior-year value - 1.
- Net-income and operating-cash-flow growth use change divided by the absolute prior-year value because these metrics may be negative.
- Acceleration equals current YoY growth minus the YoY growth observed one year earlier.
- No missing-value imputation, winsorization or standardization is performed in Step 10C.