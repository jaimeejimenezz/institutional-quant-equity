# Raw fundamental factor report

## Step

Step 10B — Raw fundamental factors.

## Summary

- Rows: 7550
- Rebalance dates: 151
- Companies: 50
- Raw factors: 17
- Latest market-cap proxy coverage: 48/50
- Infinite factor values: 0

## Share-count sources on latest date

```text
valuation_share_count_source
shares_outstanding        30
diluted_shares_quarter    18
missing                    2
```

## Latest factor coverage

```text
as_of_date              factor  companies_available  companies_total  coverage_ratio
2026-07-27                 roa                   50               50            1.00
2026-07-27            accruals                   49               50            0.98
2026-07-27     cash_conversion                   49               50            0.98
2026-07-27          net_margin                   49               50            0.98
2026-07-27      book_to_market                   48               50            0.96
2026-07-27      earnings_yield                   48               50            0.96
2026-07-27                 roe                   47               50            0.94
2026-07-27         sales_yield                   47               50            0.94
2026-07-27       current_ratio                   44               50            0.88
2026-07-27     capex_to_assets                   43               50            0.86
2026-07-27    operating_margin                   41               50            0.82
2026-07-27           fcf_yield                   40               50            0.80
2026-07-27      debt_to_assets                   37               50            0.74
2026-07-27  net_debt_to_assets                   37               50            0.74
2026-07-27   interest_coverage                   34               50            0.68
2026-07-27        gross_margin                   23               50            0.46
2026-07-27 gross_profitability                   23               50            0.46
```

## Important conventions

- Market capitalization is a proxy equal to historical close price multiplied by the latest point-in-time share count.
- shares_outstanding is preferred. Quarterly diluted_shares is used only as a fallback.
- CAPEX is stored as a positive cash outflow, therefore FCF = operating cash flow - CAPEX.
- Missing debt components are not assumed to be zero.
- No winsorization, standardization or imputation is performed in Step 10B.