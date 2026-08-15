# Feature Dictionary

<!-- FUNDAMENTAL_FEATURES_START -->

# Fundamental features

The following fundamental features are constructed point-in-time using only accounting information that was available on or before each rebalance date.

## Transformation suffixes

- `_winsorized`: cross-sectionally winsorized by rebalance date.
- `_zscore`: standardized relative to all available companies on the same rebalance date.
- `_sector_zscore`: standardized relative to available companies in the same sector and rebalance date.
- `_missing`: binary indicator equal to 1 when the underlying raw factor is unavailable.

Missing accounting values are not automatically imputed.

## `earnings_yield`

**Family:** Value

**Formula:** Net Income TTM / Market Cap Proxy

**Interpretation:** Net accounting profit relative to the estimated equity market value.

## `sales_yield`

**Family:** Value

**Formula:** Revenue TTM / Market Cap Proxy

**Interpretation:** Sales generated relative to the estimated equity market value.

## `book_to_market`

**Family:** Value

**Formula:** Equity / Market Cap Proxy

**Interpretation:** Accounting book equity relative to the estimated market value of equity.

## `fcf_yield`

**Family:** Value

**Formula:** (Operating Cash Flow TTM - CAPEX TTM) / Market Cap Proxy

**Interpretation:** Free cash flow generated relative to the estimated equity market value.

## `roe`

**Family:** Quality

**Formula:** Net Income TTM / Equity

**Interpretation:** Accounting profitability generated on positive shareholder equity.

## `roa`

**Family:** Quality

**Formula:** Net Income TTM / Assets

**Interpretation:** Accounting profitability generated relative to total assets.

## `gross_profitability`

**Family:** Quality

**Formula:** Gross Profit TTM / Assets

**Interpretation:** Gross profit generated relative to the company's asset base.

## `gross_margin`

**Family:** Quality

**Formula:** Gross Profit TTM / Revenue TTM

**Interpretation:** Share of revenue remaining after direct cost of goods or services.

## `operating_margin`

**Family:** Quality

**Formula:** Operating Income TTM / Revenue TTM

**Interpretation:** Operating profit generated for each unit of revenue.

## `net_margin`

**Family:** Quality

**Formula:** Net Income TTM / Revenue TTM

**Interpretation:** Net accounting profit generated for each unit of revenue.

## `cash_conversion`

**Family:** Quality

**Formula:** Operating Cash Flow TTM / Net Income TTM

**Interpretation:** Relationship between operating cash generation and accounting earnings.

## `debt_to_assets`

**Family:** Leverage

**Formula:** (Current Debt + Non-current Debt) / Assets

**Interpretation:** Financial debt relative to the total asset base.

## `net_debt_to_assets`

**Family:** Leverage

**Formula:** (Total Debt - Cash) / Assets

**Interpretation:** Financial debt net of cash relative to total assets.

## `current_ratio`

**Family:** Solvency

**Formula:** Current Assets / Current Liabilities

**Interpretation:** Short-term assets available relative to short-term obligations.

## `interest_coverage`

**Family:** Solvency

**Formula:** Operating Income TTM / Interest Expense TTM

**Interpretation:** Operating earnings available relative to interest expense.

## `capex_to_assets`

**Family:** Investment

**Formula:** CAPEX TTM / Assets

**Interpretation:** Capital expenditure intensity relative to the company's asset base.

## `accruals`

**Family:** Accruals

**Formula:** (Net Income TTM - Operating Cash Flow TTM) / Assets

**Interpretation:** Difference between accounting earnings and operating cash generation relative to assets.

## `revenue_growth_yoy`

**Family:** Growth

**Formula:** Revenue TTM / Revenue TTM 12M Ago - 1

**Interpretation:** Year-over-year growth in trailing twelve-month revenue.

## `net_income_growth_yoy`

**Family:** Growth

**Formula:** (Net Income TTM - Net Income TTM 12M Ago) / abs(Net Income TTM 12M Ago)

**Interpretation:** Year-over-year improvement or deterioration in net income, allowing negative values.

## `operating_cash_flow_growth_yoy`

**Family:** Growth

**Formula:** (Operating Cash Flow TTM - Operating Cash Flow TTM 12M Ago) / abs(Operating Cash Flow TTM 12M Ago)

**Interpretation:** Year-over-year change in operating cash flow, allowing negative historical values.

## `asset_growth_yoy`

**Family:** Investment

**Formula:** Assets / Assets 12M Ago - 1

**Interpretation:** Year-over-year expansion or contraction of the total asset base.

## `revenue_growth_acceleration`

**Family:** Growth

**Formula:** Current Revenue Growth YoY - Revenue Growth YoY 12M Ago

**Interpretation:** Change in the company's year-over-year revenue growth rate.

## `net_income_growth_acceleration`

**Family:** Growth

**Formula:** Current Net Income Growth YoY - Net Income Growth YoY 12M Ago

**Interpretation:** Change in the company's year-over-year net-income growth rate.

## `operating_cash_flow_growth_acceleration`

**Family:** Growth

**Formula:** Current Operating Cash Flow Growth YoY - Operating Cash Flow Growth YoY 12M Ago

**Interpretation:** Change in the company's year-over-year operating cash-flow growth rate.

## Important construction conventions

- Market Cap Proxy uses historical close price multiplied by point-in-time shares outstanding, with quarterly diluted shares used only as a fallback.
- CAPEX is represented as a positive cash outflow; therefore Free Cash Flow equals Operating Cash Flow minus CAPEX.
- Missing debt components are not automatically interpreted as zero.
- Growth signals compare each company only with its own historical accounting information.
- Cross-sectional transformations are calculated independently for each rebalance date.

<!-- FUNDAMENTAL_FEATURES_END -->
