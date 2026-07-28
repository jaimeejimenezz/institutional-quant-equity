# Monthly Labels and Rebalancing Convention

## Purpose

The monthly label dataset transforms daily market observations
into one observation per company and rebalance date.

The target measures the adjusted return during the following
21 market sessions.

## Rebalance date

For every calendar month, `as_of_date` is the final observed
market session in the processed market dataset.

All features associated with that observation must use only
information available on or before `as_of_date`.

## Forecast horizon

Given an `as_of_date` denoted by `t`:

- `first_future_date` is the first market session after `t`.
- `target_end_date` is the twenty-first market session after `t`.
- The first realized return is dated `t+1`.
- The final realized return is dated `t+21`.

The absolute target is:

```text
target_21d =
    adjusted_close(t+21) / adjusted_close(t) - 1

This is equivalent to compounding the 21 close-to-close returns
whose timestamps run from t+1 through t+21.

Relative target

For every rebalance date, the cross-sectional median of the
available company targets is calculated.

target_21d_excess =
    target_21d - cross_sectional_median_21d

A positive value means that the company outperformed the median
company during the forecast horizon.

Top-quintile label

Companies are ordered from highest to lowest target_21d.

The top 20 percent receive:

label_top_quintile = 1

All remaining companies receive:

label_top_quintile = 0

The selected count is:

ceil(number_of_available_companies * 0.20)

Tickers are sorted alphabetically before ranking so ties are
resolved deterministically.

Missing observations

A company receives a label only if it has a valid positive
adjusted close on both:

as_of_date
target_end_date

No later price is substituted for a missing target-end price.

Months without a complete 21-session future horizon remain in
the rebalance calendar with:

has_full_horizon = false

They do not appear in the monthly labels.

Look-ahead protection

The following conditions must always hold:

first_future_date > as_of_date
target_end_date >= first_future_date

Targets may be used for model training and evaluation, but never
as input features.

The final months without 21 future market sessions are excluded
from model training.