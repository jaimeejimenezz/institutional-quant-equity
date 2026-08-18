# Research Results

## Purpose

This document summarizes the evidence produced by the frozen research release of the
Institutional Quant Equity Research Platform.

It is intentionally separated from `docs/methodology.md`.

- `methodology.md` explains how the research is designed.
- this document reports what the materialized research run produced.

The results below should be interpreted as historical out-of-sample research evidence, not as
a guarantee of future performance.

## Release scope

The current release covers:

- 50 US large-cap equities;
- 11 sectors;
- monthly research observations;
- 91 frozen predictors;
- expanding walk-forward validation;
- 77 out-of-sample monthly test folds;
- five portfolio-construction methods;
- explicit risk estimation;
- execution-aware gross and net backtesting;
- capacity, sensitivity, bootstrap, and ablation analysis;
- a dedicated data-quality and leakage-control framework.

The primary frozen signal period runs from:

```text
2020-01-31 through 2026-05-29
```

The evaluated portfolio history extends beyond the final signal date because the last signal
must still be held and evaluated over its forward horizon.

## Temporal boundaries

Different layers of the release end on different dates:

| Research layer | Last date |
|---|---|
| Frozen alpha signal | 2026-05-29 |
| Evaluated portfolio history | 2026-06-30 |
| Stored raw SPY benchmark data | 2026-07-27 |

These dates are intentionally different and reflect the distinction between signal formation,
future holding-period evaluation, and raw benchmark-data coverage.

## Frozen feature contract

The final modeling panel contains 91 predictors:

| Feature family | Count |
|---|---:|
| Technical | 19 |
| Fundamental global standardized | 24 |
| Fundamental sector standardized | 24 |
| Fundamental missingness indicators | 24 |
| **Total** | **91** |

The frozen full model remains the official research specification throughout robustness and
ablation analysis.

## Walk-forward evidence

The final evaluation protocol uses:

```text
mode: expanding
minimum training history: 60 months
validation window: 12 months
out-of-sample folds: 77
first OOS test date: 2020-01-31
last OOS test date: 2026-05-29
```

The cross-section for each month remains inside a single temporal fold.

The finalized walk-forward validation completed all configured readiness checks without a
failed readiness condition.

## Frozen alpha ensemble

The final alpha signal combines three frozen components:

- Technical Composite;
- Elastic Net;
- LightGBM Ranker.

The signal is persisted as:

```text
data/processed/final_alpha_signal.parquet
```

The ensemble is evaluated only after its component and weighting logic have been frozen.

## Predictive ranking evidence

The frozen signal produced positive average cross-sectional ranking evidence over the OOS
period.

Rounded release-level statistics are:

| Statistic | Result |
|---|---:|
| Mean monthly Information Coefficient | ~0.041 |
| 95% bootstrap interval for mean IC | ~[-0.002, 0.082] |
| Bootstrap probability mean IC > 0 | ~96.8% |
| Mean top-minus-bottom ranking spread | ~1.23% |
| Bootstrap probability spread > 0 | ~99.7% |

These statistics should be read together rather than independently.

The point estimate for mean IC is positive, and the bootstrap distribution places most of its
mass above zero. However, the reported 95% interval still overlaps zero. The top-minus-bottom
spread provides stronger directional evidence in the current sample, with a bootstrap
probability of a positive spread close to 100%.

This is evidence of economically useful ranking structure, not proof that every future month
will have a positive IC.

## Horizon robustness

The platform evaluates predictive behavior over multiple forward horizons.

Representative OOS ranking results remain positive across the tested horizons, including
approximately:

| Horizon | Mean IC |
|---|---:|
| 10 sessions | ~0.050 |
| 21 sessions | ~0.044 |
| 42 sessions | ~0.053 |

The corresponding annualized IC Information Ratio is also positive, with representative
values around:

```text
21 sessions: ~0.73
42 sessions: ~1.04
```

The exact horizon statistics belong to the materialized robustness tables; the relevant
research conclusion is that predictive ranking quality is not isolated to one exact 21-session
definition.

## Quintile and precision evidence

The predictive evaluation includes:

- quintile return profiles;
- top-minus-bottom spread;
- top-quintile precision;
- yearly stability;
- sector stability.

Representative top-quintile precision lies around:

```text
~0.26 to ~0.28
```

This is above the 20% unconditional rate implied by a five-bucket cross-section, but the metric
is not interpreted alone. It is used together with continuous IC and spread evidence.

## Model comparison

The research compares transparent and flexible model families rather than assuming that the
most complex model must dominate.

The evaluated model stack includes:

- simple technical and composite baselines;
- regularized linear models;
- LightGBM regression;
- LightGBM ranking;
- the frozen multi-model ensemble.

The final ensemble was selected before the robustness stage and remains frozen during
subsequent diagnostics.

Ablation results are therefore interpreted as evidence about dependence on components, not as
permission to retrospectively replace the official full model.

## Portfolio-construction comparison

The platform converts the same frozen alpha research into five portfolio-construction methods:

1. `top_n_equal_weight`
2. `score_weighted`
3. `alpha_risk_turnover`
4. `cvar`
5. `median_mad_de`

All methods are evaluated under a common execution and reporting framework.

The common portfolio controls include:

```text
long-only
fully invested
maximum security weight: 5%
maximum sector weight: 25%
minimum diversification: 20 positions
```

The advanced methods additionally incorporate risk, turnover, beta, liquidity, or tail-risk
controls where appropriate.

## Score Weighted OOS snapshot

The dashboard uses Score Weighted as a transparent reference view for the executed OOS
portfolio.

Rounded release-level statistics are:

| Metric | Score Weighted |
|---|---:|
| Net CAGR | ~24.4% |
| Net Sharpe ratio | ~1.09 |
| Maximum drawdown | ~-34.3% |
| Alpha versus SPY | ~+6.7% |
| Beta versus SPY | ~1.06 |
| Mean one-way turnover | ~25.9% |
| Effective execution cost | ~5.14 bps |

These figures are rounded dashboard-level summaries and should not be treated as exact
replacements for the underlying machine-readable performance tables.

The Score Weighted method is shown because it is interpretable and is the default dashboard
reference. Its presentation does not imply that the research selected it retrospectively as
the universally optimal portfolio constructor.

## Interpretation of portfolio performance

The portfolio evidence suggests that the ranking signal can be translated into economically
meaningful long-only portfolios after explicit constraints and execution costs.

However, the performance statistics must be interpreted alongside:

- market beta near one;
- substantial historical drawdown;
- turnover;
- transaction costs;
- finite sample length;
- static-universe bias;
- capacity assumptions.

The backtest should therefore be interpreted as an execution-aware historical simulation, not
as evidence of a risk-free or market-neutral strategy.

## Risk results

The risk layer produces security-level and portfolio-level diagnostics for every rebalance.

The security model includes:

- annualized realized volatility;
- beta versus SPY;
- liquidity / Average Dollar Volume;
- shrinkage covariance estimates.

The covariance engine materialized one 50x50 covariance matrix for each OOS rebalance date,
for 77 matrices in the current release.

The average estimated shrinkage intensity is approximately:

```text
0.073
```

This indicates that the covariance estimator applies a meaningful but not dominant amount of
shrinkage relative to the raw sample covariance.

Portfolio diagnostics verify the implemented security and sector limits and provide effective
position counts, beta, concentration, and risk summaries for each construction method.

## Portfolio diversification

Representative average position counts from the construction layer are approximately:

| Method | Average positions |
|---|---:|
| Top-N Equal Weight | 25.0 |
| Score Weighted | 25.0 |
| Alpha-Risk-Turnover | ~21.4 |
| CVaR | ~23.4 |
| Median-MAD DE | ~21.3 |

The optimized methods therefore use the allowed flexibility to concentrate somewhat relative
to the simple 25-name baselines while remaining above the minimum-diversification contract.

## Execution evidence

The execution engine simulates portfolio holdings, trades, and costs rather than converting
target weights directly into performance.

The advanced cost model includes:

```text
commission: 0.5 bps
half-spread: 2.0 bps
slippage: 2.5 bps
market impact coefficient: 0.10
```

The research persists separate gross and net portfolio histories:

```text
data/processed/backtest_all_methods_gross_daily.parquet
data/processed/backtest_all_methods_net_daily.parquet
```

and corresponding positions and trades:

```text
data/processed/positions_all_methods_net.parquet
data/processed/trades_all_methods_net.parquet
```

This separation makes execution drag directly inspectable.

## Capacity analysis

Capacity is evaluated at multiple portfolio capital levels:

```text
$100K
$1M
$10M
$100M
```

The purpose is to measure how liquidity participation and market impact change as the same
research portfolio is scaled.

The analysis is scenario-based. It does not claim that the largest tested capital level could
be deployed without additional live execution, borrow, market-impact, and operational
analysis.

## Transaction-cost sensitivity

The robustness framework evaluates multiple fixed cost assumptions, including:

```text
0 bps
5 bps
10 bps
20 bps
50 bps
```

It also evaluates the advanced liquidity-dependent execution model.

The economic conclusion is therefore not dependent on one single favorable transaction-cost
assumption.

## Portfolio-parameter sensitivity

The portfolio layer is stressed across multiple:

- top-N selections;
- security weight caps;
- related construction settings.

The objective is not to identify the best ex-post parameter combination. It is to determine
whether the broad conclusion is stable under plausible neighboring specifications.

## Rebalance-frequency sensitivity

Monthly and lower-frequency portfolio rebalancing are compared.

This evaluates whether performance is dependent on an exact monthly trading cadence and
quantifies the trade-off between signal responsiveness and lower turnover.

## Rolling-window robustness

The research evaluates rolling performance windows including:

```text
12 months
24 months
36 months
```

This reduces reliance on a single full-period summary and exposes periods where the strategy
is materially stronger or weaker.

## Bootstrap evidence

The robustness framework includes 10,000 monthly bootstrap replications.

The bootstrap analysis covers:

- portfolio return distributions;
- strategy ranking stability;
- pairwise strategy comparisons;
- final signal IC;
- final signal ranking spread.

Bootstrap results are interpreted as sampling-uncertainty evidence under the observed OOS
return sequence, not as an additional independent historical sample.

## Temporal and regime robustness

Performance is decomposed by:

- calendar year;
- market regime;
- conditional months;
- rolling windows.

This is intended to identify whether the aggregate result is dominated by one isolated market
environment.

The current robustness framework reports successful completion of the temporal-regime suite.

## Sector stability

The final alpha signal is evaluated separately across sectors.

This diagnostic tests whether the ranking result is broadly distributed or dominated by one
industry group.

Sector evidence forms part of the statistical robustness layer and is displayed independently
from the aggregate IC.

## Ensemble ablation

The platform removes individual ensemble components and recomputes the restricted signals.

These experiments answer whether the frozen signal depends excessively on one modeling family.

The resulting comparisons are diagnostic only. They do not trigger retrospective ensemble
reselection.

## Feature-family ablation

The frozen feature contract contains:

```text
91 total predictors
72 fundamental predictors
19 technical predictors
```

Two principal feature-family experiments are materialized:

### No fundamentals

```text
19 predictors
```

This specification removes the 72 fundamental predictors and retains the technical feature
family.

### No momentum

```text
85 predictors
```

This specification removes the defined six-feature momentum / return / reversal family while
retaining the remaining technical and fundamental predictors.

The ablation research includes:

- feature-contract verification;
- model retraining under the restricted feature set;
- frozen predictive comparison;
- economic portfolio comparison;
- paired bootstrap diagnostics.

## Ablation interpretation

Some restricted specifications can produce isolated metrics that look stronger than the full
frozen model over the observed OOS sample.

That does not invalidate the frozen research decision.

Promoting an ablation after seeing the same OOS results would amount to selecting on the test
set. The correct conclusion is therefore about dependency and robustness, not about
retrospectively replacing the model.

## Robustness completion

The final robustness audit contains:

```text
17 check suites
17 passed suites
0 failed suites
```

Coverage across the defined research dimensions is:

```text
12 of 13 dimensions complete
```

The only deferred dimension is:

```text
expanded_universe = DEFERRED_LIMITATION
```

This is not recorded as a failed test.

A genuine expanded-universe experiment requires adding new securities and reconstructing their
full point-in-time data, features, labels, walk-forward predictions, risk, portfolio, execution,
and robustness artifacts.

Treating this as a documented limitation is more methodologically correct than simulating
expanded breadth using only the existing 50-name universe.

## Data-quality results

The institutional data-quality framework reports:

```text
8 of 8 validation families passed
132 of 132 controls passed
0 violations
0 escalations
```

The validation families cover the major pipeline boundaries, including:

- universe;
- market data;
- labels;
- technical features;
- fundamentals;
- modeling panel;
- walk-forward evaluation;
- reporting/dashboard contracts.

These results support the integrity of the materialized research run, but they do not imply
that the data source or research design is free of every possible bias.

## Leakage-control evidence

Leakage prevention is validated at multiple levels:

- point-in-time fundamental availability;
- technical windows ending at `as_of_date`;
- future targets excluded from predictors;
- date-grouped temporal folds;
- past-only fold preprocessing;
- test-target isolation;
- frozen final ensemble semantics;
- predictive evaluation contracts.

The modeling-panel final audit and walk-forward readiness layers completed without a failed
readiness condition in the frozen release.

## Portable reporting release

Nine processed artifacts are intentionally versioned for the portable dashboard:

```text
data/processed/backtest_all_methods_gross_daily.parquet
data/processed/backtest_all_methods_net_daily.parquet
data/processed/benchmark_spy_daily.parquet
data/processed/final_alpha_signal.parquet
data/processed/modeling_panel.parquet
data/processed/positions_all_methods_net.parquet
data/processed/risk_estimates.parquet
data/processed/target_weights_all_methods.parquet
data/processed/trades_all_methods_net.parquet
```

Together they represent approximately 12.76 MB of curated research state.

They are sufficient to reproduce the interactive dashboard without rerunning the complete SEC,
model-training, optimization, and robustness pipeline.

## Research evidence hierarchy

The current release should be read in the following order:

```text
data quality and temporal integrity
    -> OOS predictive ranking evidence
    -> frozen ensemble
    -> risk-aware portfolio construction
    -> execution-aware net performance
    -> sensitivity and capacity
    -> bootstrap and ablations
    -> limitations
```

A favorable final CAGR without the earlier validation layers would be insufficient evidence.

## What the results support

The current frozen release supports the following research conclusions:

1. The 91-feature point-in-time modeling panel contains measurable out-of-sample
   cross-sectional ranking information.
2. The final ensemble has positive average IC and positive top-minus-bottom spread over the OOS
   period.
3. Ranking evidence remains positive under alternative prediction horizons.
4. The frozen signal can be translated into constrained long-only portfolios.
5. Execution-aware net results remain economically meaningful after explicit cost modeling.
6. The result is not dependent on one portfolio constructor, one cost assumption, or one
   isolated robustness test.
7. Data-quality, leakage, portfolio, execution, and robustness contracts pass in the current
   materialized release.
8. The remaining expanded-universe limitation is explicitly documented rather than hidden.

## What the results do not support

The current release does **not** establish that:

- the strategy will reproduce its historical CAGR in the future;
- the mean IC is statistically certain to remain positive;
- the portfolio is market neutral;
- the static 50-name universe is free of survivorship bias;
- execution at $100M is guaranteed;
- yfinance data is equivalent to institutional-grade licensed market data;
- the backtest reproduces historical order-book fills;
- an ablation that looks stronger ex post should replace the frozen model;
- the research constitutes a live investment track record.

These distinctions are essential to the interpretation of the project.

## Principal limitations

The most important limitations are:

### Static historical universe

Current constituents are applied historically. This introduces survivorship and membership
bias.

### Limited breadth

The cross-section contains 50 securities, which is adequate for the current platform but
smaller than a broad institutional equity universe.

### Finite OOS history

The frozen evaluation contains 77 monthly test folds. This is substantial for the project but
still leaves statistical uncertainty.

### Data-source quality

The market-data source is suitable for research but is not an institutional licensed feed.

### Fundamental mapping risk

Issuer XBRL reporting choices can create residual heterogeneity even after canonicalization.

### Execution approximation

Commissions, spread, slippage, and market impact are modeled rather than reconstructed from
historical intraday order-book data.

### No live track record

All returns are simulated historical research results.

## Overall assessment

The strongest aspect of the current release is not any single performance metric.

The stronger conclusion is that the project connects:

- point-in-time data engineering;
- a frozen 91-feature research contract;
- date-grouped walk-forward modeling;
- statistically evaluated cross-sectional alpha;
- risk-aware portfolio construction;
- explicit execution costs;
- capacity analysis;
- robustness and ablation testing;
- institutional-style data-quality controls;
- a portable reporting layer.

The evidence is therefore broader than a conventional backtest based on one signal and one
portfolio rule.

At the same time, the confidence interval around mean IC, the static universe, the finite OOS
sample, and the absence of live execution evidence justify a cautious interpretation.

The platform should be viewed as a completed institutional-style quantitative research
prototype with positive historical OOS evidence and explicit research-governance controls,
not as a claim of guaranteed future investment performance.

## Source artifacts

The principal machine-readable and human-readable evidence is stored under:

```text
data/processed/
reports/models/
reports/risk/
reports/portfolio/
reports/backtests/
reports/execution/
reports/robustness/
reports/data_quality/
reports/validation/
reports/dashboard/
reports/tables/
```

The dashboard provides an interactive view over the same materialized research state.

## Related documentation

- `README.md` — portfolio-level project overview.
- `docs/architecture.md` — end-to-end system architecture and artifact lineage.
- `docs/reproducibility.md` — installation and full reconstruction workflow.
- `docs/methodology.md` — quantitative research design and governance.
- `docs/DATA_DICTIONARY.md` — dataset and column semantics.
- `docs/FEATURE_DICTIONARY.md` — feature definitions.
- `docs/UNIVERSE_METHODOLOGY.md` — universe design and survivorship limitation.

Together these documents separate architecture, reproducibility, methodology, and empirical
results so that each layer can be reviewed independently.
