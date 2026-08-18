# Quantitative Research Methodology

## Purpose and research question

The Institutional Quant Equity Research Platform studies whether a cross-sectional combination
of technical and point-in-time fundamental information can produce a persistent equity-ranking
signal and whether that signal can be converted into implementable portfolios after explicit
risk, concentration, turnover, liquidity, and transaction-cost controls.

The platform is designed as a research system rather than as a single predictive model. The
methodology therefore separates five questions:

1. **Can the feature set rank future relative equity returns out of sample?**
2. **Is the predictive evidence stable across time, sectors, and model families?**
3. **Can the ranking be translated into diversified portfolios under realistic constraints?**
4. **How much of the apparent performance survives execution costs and capacity assumptions?**
5. **Do the conclusions remain qualitatively stable under sensitivity, bootstrap, and ablation
   analysis?**

The research objective is not to maximize a single backtest metric retrospectively. The
methodology emphasizes temporal integrity, frozen model decisions, auditable stage outputs,
and robustness of conclusions.

## Research universe

The configured universe contains 50 US large-cap equities distributed across 11 sectors.

The current universe contract is:

```text
construction_method: static_curated_large_cap
membership_mode: current_constituents_applied_historically
expected_count: 50
market: US
currency: USD
```

This design provides a stable cross-section for controlled model comparison, but it does not
represent historical index membership. The resulting survivorship and membership limitations
are explicitly documented and are not hidden inside the backtest.

A genuine expanded-universe robustness experiment would require adding new securities and
reconstructing their full point-in-time market and accounting history. The current robustness
framework therefore records expanded-universe coverage as a deferred limitation rather than
simulating it through exclusions from the same 50-name research set.

## Research timeline and observation unit

The configured research history begins on `2014-01-01`.

The core observation unit is a monthly security cross-section indexed by:

```text
(as_of_date, ticker)
```

The research calendar uses the last available market session of each rebalance month as the
monthly observation date.

The final modeling panel spans 151 monthly dates. The frozen out-of-sample signal begins on
2020-01-31 and ends on 2026-05-29, producing 77 monthly out-of-sample test folds.

Several terminal dates coexist in the current release because the prediction, holding, and raw
market-data layers serve different purposes:

| Meaning | Current release |
|---|---|
| Last frozen alpha signal date | 2026-05-29 |
| Last evaluated portfolio day | 2026-06-30 |
| Last stored raw SPY market date | 2026-07-27 |

These dates should not be interpreted as conflicting cutoffs.

## Prediction target

### Primary continuous target

The primary modeling target is the 21-session forward return relative to the cross-sectional
median:

\[
y_{i,t}
=
R_{i,t+1:t+21}
-
\operatorname{median}_{j \in U_t}
\left(
R_{j,t+1:t+21}
\right)
\]

where:

- \(i\) is the security;
- \(t\) is the monthly `as_of_date`;
- \(U_t\) is the eligible cross-section at date \(t\);
- \(R_{i,t+1:t+21}\) is the security's forward 21-session return.

The target is relative rather than purely absolute because the research problem is
cross-sectional security selection. The model is therefore rewarded for ranking securities
within each monthly opportunity set rather than for forecasting the market direction.

### Supporting labels

The label pipeline also stores:

- absolute 21-session forward return;
- a top-quintile classification derived from the monthly cross-section.

The binary label is used for ranking diagnostics such as top-quintile precision. It is not a
replacement for the continuous relative-return target.

## Information timing

The methodology enforces the following temporal order:

```text
information available by as_of_date
    -> feature construction
    -> signal formation
    -> portfolio target
    -> future holding / evaluation period
    -> realized target and performance
```

No target information is eligible to influence feature construction, model fitting, fold
preprocessing, hyperparameter selection, or portfolio selection for the same test month.

The future-return target starts after the observation date. Point-in-time fundamentals are
selected using filing availability rather than fiscal-period end dates alone.

## Market data

Daily market data is obtained through the project's market-data provider abstraction, currently
configured around yfinance.

The market pipeline preserves and validates daily fields required for:

- historical returns;
- momentum and reversal;
- realized volatility;
- drawdown;
- moving averages;
- liquidity;
- execution simulation;
- benchmark comparison;
- covariance estimation.

Validation includes:

- unique date/ticker keys;
- minimum historical coverage;
- missing-price controls;
- invalid price checks;
- extreme-return diagnostics;
- adjustment-factor behavior;
- provider and schema consistency.

The processed daily market dataset is persisted as:

```text
data/processed/market_daily.parquet
```

## Point-in-time SEC fundamentals

Fundamental features are built from SEC EDGAR Companyfacts.

The methodology does not attach accounting information to a historical month merely because a
financial period had ended. A financial observation becomes eligible only after it was
actually available to the market.

### Fundamental processing chain

The pipeline follows this sequence:

```text
SEC Companyfacts
    -> normalized XBRL observations
    -> canonical financial concepts
    -> duration and statement classification
    -> point-in-time availability selection
    -> quarterly / TTM reconstruction
    -> raw fundamental factors
    -> growth and acceleration factors
    -> cross-sectional transformations
    -> missingness indicators
```

### Availability rule

The configured point-in-time layer applies:

```text
availability_lag_days: 1
require_statement_type_match: true
exclude_other_duration: true
```

This prevents the feature layer from using an accounting observation before its filing-based
availability date.

### Canonicalization

SEC issuers can report economically similar quantities under different XBRL concepts. The
canonicalization layer maps source tags into normalized research concepts before factor
construction.

Duration-sensitive concepts are classified using configured ranges for:

- quarter;
- half-year;
- nine-month;
- annual observations.

The methodology then reconstructs quarterly or trailing-twelve-month values only where the
underlying accounting semantics permit it.

## Fundamental factors

The frozen feature contract contains 72 fundamental predictors:

- 24 globally standardized factors;
- 24 sector-standardized versions of the same factors;
- 24 missingness indicators.

The raw factor families represent economic dimensions such as:

- valuation;
- profitability;
- operating quality;
- cash flow;
- leverage;
- investment;
- accrual behavior;
- balance-sheet structure;
- fundamental growth;
- growth acceleration.

The exact factor definitions and column-level semantics are documented in:

- `docs/FEATURE_DICTIONARY.md`
- `docs/DATA_DICTIONARY.md`

### Cross-sectional transformation

Fundamental transformations are performed separately by monthly cross-section.

The configured transformation contract uses:

```text
winsor_lower_quantile: 0.025
winsor_upper_quantile: 0.975
zscore_ddof: 0
```

Two standardized representations are retained:

1. **Global z-score** — relative standing within the full monthly cross-section.
2. **Sector z-score** — relative standing within the security's sector.

The sector representation reduces the risk that the model merely learns persistent accounting
differences between industries.

### Missingness indicators

Missing accounting information is not silently treated as economically neutral. The feature
contract includes one missingness indicator per fundamental factor.

This allows the predictive layer to distinguish a genuinely average standardized observation
from a value that is unavailable and handled through the preprocessing contract.

## Technical factors

The frozen feature contract contains 19 technical predictors.

Configured families include:

- 12-month momentum excluding the most recent month;
- 6-month momentum excluding the most recent month;
- 3-month return;
- 1-month return;
- 1-week return;
- reversal signals;
- short- and long-horizon volatility;
- drawdown;
- short- and long-moving-average relationships;
- liquidity and trading-activity measures.

Historical windows use data ending at the current `as_of_date`.

### Winsorization and sector neutralization

Technical feature processing uses:

```text
winsor_lower_quantile: 0.01
winsor_upper_quantile: 0.99
sector_neutralization: true
```

The cross-sectional processing layer first controls extreme observations and then removes the
sector component where appropriate.

Sector-neutral technical signals help distinguish security-specific ranking information from
broad sector momentum or sector-level market regimes.

## Final feature contract

The frozen modeling panel contains 91 predictors:

| Feature family | Count |
|---|---:|
| Technical | 19 |
| Fundamental global standardized | 24 |
| Fundamental sector standardized | 24 |
| Fundamental missingness | 24 |
| **Total** | **91** |

The distinction between the frozen feature contract and robustness ablations is important.
Ablation experiments remove feature families for diagnostic purposes, but they do not redefine
the full model after the out-of-sample evaluation has been observed.

## Modeling panel

The canonical modeling panel joins monthly features, targets, and security metadata.

Each row corresponds to one security at one monthly observation date.

The modeling panel contains:

- `as_of_date`;
- ticker;
- sector;
- 91 predictors;
- continuous forward-return target;
- absolute forward return;
- top-quintile label.

The panel is persisted as:

```text
data/processed/modeling_panel.parquet
```

Before it is admitted to model training, dedicated audit layers validate:

- key uniqueness;
- expected cross-sectional size;
- date alignment;
- feature coverage;
- target availability;
- point-in-time fundamental timing;
- absence of future-dependent feature columns;
- compatibility with walk-forward evaluation.

## Leakage prevention

Leakage prevention is treated as a system-level contract rather than as a single code check.

### Feature leakage controls

All rolling technical features terminate at the observation date.

Fundamental observations are restricted to information available by the point-in-time
availability rule.

Future-return columns are isolated as labels and are not eligible predictors.

### Fold leakage controls

The full monthly cross-section belongs to one date-grouped fold. Securities from the same month
are never split between train and test simply because rows are shuffled.

Fit-dependent preprocessing is learned only from the eligible historical fold data.

Examples include:

- imputation;
- scaling;
- model coefficient estimation;
- model hyperparameter selection;
- ranking-model training.

### Target isolation controls

Dedicated audit scripts verify that test-period targets do not influence the predictions
generated for that same period.

### Research-governance controls

The final ensemble is frozen before robustness and ablation conclusions are interpreted.

This prevents a common second-order form of leakage: using the observed out-of-sample period
to choose a new "best" model and then continuing to describe that period as untouched
out-of-sample evidence.

## Walk-forward validation

### Why walk-forward evaluation

A random train/test split would violate the chronological structure of the problem and create
unrealistic information overlap.

The final evaluation therefore uses expanding walk-forward validation grouped by monthly
`as_of_date`.

### Fold structure

The configured contract uses:

```text
mode: expanding
minimum_training_months: 60
validation_months: 12
out_of_sample_start: 2020
```

The finalized research run contains:

```text
77 out-of-sample folds
first test date: 2020-01-31
last test date: 2026-05-29
```

A purge boundary is applied around the train/validation/test transition to avoid contamination
from overlapping forward-return horizons.

### Conceptual fold

For test month \(t\):

```text
historical training window
    -> validation window
    -> purge boundary
    -> test cross-section at t
```

The training window expands as time progresses, while the validation horizon remains
historical relative to the test month.

## Model families

The platform compares transparent baselines with regularized linear and nonlinear ranking
models.

### Baseline models

Baseline models establish whether the machine-learning layer adds value beyond simple research
heuristics.

The baseline family includes technical and composite ranking signals, including momentum-based
comparators.

### Ridge

Ridge regression estimates a linear mapping from the standardized feature vector to the
continuous relative-return target while shrinking coefficients toward zero.

For coefficients \(\beta\), the objective is:

\[
\min_{\beta}
\left[
\sum_i (y_i - x_i^\top\beta)^2
+
\lambda \lVert \beta \rVert_2^2
\right]
\]

Ridge is useful as a stable linear benchmark when many correlated features are present.

### Elastic Net

Elastic Net combines \(L_1\) and \(L_2\) regularization:

\[
\min_{\beta}
\left[
\sum_i (y_i - x_i^\top\beta)^2
+
\lambda
\left(
\rho \lVert \beta \rVert_1
+
(1-\rho)\lVert \beta \rVert_2^2
\right)
\right]
\]

The validation process selects the regularization parameters using historical data inside each
eligible walk-forward fold.

The \(L_1\) component allows sparse feature selection while the \(L_2\) component stabilizes
groups of correlated predictors.

### LightGBM regression

The regression model captures nonlinear interactions and threshold effects that are not
available to the linear specifications.

Hyperparameters are selected from historical validation data within the walk-forward
framework.

### LightGBM ranking

The ranking model is directly aligned with the cross-sectional objective. Securities are
grouped by monthly observation date so that the model learns relative ordering within each
cross-section.

The ranking specification is especially relevant because the portfolio layer consumes a
cross-sectional alpha ranking rather than an isolated absolute return forecast.

## Model evaluation

Predictive performance is evaluated at the monthly cross-sectional level.

### Information Coefficient

The primary ranking statistic is Spearman Information Coefficient:

\[
IC_t
=
\rho_s
\left(
\hat{s}_{i,t},
y_{i,t}
\right)
\]

where \(\hat{s}_{i,t}\) is the model score and \(y_{i,t}\) is the realized relative-return
target.

A positive IC indicates that higher predicted rankings are associated with better subsequent
relative returns.

### IC Information Ratio

The stability of monthly IC is summarized with an annualized Information Ratio:

\[
IR_{IC}
=
\frac{\overline{IC}}
{\sigma(IC)}
\sqrt{12}
\]

This measures consistency of ranking quality across monthly observations.

### Quintile analysis

Each month, securities are sorted by predicted score and divided into ranking buckets.

The methodology examines:

- mean future return by quintile;
- monotonicity of the quintile profile;
- top-minus-bottom spread;
- stability of the spread through time.

### Top-quintile precision

For the highest-ranked subset, the platform measures how often selected names belong to the
realized top quintile.

This diagnostic is intuitive but is interpreted alongside continuous ranking metrics rather
than alone.

### Stability diagnostics

Predictive evidence is decomposed by:

- year;
- sector;
- model;
- subperiod;
- feature family.

This helps distinguish a broadly distributed alpha signal from one driven by a small number of
dates or industries.

## Frozen ensemble

The final signal combines three frozen predictive components:

- Technical Composite;
- Elastic Net;
- LightGBM Ranker.

The components are transformed into a common cross-sectional ranking representation before
combination.

The final signal is persisted as:

```text
data/processed/final_alpha_signal.parquet
```

### Why freeze the ensemble

The final ensemble is a governance decision.

Once the complete out-of-sample period has been observed, subsequent robustness experiments
may challenge the signal, but they may not be used to redesign the ensemble and then reuse the
same OOS period as if it were an untouched test set.

The frozen full model therefore remains the official research specification even when a
diagnostic ablation happens to produce a stronger realized portfolio metric over the same
sample.

## Statistical robustness of the signal

The final predictive signal is evaluated through monthly and cross-sectional robustness
procedures.

The statistical layer considers:

- average monthly IC;
- bootstrap distribution of mean IC;
- probability of positive mean IC;
- top-minus-bottom spread;
- bootstrap distribution of the spread;
- yearly stability;
- sector stability.

The purpose of the bootstrap is not to create additional observations. It measures sampling
uncertainty under resampling assumptions applied to the observed OOS sequence.

A result is interpreted jointly with its confidence interval and distribution rather than only
through its point estimate.

## Risk model

The risk layer is estimated independently from the alpha model.

### Security-level risk

Configured inputs include:

```text
volatility_window_sessions: 252
beta_window_sessions: 252
liquidity_window_sessions: 60
minimum_return_observations: 126
minimum_liquidity_observations: 20
annualization_factor: 252
```

The security-level risk dataset contains variables used for portfolio diagnostics and
constraints, including:

- annualized volatility;
- beta relative to SPY;
- liquidity / Average Dollar Volume measures.

### Covariance estimation

The covariance model uses:

```text
covariance_window_sessions: 252
covariance_minimum_observations: 126
covariance_method: ledoit_wolf
```

Ledoit-Wolf shrinkage is used because the empirical covariance matrix can be unstable when the
cross-section is large relative to the available rolling sample.

The estimator shrinks the noisy sample covariance toward a more structured target, improving
numerical conditioning for portfolio optimization.

## Portfolio construction philosophy

The research distinguishes alpha quality from portfolio-construction quality.

A predictive model can rank securities correctly but still produce a poor investment
portfolio if the mapping to weights creates excessive concentration, turnover, beta, liquidity
risk, or execution cost.

The platform therefore compares five portfolio methods under a common risk and execution
framework:

1. `top_n_equal_weight`
2. `score_weighted`
3. `alpha_risk_turnover`
4. `cvar`
5. `median_mad_de`

## Common portfolio constraints

The common portfolio contract is:

```text
long_only: true
fully_invested: true
maximum_security_weight: 0.05
maximum_sector_weight: 0.25
minimum_positions: 20
```

Advanced optimizers can also apply:

```text
minimum_portfolio_beta: 0.85
maximum_portfolio_beta: 1.15
max_position_adv_fraction: 0.01
```

These constraints prevent the optimizer from converting a ranking signal into a highly
concentrated or economically implausible solution.

## Top-N equal weight

The top-ranked securities are selected and assigned equal weights subject to the portfolio
contract.

This method is deliberately simple and provides a low-model-risk benchmark for the more
complex weighting schemes.

Its principal advantage is interpretability. Its main limitation is that it ignores score
magnitude, covariance, transaction costs, and cross-sectional differences in risk.

## Score-weighted portfolio

The score-weighted method selects a candidate set from the highest-ranked securities and
allocates larger weights to stronger positive signals while respecting concentration and
sector constraints.

This method provides an intermediate specification between equal weighting and full numerical
optimization.

## Alpha-risk-turnover optimization

The constrained optimizer balances expected alpha, covariance risk, and trading activity.

Conceptually:

\[
\max_w
\left(
\hat{\alpha}^{\top} w
-
\lambda w^\top \Sigma w
-
\eta \lVert w-w_{t-1}\rVert_1
\right)
\]

subject to the portfolio constraints.

The configured optimizer uses:

```text
annualized_alpha_scale: 0.10
risk_aversion: 0.50
turnover_penalty: 0.01
```

The previous portfolio enters the objective through the turnover penalty, making the
optimization path-dependent in an economically meaningful way.

## CVaR portfolio

Conditional Value at Risk focuses on the adverse tail of the scenario distribution rather than
only variance.

For confidence level \(\alpha\), CVaR can be interpreted as the expected loss conditional on
being in the worst \(1-\alpha\) fraction of outcomes.

The configured portfolio layer uses:

```text
confidence_level: 0.95
horizon_days: 21
scenario_lookback: 252
minimum_scenarios: 126
cvar_penalty: 0.05
```

The CVaR construction is evaluated under the same broad concentration framework as the other
portfolio methods.

## Median-MAD Differential Evolution

The Median-MAD portfolio is the robust optimization method inherited conceptually from the
earlier portfolio-optimization research and adapted to the current equity-ranking platform.

For historical scenario portfolio returns \(r_p\), the central location is the median:

\[
m_p = \operatorname{median}(r_p)
\]

and dispersion is represented by mean absolute deviation around the median:

\[
MAD_p
=
\frac{1}{T}
\sum_{t=1}^{T}
|r_{p,t}-m_p|
\]

The optimization seeks an attractive robust central return while controlling MAD, turnover,
and feasibility constraints.

The current configuration uses:

```text
lookback_days: 252
minimum_observations: 126
mad_limit: 0.008
mad_violation_penalty: 10.0
turnover_penalty: 0.001
seed: 42
max_iterations: 80
population_size: 8
mutation_min: 0.5
mutation_max: 1.0
recombination: 0.7
polish: false
```

Differential Evolution searches the non-convex weight space. Candidate portfolios are mapped
back toward the feasible region so that concentration, sector, diversification, liquidity, and
other portfolio constraints remain part of the research comparison.

The Median-MAD method is treated as one portfolio-construction alternative, not as a separate
research universe or a replacement for the alpha model.

## Execution-aware backtesting

The methodology distinguishes target weights from realized portfolio performance.

A target portfolio becomes an executable simulation through:

```text
target weights
    -> orders
    -> trades
    -> shares
    -> positions
    -> transaction costs
    -> gross and net portfolio values
```

### Portfolio state

The execution engine tracks:

- position shares;
- security market values;
- portfolio cash;
- weight drift;
- rebalance trades;
- one-way turnover.

Positions evolve between rebalance dates as market prices change.

### Transaction costs

The advanced execution model contains:

```text
commission_bps: 0.5
half_spread_bps: 2.0
slippage_bps: 2.5
market_impact_coefficient: 0.10
```

The explicit decomposition prevents transaction cost from being represented by one unexplained
haircut.

The net return series is therefore derived after trading costs rather than obtained by
subtracting a constant annual fee from gross performance.

## Turnover

One-way turnover measures the amount of capital that must be reallocated between consecutive
target portfolios.

A generic representation is:

\[
TO_t
=
\frac{1}{2}
\sum_i
|w_{i,t}-w_{i,t^-}|
\]

where \(w_{i,t^-}\) represents the pre-trade portfolio weights.

Turnover is important both as a direct execution-cost driver and as a diagnostic of signal or
portfolio instability.

## Capacity

Capacity analysis evaluates how the same portfolio behaves at different capital levels.

The research includes representative scales:

```text
$100K
$1M
$10M
$100M
```

As capital increases, the fixed bps components remain conceptually similar while participation
and market-impact assumptions become more important.

Capacity results are interpreted as scenario analysis, not as a guarantee of executable
institutional scale.

## Portfolio performance metrics

The execution layer reports both gross and net performance.

The evaluation framework includes:

- cumulative return;
- CAGR;
- annualized volatility;
- Sharpe ratio;
- maximum drawdown;
- beta versus SPY;
- alpha versus SPY;
- turnover;
- transaction-cost drag;
- concentration and sector exposure.

No single performance statistic is treated as sufficient evidence.

For example, a higher CAGR accompanied by materially worse drawdown, capacity, turnover, or
instability is not automatically considered a superior research outcome.

## Benchmarking

SPY is used as the principal market benchmark.

The benchmark provides context for:

- market-relative return;
- beta;
- alpha;
- drawdown behavior;
- regime analysis.

The objective is not to claim that the portfolio is market neutral. The optimized portfolio
layer explicitly allows a controlled range of market beta.

## Robustness framework

Robustness analysis is applied after the frozen model and portfolio framework have been
defined.

The major dimensions are:

### Temporal robustness

Performance is examined across:

- calendar years;
- market regimes;
- rolling evaluation windows.

This tests whether results are concentrated in one isolated market environment.

### Transaction-cost sensitivity

Cost assumptions are stressed across multiple bps scenarios rather than relying on one chosen
cost level.

### Portfolio-parameter sensitivity

The platform varies:

- number of selected securities;
- security weight caps;
- related portfolio settings.

A stable conclusion should not disappear under small, plausible parameter perturbations.

### Rebalance-frequency sensitivity

Monthly and lower-frequency alternatives are compared to determine whether the result depends
on one exact trading cadence.

### Prediction-horizon robustness

Alternative forward horizons are evaluated, including shorter and longer horizons around the
primary 21-session target.

### Universe exclusions

Securities are removed from the frozen universe under controlled experiments to test whether a
small set of names dominates the conclusion.

This is different from a true expanded-universe test.

### Monthly bootstrap

Monthly portfolio returns are resampled to characterize uncertainty in:

- annualized performance;
- Sharpe;
- strategy ranking;
- pairwise strategy differences.

### Signal bootstrap

The final alpha signal is separately bootstrapped at the predictive level using monthly IC and
ranking-spread statistics.

### Sector stability

Signal quality is decomposed across sectors to test whether the effect is concentrated in one
industry group.

## Ablation methodology

Ablations answer "what happens if this component is removed?" They are diagnostic, not a second
model-selection round.

### Ensemble-component ablation

Each major model component is removed from the frozen ensemble and the signal is recomputed
under the corresponding restricted specification.

### Feature-family ablation

The frozen feature contract is split into interpretable families.

The principal feature-family experiments are:

- `no_fundamentals` — technical predictors only;
- `no_momentum` — complete feature set excluding the defined momentum/reversal family.

The full frozen model contains:

```text
91 total predictors
72 fundamental predictors
19 technical predictors
```

The no-fundamentals experiment therefore uses the 19 technical predictors.

The no-momentum experiment removes the defined momentum family while preserving the remaining
technical and fundamental features.

### Economic ablation

Predictive ablations are also propagated through the portfolio and execution stack where
appropriate.

This is necessary because a change in IC does not always translate proportionally into a
change in net portfolio performance.

### Governance rule

Even if an ablated specification happens to outperform the frozen full model over the observed
OOS sample, the platform does not retrospectively promote it to the official model.

Doing so would convert an ex-post robustness experiment into an unacknowledged optimization on
the test set.

## Data quality framework

The project uses both automated tests and materialized audit evidence.

The main validation families cover:

- universe integrity;
- market data;
- labels and rebalance dates;
- technical features;
- SEC fundamentals;
- modeling panel;
- walk-forward evaluation;
- dashboard/reporting contracts.

Checks are stored both as code-level tests and as tables or reports under `reports/`.

The dashboard's Data Quality view is downstream from those materialized controls rather than a
replacement for them.

## Interpretation of predictive evidence

Predictive evidence should be interpreted probabilistically.

A positive mean IC is meaningful only in the context of:

- its dispersion;
- the number of monthly observations;
- its confidence interval;
- yearly stability;
- sector stability;
- portfolio translation.

The frozen signal's bootstrap evidence is therefore reported with uncertainty rather than as a
binary "works / does not work" statement.

The same principle applies to top-minus-bottom spreads and portfolio performance.

## Interpretation of economic evidence

A high backtested CAGR is not treated as proof of deployable alpha.

Economic evidence is considered stronger when it survives:

- transaction costs;
- turnover;
- portfolio constraints;
- capacity stress;
- multiple portfolio constructors;
- temporal subperiods;
- parameter sensitivity;
- bootstrap uncertainty.

The final research conclusions therefore distinguish:

1. predictive ranking quality;
2. portfolio-construction effectiveness;
3. execution-aware realized simulation;
4. robustness of the observed evidence.

## Research-governance hierarchy

The project follows this decision hierarchy:

```text
define research contract
    -> build point-in-time data
    -> freeze features and labels
    -> define walk-forward protocol
    -> train and validate models
    -> freeze final alpha ensemble
    -> construct portfolios
    -> model execution
    -> evaluate robustness
    -> document limitations
```

Robustness analysis is downstream of the frozen research decision.

This hierarchy is central to avoiding retrospective overfitting.

## Known methodological limitations

### Static universe

The historical use of a current curated 50-name universe introduces survivorship and
membership bias.

### Limited cross-sectional breadth

Fifty securities are sufficient for the current research system and portfolio-construction
comparison, but they are not a substitute for a broad institutional equity universe.

### Market-data source

yfinance is appropriate for this research implementation but is not a licensed institutional
market-data feed.

### Fundamental normalization

SEC Companyfacts provides point-in-time public filing information, but issuer-specific XBRL
choices and concept mappings can introduce residual accounting heterogeneity.

### Model-selection uncertainty

Hyperparameter validation reduces in-sample overfitting but does not eliminate all research
degrees of freedom.

### Backtest realism

The execution model includes explicit cost components and market impact, but it does not
reconstruct historical order books, intraday queue position, partial fills, borrow constraints,
or live operational frictions.

### Capacity

Capacity scenarios are model-based estimates. They should not be interpreted as guaranteed
tradable capital.

### Statistical uncertainty

The OOS sample contains 77 monthly test periods. This is meaningful but still finite, and
confidence intervals can remain wide.

### No live-trading evidence

All performance results are historical research simulations. They are not live account
returns.

## Methodological acceptance criteria

A research release should only be considered valid if all of the following remain true:

- data used by a signal was available by the relevant observation date;
- feature windows do not extend into the future;
- test targets do not influence test predictions;
- cross-sectional dates are kept intact inside temporal folds;
- preprocessing is learned inside historical folds;
- the final ensemble remains frozen during robustness evaluation;
- portfolio constraints are satisfied;
- execution costs are explicitly represented;
- net performance is evaluated after costs;
- robustness experiments do not silently become model reselection;
- known limitations are reported alongside favorable results.

## Relationship to the architecture and reproducibility documents

This document describes **why** and **how** the quantitative research is designed.

For system implementation boundaries, artifact lineage, and layer separation, see:

- `docs/architecture.md`

For environment setup, data prerequisites, executable scripts, portable dashboard
reproduction, and full pipeline reconstruction, see:

- `docs/reproducibility.md`

Detailed factor and dataset definitions remain in:

- `docs/FEATURE_DICTIONARY.md`
- `docs/DATA_DICTIONARY.md`
- `docs/LABELS_AND_REBALANCING.md`
- `docs/TECHNICAL_FEATURES.md`
- `docs/TECHNICAL_FEATURE_PROCESSING.md`
- `docs/UNIVERSE_METHODOLOGY.md`

The remaining high-level documentation layer is `docs/research_results.md`, which should report
the frozen predictive, portfolio, risk, execution, robustness, and data-quality evidence
without redefining the methodology.
