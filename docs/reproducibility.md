# Reproducibility Guide

## Scope

This document defines how to reproduce the Institutional Quant Equity Research Platform at
two different levels:

1. **Dashboard reproduction** — run the portfolio-ready Streamlit application from a clean
   clone using the curated processed artifacts committed with the repository.
2. **Full research reconstruction** — rebuild the quantitative research chain from external
   market and SEC data through features, labels, models, risk, portfolio construction,
   execution, robustness, and reporting.

These two targets are intentionally different. A portable dashboard release is expected to be
fast and self-contained. A full research reconstruction is data-intensive, computationally
heavier, and requires external data access.

## Supported runtime

The project targets Python 3.11 or later and is currently validated on Python 3.11.

The current validated scientific environment includes:

- SciPy 1.17.1
- CVXPY 1.9.2
- LightGBM 4.7.0
- Streamlit 1.61.1

The project metadata declares SciPy, CVXPY, and LightGBM directly because they are imported by
the quantitative implementation rather than treated as incidental transitive dependencies.

The active CVXPY environment used during the reproducibility audit exposed the following
solver interfaces:

- CLARABEL
- SCS
- SCIPY
- HIGHS
- OSQP

The exact solver selected by a CVXPY problem remains an implementation/runtime concern; the
reproducibility contract requires at least one supported solver to be available.

## Installation from a clean clone

### Windows PowerShell

```powershell
git clone https://github.com/jaimeejimenezz/institutional-quant-equity.git
cd institutional-quant-equity

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### POSIX shells

```bash
git clone https://github.com/jaimeejimenezz/institutional-quant-equity.git
cd institutional-quant-equity

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` contains only:

```text
.
```

This is intentional. Runtime dependencies are defined in `pyproject.toml`, so
`pip install -r requirements.txt` installs the project itself and resolves its declared
dependencies from the canonical package metadata.

Development tooling can be installed with:

```bash
python -m pip install -e ".[dev]"
```

## Environment verification

After installation, run:

```bash
python --version
python -m pip check
python -c "import quant_equity"
python -c "import scipy, cvxpy, lightgbm; print(scipy.__version__, cvxpy.__version__, lightgbm.__version__)"
python -c "import cvxpy as cp; print(cp.installed_solvers())"
```

The package import must succeed, `pip check` must report no broken requirements, and CVXPY
must expose at least one installed solver.

## Automated quality checks

The repository uses Ruff and pytest as the primary code-quality gates.

Run:

```bash
ruff check .
pytest
```

For a repository change that will be committed, also run:

```bash
git diff --check
```

The documentation release that introduced this guide was built on a test baseline of 398
passing tests before the runtime-dependency tests in this patch were added. The exact total may
increase as the repository evolves; the relevant reproducibility condition is that the complete
suite passes.

Warnings from NumPy correlation calculations on constant inputs and Elastic Net convergence in
small synthetic test fixtures are known diagnostics in the current test suite. They are not
test failures.

## Central configuration

Research configuration is stored in:

```text
config/project.yaml
```

Important reproducibility parameters include:

- global random seed: `42`;
- research start date: `2014-01-01`;
- market: US equities;
- base currency: USD;
- expected universe size: 50 securities;
- monthly rebalance frequency;
- primary prediction horizon: 21 sessions;
- long-only, fully invested portfolios;
- maximum security weight: 5%;
- maximum sector weight: 25%;
- SEC point-in-time availability lag: 1 day;
- 252-session risk and covariance lookbacks;
- Ledoit-Wolf covariance estimation;
- Median-MAD Differential Evolution seed: `42`.

Configuration should be treated as part of the research specification. Changing these values
creates a different experiment and should not be presented as reproduction of the frozen
research release.

## Randomness and determinism

The project uses a global seed of `42`. Stochastic components should derive their randomness
from explicit seeds rather than uncontrolled global state.

The Median-MAD Differential Evolution configuration also fixes seed `42`.

Deterministic seeding makes repeated runs materially more reproducible, but bit-for-bit
identity across operating systems, BLAS implementations, solver versions, and CPU
architectures is not guaranteed. Numerical optimization and multithreaded machine-learning
libraries can exhibit small platform-dependent differences.

For portfolio and research validation, the expected requirement is semantic reproducibility:

- the same temporal folds;
- the same feature contract;
- the same constraints;
- the same model-selection procedure;
- numerically equivalent outputs within the tolerances enforced by tests and audit scripts.

## Storage model

The repository uses four data layers:

```text
data/
├── raw/
├── interim/
├── processed/
└── reference/
```

### Raw

Source-preserving inputs such as downloaded market data and SEC Companyfacts payloads.
These are not intended to be manually edited.

### Interim

Normalized, canonicalized, or reconstructed datasets used between raw ingestion and final
feature production.

### Processed

Research-ready Parquet artifacts such as labels, features, modeling panels, predictions, risk
estimates, portfolios, positions, trades, and backtests.

### Reference

Stable research metadata, including the universe definition and XBRL concept mapping.

## Dashboard reproduction

The dashboard is intentionally decoupled from model training. A clean clone does not need to
rebuild the SEC pipeline, refit models, or rerun portfolio optimization just to inspect the
research application.

After installation:

```bash
streamlit run app/main.py
```

The application reads persisted research artifacts through the reporting/data-access layer.

### Versioned dashboard artifacts

The portable dashboard release includes the following processed Parquet files:

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

These files are intentionally versioned even though most generated files under
`data/processed/` remain ignored. They form the minimal portable reporting contract.

### Dashboard health check

After launching Streamlit, a local smoke test can be performed against:

```text
/_stcore/health
```

A healthy application should return HTTP 200 and the Streamlit health response.

## Full research reconstruction

Full reconstruction is a different workflow from dashboard reproduction. It requires external
data access and rebuilds the materialized research state.

The repository is organized as explicit stage scripts rather than one opaque monolithic
command. This is intentional: each material stage produces auditable outputs and can be
validated before the next stage is run.

The commands below reflect the current script inventory. They should be executed from the
repository root with the virtual environment active.

## External-data prerequisites

### Market data

The market-data pipeline currently uses yfinance through the project's provider abstraction.
An internet connection is required when source data is not already cached.

### SEC EDGAR

SEC Companyfacts downloads require an identifying HTTP user agent.

Set the environment variable before running SEC downloads.

PowerShell:

```powershell
$env:SEC_USER_AGENT = "Your Name your.email@example.com"
```

POSIX:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
```

Use a real identifying contact string suitable for SEC requests. Do not commit personal
credentials or private environment values to the repository.

## Research reconstruction sequence

### 1. Setup and universe

```bash
python scripts/check_setup.py
python scripts/check_universe.py
```

These checks validate the project environment and the configured universe before material data
pipelines run.

### 2. Market data

```bash
python scripts/check_market_provider.py
python scripts/download_market_data.py
python scripts/validate_market_data.py
python scripts/download_spy_benchmark.py
```

Principal downstream market artifact:

```text
data/processed/market_daily.parquet
```

### 3. Rebalance calendar and labels

```bash
python scripts/build_monthly_labels.py
python scripts/validate_monthly_labels.py
```

Principal outputs include:

```text
data/processed/rebalance_calendar.parquet
data/processed/labels_monthly.parquet
```

### 4. Technical features and factor research

```bash
python scripts/build_raw_technical_features.py
python scripts/build_technical_features.py
python scripts/validate_technical_features.py
python scripts/research_technical_factors.py
python scripts/research_technical_factor_diagnostics.py
python scripts/finalize_technical_feature_selection.py
```

The processed technical feature matrix is:

```text
data/processed/features_technical_monthly.parquet
```

### 5. SEC fundamentals

With `SEC_USER_AGENT` configured:

```bash
python scripts/download_sec_companyfacts.py
python scripts/normalize_sec_companyfacts.py
python scripts/build_canonical_fundamental_events.py
python scripts/build_point_in_time_fundamentals.py
python scripts/build_quarterly_ttm_fundamentals.py
python scripts/audit_fundamental_factor_inputs.py
python scripts/build_fundamental_base.py
python scripts/build_raw_fundamental_factors.py
python scripts/build_fundamental_growth_factors.py
python scripts/build_processed_fundamental_features.py
python scripts/research_fundamental_factor_diagnostics.py
```

The processed fundamental feature matrix is:

```text
data/processed/features_fundamental_monthly.parquet
```

### 6. Modeling panel and leakage audit

```bash
python scripts/build_modeling_panel.py
python scripts/audit_modeling_panel_leakage.py
python scripts/finalize_modeling_panel.py
python scripts/document_modeling_panel.py
```

Principal output:

```text
data/processed/modeling_panel.parquet
```

The modeling-panel stage should not be considered complete solely because the Parquet file
exists. Leakage, readiness, coverage, and point-in-time checks must also pass.

### 7. Walk-forward validation

```bash
python scripts/build_walk_forward_folds.py
python scripts/audit_walk_forward_preprocessing.py
python scripts/audit_walk_forward_readiness.py
python scripts/finalize_walk_forward_validation.py
```

Principal materialized fold output:

```text
data/processed/walk_forward_folds.parquet
```

The frozen research release contains 77 out-of-sample folds.

### 8. Predictive models

```bash
python scripts/run_model_baselines.py
python scripts/train_linear_models.py
python scripts/evaluate_linear_models.py
python scripts/train_regularized_linear_models.py
python scripts/train_lightgbm_regression.py
python scripts/train_lightgbm_ranking.py
python scripts/analyze_model_comparison.py
```

Representative outputs include:

```text
data/processed/predictions_oos_all_models.parquet
data/processed/predictions_oos_regularized_linear.parquet
data/processed/predictions_oos_lightgbm.parquet
data/processed/predictions_oos_lightgbm_ranker.parquet
```

All predictions used for final research claims must remain out of sample.

### 9. Frozen alpha signal

```bash
python scripts/build_final_alpha_signal.py
python scripts/audit_frozen_predictive_evaluation_semantics.py
python scripts/audit_predictive_evaluation_contract.py
python scripts/run_final_signal_statistical_robustness.py
```

Principal output:

```text
data/processed/final_alpha_signal.parquet
```

The frozen ensemble is a research-governance boundary. Diagnostic ablations must not be used
to retrospectively reselect the full model on the same out-of-sample period.

### 10. Risk model

```bash
python scripts/build_risk_estimates.py
python scripts/build_covariance_matrices.py
python scripts/evaluate_reference_portfolio_risk.py
```

Principal outputs:

```text
data/processed/risk_estimates.parquet
data/processed/covariance_matrices.parquet
```

### 11. Portfolio construction

```bash
python scripts/build_baseline_portfolios.py
python scripts/build_optimized_portfolios.py
python scripts/build_cvar_portfolios.py
python scripts/build_median_mad_portfolios.py
```

Principal combined target-weight artifact:

```text
data/processed/target_weights_all_methods.parquet
```

The current research compares:

- `top_n_equal_weight`;
- `score_weighted`;
- `alpha_risk_turnover`;
- `cvar`;
- `median_mad_de`.

### 12. Execution-aware backtest

```bash
python scripts/run_portfolio_execution_backtest.py
python scripts/run_capacity_analysis.py
```

Principal outputs include:

```text
data/processed/backtest_all_methods_gross_daily.parquet
data/processed/backtest_all_methods_net_daily.parquet
data/processed/positions_all_methods_net.parquet
data/processed/trades_all_methods_net.parquet
```

Execution costs include commission, spread, slippage, and an approximate market-impact
component.

### 13. Robustness and sensitivity

The robustness suite is deliberately executed after the frozen research signal and portfolio
contracts exist.

Representative commands are:

```bash
python scripts/run_temporal_robustness.py
python scripts/run_transaction_cost_sensitivity.py
python scripts/run_portfolio_parameter_sensitivity.py
python scripts/run_rebalance_frequency_sensitivity.py
python scripts/run_rolling_window_robustness.py
python scripts/run_prediction_horizon_robustness.py
python scripts/run_universe_exclusion_robustness.py
python scripts/run_monthly_bootstrap.py
python scripts/run_ensemble_component_ablation.py
python scripts/run_portfolio_construction_ablation.py
python scripts/run_feature_family_ablation_contract.py
python scripts/run_feature_family_ablation_retraining.py
python scripts/run_feature_family_economic_ablation.py
python scripts/run_feature_family_paired_bootstrap.py
python scripts/finalize_robustness_evaluation.py
```

The current robustness release records the genuine expanded-universe experiment as a deferred
limitation rather than manufacturing it from the fixed 50-name universe.

## Expected artifact lineage

A successful full reconstruction should reproduce the following high-level dependency chain:

```text
universe
  -> market data
  -> rebalance calendar and labels
  -> technical features
  -> point-in-time SEC fundamentals
  -> fundamental features
  -> modeling panel
  -> walk-forward folds
  -> out-of-sample model predictions
  -> frozen alpha signal
  -> security risk and covariance
  -> target portfolios
  -> positions, trades, and execution costs
  -> gross and net backtests
  -> robustness evidence
  -> dashboard reporting artifacts
```

## Integrity and audit evidence

The project persists both executable tests and research-run evidence.

Important audit locations include:

```text
reports/data_quality/
reports/validation/
reports/models/
reports/risk/
reports/portfolio/
reports/execution/
reports/robustness/
reports/dashboard/
reports/tables/
```

These outputs serve different purposes:

- **tests** protect implementation and contract behavior;
- **audit scripts** assert research invariants on materialized artifacts;
- **CSV tables** preserve machine-readable diagnostics;
- **Markdown reports** summarize the evidence for human inspection;
- **Parquet artifacts** preserve the stage outputs consumed downstream.

A reproducible run should not be accepted solely because the final dashboard launches.
Material validation and leakage checks must also pass.

## Current release temporal boundaries

The current research release uses several terminal dates for different purposes:

| Meaning | Date |
|---|---|
| Last frozen alpha signal date | 2026-05-29 |
| Last evaluated portfolio day | 2026-06-30 |
| Last stored raw SPY market date | 2026-07-27 |

These dates are expected to differ. The signal date, future holding/evaluation horizon, and raw
benchmark coverage represent different stages of the pipeline.

## Portable release versus archival research data

Only the curated dashboard artifacts are intentionally versioned from `data/processed/`.

A clean clone therefore reproduces the **reporting release**, not the complete historical raw
data archive.

Full reconstruction may regenerate additional artifacts such as:

- full market history;
- normalized SEC data;
- technical and fundamental feature matrices;
- covariance matrices;
- model predictions;
- intermediate portfolio and robustness outputs.

Those generated research files do not all need to be committed to Git for the project to be
reproducible.

## Reproducibility acceptance checklist

A release is considered reproducible when the following conditions hold:

- the package installs from `requirements.txt`;
- the direct quantitative dependencies are declared in `pyproject.toml`;
- `python -m pip check` reports no broken requirements;
- `import quant_equity` succeeds;
- Ruff passes;
- the full pytest suite passes;
- configuration contains no duplicate YAML mapping keys;
- dashboard source contracts pass;
- point-in-time and leakage audits pass;
- walk-forward predictions remain out of sample;
- portfolio constraints pass;
- execution accounting and cost checks pass;
- robustness coverage is explicitly reported, including deferred limitations;
- the dashboard launches from the versioned portable artifacts.

## Related documentation

- `docs/architecture.md` — system layers, data flow, contracts, and reporting separation.
- `docs/UNIVERSE_METHODOLOGY.md` — universe construction and survivorship limitation.
- `docs/LABELS_AND_REBALANCING.md` — target and rebalance semantics.
- `docs/FEATURE_DICTIONARY.md` — feature definitions.
- `docs/DATA_DICTIONARY.md` — modeling-panel and artifact semantics.

The next documentation layer is `docs/methodology.md`, which explains the research design and
interpretation in more depth than this operational guide.
