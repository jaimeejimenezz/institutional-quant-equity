from __future__ import annotations

import pandas as pd
import streamlit as st
from components.robustness_charts import (
    feature_ablation_figure,
    horizon_sensitivity_figure,
    rebalance_sensitivity_figure,
    regime_heatmap_figure,
    rolling_window_figure,
    signal_bootstrap_figure,
    strategy_return_bootstrap_figure,
    universe_exclusion_figure,
)
from components.shell import render_page_header
from data_access import load_dashboard_source

from quant_equity.reporting.dashboard_robustness import (
    construction_ablation_table,
    coverage_table,
    feature_ablation_table,
    humanize_identifier,
    inventory_table,
    robustness_headline,
    strategy_bootstrap_table,
    strategy_display_label,
    universe_exclusion_table,
)


def _percent(value: object, digits: int = 1, *, signed: bool = False) -> str:
    if pd.isna(value):
        return "N/A"
    spec = f"+.{digits}%" if signed else f".{digits}%"
    return format(float(value), spec)


def _ratio(value: object, digits: int = 2, *, signed: bool = False) -> str:
    if pd.isna(value):
        return "N/A"
    spec = f"+.{digits}f" if signed else f".{digits}f"
    return format(float(value), spec)


def _format_percent_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    digits: int = 1,
    signed: bool = False,
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = result[column].map(
                lambda value: _percent(value, digits, signed=signed)
            )
    return result


def _format_ratio_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    digits: int = 2,
    signed: bool = False,
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = result[column].map(
                lambda value: _ratio(value, digits, signed=signed)
            )
    return result


render_page_header(
    "Robustness",
    (
        "Stress tests, statistical uncertainty, specification sensitivity, "
        "ablations, universe dependence, and market-regime evidence."
    ),
)

st.caption(
    
        "Robustness diagnostics do not select a new model retrospectively. "
        "The frozen full specification remains the research reference."
    
)

inventory = load_dashboard_source("robustness_inventory")
coverage = load_dashboard_source("robustness_coverage")
bootstrap_strategy = load_dashboard_source("bootstrap_strategy")
signal_bootstrap = load_dashboard_source("signal_bootstrap")
feature_ablation = load_dashboard_source("feature_family_ablation")
economic_ablation = load_dashboard_source("economic_ablation")
construction_ablation = load_dashboard_source("construction_ablation")
horizon = load_dashboard_source("horizon_sensitivity")
rebalance = load_dashboard_source("rebalance_sensitivity")
rolling = load_dashboard_source("rolling_window_sensitivity")
universe = load_dashboard_source("universe_exclusions")
regime = load_dashboard_source("regime_performance")

headline = robustness_headline(inventory, coverage, signal_bootstrap)
signal = signal_bootstrap.iloc[0]

headline_row_one = st.columns(2)
headline_row_one[0].metric(
    "Validation suites",
    f"{headline['passed_suites']}/{headline['total_suites']} PASS",
)
headline_row_one[1].metric(
    "Coverage",
    f"{headline['complete_dimensions']}/{len(coverage)} complete",
)

headline_row_two = st.columns(2)
headline_row_two[0].metric(
    "Documented limitations",
    str(headline["deferred_dimensions"]),
)
headline_row_two[1].metric(
    "Bootstrap replications",
    f"{headline['bootstrap_replications']:,}",
)

st.caption(f"Frozen-signal evaluation window: {headline['oos_months']} OOS months.")

st.subheader("Validation coverage")
st.caption(
    
        "Every audited validation suite passes. Expanded-universe evidence is "
        "explicitly deferred because a valid test requires new securities and a "
        "rebuilt point-in-time upstream pipeline."
    
)

st.markdown("**Coverage map**")
st.dataframe(
    coverage_table(coverage),
    width="stretch",
    hide_index=True,
    height=455,
)

deferred = coverage.loc[
    coverage["status"].astype(str).str.upper().eq("DEFERRED_LIMITATION")
]
if not deferred.empty:
    deferred_row = deferred.iloc[0]
    st.info(
        
            f"Documented limitation — {deferred_row['dimension']}: "
            f"{deferred_row['note']} This is not a failed robustness suite."
        
    )

st.markdown("**Validation-suite inventory**")
st.dataframe(
    inventory_table(inventory),
    width="stretch",
    hide_index=True,
    height=500,
)

st.subheader("Statistical evidence")
st.caption(
    
        "Confidence intervals quantify uncertainty around the frozen alpha signal "
        "and realized portfolio outcomes. They are evidence, not tuning criteria."
    
)

signal_row_one = st.columns(2)
signal_row_one[0].metric("Mean IC", _ratio(signal["observed_mean_ic"], 3))
signal_row_one[1].metric(
    "Mean IC 95% CI",
    (
        f"{_ratio(signal['mean_ic_ci_lower'], 3)} to "
        f"{_ratio(signal['mean_ic_ci_upper'], 3)}"
    ),
)

signal_row_two = st.columns(2)
signal_row_two[0].metric(
    "Probability mean IC > 0",
    _percent(signal["probability_mean_ic_positive"]),
)
signal_row_two[1].metric(
    "Mean top-bottom spread",
    _percent(signal["observed_mean_top_bottom_spread"], 2),
)

st.caption(
    
        f"P(mean spread > 0): "
        f"{_percent(signal['probability_mean_spread_positive'])} · "
        f"P(IC and spread > 0): "
        f"{_percent(signal['probability_both_ic_and_spread_positive'])}"
    
)

stat_left, stat_right = st.columns([0.9, 1.1])
with stat_left:
    st.markdown("**Frozen signal bootstrap**")
    st.plotly_chart(signal_bootstrap_figure(signal_bootstrap), width="stretch")

with stat_right:
    st.markdown("**Portfolio return bootstrap**")
    st.plotly_chart(
        strategy_return_bootstrap_figure(bootstrap_strategy),
        width="stretch",
    )

bootstrap_display = strategy_bootstrap_table(bootstrap_strategy)
bootstrap_display = _format_percent_columns(
    bootstrap_display,
    ("Return", "P(excess > 0)"),
)
bootstrap_display = _format_ratio_columns(
    bootstrap_display,
    ("Sharpe",),
)
st.dataframe(bootstrap_display, width="stretch", hide_index=True)

st.subheader("Parameter sensitivity")
st.caption(
    
        "The same frozen research specification is evaluated across forward "
        "horizons, rebalance frequencies, and rolling evaluation windows."
    
)

parameter_left, parameter_right = st.columns(2)
with parameter_left:
    st.markdown("**Prediction horizon**")
    st.plotly_chart(horizon_sensitivity_figure(horizon), width="stretch")

with parameter_right:
    st.markdown("**Quarterly versus monthly rebalance**")
    st.plotly_chart(rebalance_sensitivity_figure(rebalance), width="stretch")

st.markdown("**Rolling evaluation windows**")
st.plotly_chart(rolling_window_figure(rolling), width="stretch")

st.subheader("Ablation diagnostics")
st.caption(
    
        "Ablations measure dependence on feature families and construction "
        "controls. Better observed historical results for an ablation do not "
        "authorize replacing or retuning the frozen full model on the same OOS period."
    
)

st.markdown("**Feature-family predictive evidence**")
st.plotly_chart(feature_ablation_figure(feature_ablation), width="stretch")

st.markdown("**Predictive and economic comparison**")
ablation_display = feature_ablation_table(feature_ablation, economic_ablation)
ablation_display = _format_ratio_columns(
    ablation_display,
    ("Mean IC", "IC IR", "Sharpe"),
)
ablation_display = _format_percent_columns(
    ablation_display,
    ("T-B spread", "Top-Q hit", "CAGR"),
)
ablation_display = _format_percent_columns(
    ablation_display,
    ("CAGR vs full",),
    signed=True,
)
st.dataframe(ablation_display, width="stretch", hide_index=True)

st.markdown("**Portfolio-construction controls**")
construction_display = construction_ablation_table(construction_ablation)
construction_display = _format_percent_columns(
    construction_display,
    ("CAGR", "Turnover", "Max sector"),
)
construction_display = _format_ratio_columns(
    construction_display,
    ("Sharpe",),
)
st.dataframe(construction_display, width="stretch", hide_index=True)

st.subheader("Frozen-universe dependence")
st.caption(
    
        "These tests remove groups from the existing frozen universe. They are not "
        "an expanded-universe experiment and should not be presented as one."
    
)

st.plotly_chart(universe_exclusion_figure(universe), width="stretch")
universe_display = universe_exclusion_table(universe)
universe_display = _format_percent_columns(universe_display, ("CAGR",))
universe_display = _format_ratio_columns(universe_display, ("Sharpe",))
universe_display = _format_percent_columns(
    universe_display,
    ("CAGR vs full",),
    signed=True,
)
st.dataframe(universe_display, width="stretch", hide_index=True)

st.subheader("Market-regime evidence")
st.caption(
    
        "Stored Sharpe ratios are shown by regime. An asterisk marks any regime "
        "with a short-sample warning. The heatmap color scale is clipped to ±3 "
        "for legibility; the cell labels retain the stored Sharpe values."
    
)
st.plotly_chart(regime_heatmap_figure(regime), width="stretch")

regime_display = regime.loc[
    :,
    [
        "regime",
        "strategy_name",
        "trading_days",
        "cagr",
        "sharpe_ratio",
        "maximum_drawdown",
        "short_sample_warning",
    ],
].copy()
regime_display["regime"] = regime_display["regime"].map(humanize_identifier)
regime_display["strategy_name"] = regime_display["strategy_name"].map(
    strategy_display_label
)
regime_display = regime_display.rename(
    columns={
        "regime": "Regime",
        "strategy_name": "Strategy",
        "trading_days": "Days",
        "cagr": "CAGR",
        "sharpe_ratio": "Sharpe",
        "maximum_drawdown": "Max DD",
        "short_sample_warning": "Short sample",
    }
)
regime_display = _format_percent_columns(
    regime_display,
    ("CAGR", "Max DD"),
)
regime_display = _format_ratio_columns(regime_display, ("Sharpe",))
st.dataframe(
    regime_display,
    width="stretch",
    hide_index=True,
    height=390,
)

st.caption(
    
        "Research interpretation rule: robustness analysis measures stability, "
        "dependence, and uncertainty. It is not a retrospective search for the "
        "best-looking historical specification."
    
)
