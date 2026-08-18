from __future__ import annotations

import math

import pandas as pd
import streamlit as st
from components.research_charts import (
    ensemble_candidate_figure,
    ensemble_correlation_figure,
    feature_importance_figure,
    model_comparison_figure,
    monthly_research_figure,
    stability_heatmap_figure,
)
from components.shell import render_page_header
from data_access import load_dashboard_source

from quant_equity.reporting.dashboard_research import (
    ensemble_candidate_table,
    ensemble_component_monthly,
    ensemble_correlation_matrix,
    feature_family_summary,
    feature_importance_table,
    model_comparison_table,
    sector_stability_matrix,
    yearly_stability_matrix,
)


def _format_decimal(value: object, *, digits: int = 2) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "N/A"
    return f"{float(number):.{digits}f}"


def _format_percent(value: object, *, digits: int = 1) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "N/A"
    return f"{float(number):.{digits}%}"


model_summary = model_comparison_table(load_dashboard_source("model_summary"))
model_monthly = ensemble_component_monthly(load_dashboard_source("model_monthly"))
model_yearly = yearly_stability_matrix(load_dashboard_source("model_yearly"))
model_sector = sector_stability_matrix(load_dashboard_source("model_sector"))
feature_importance = load_dashboard_source("feature_importance")
top_features = feature_importance_table(feature_importance, top_n=15)
feature_families = feature_family_summary(feature_importance)
ensemble_summary = ensemble_candidate_table(load_dashboard_source("ensemble_summary"))
ensemble_correlations = ensemble_correlation_matrix(
    load_dashboard_source("ensemble_correlations")
)

render_page_header(
    "Models & Factors",
    "Out-of-sample predictive evidence, temporal stability, factor usage, and ensemble research.",
    eyebrow="Institutional Quant Equity Research Platform",
)

st.caption(
    "Research diagnostics only. The frozen alpha signal remains the operational research signal; "
    "this page does not retrain, retune, or replace it using observed out-of-sample results."
)

core_rows = ensemble_summary[
    ensemble_summary["model_name"].astype(str) == "core_percentile_ensemble"
]
core = core_rows.iloc[0] if not core_rows.empty else ensemble_summary.iloc[0]

kpis = st.columns(5, gap="small")
kpis[0].metric("Core mean IC", _format_decimal(core["mean_ic"], digits=3), border=True)
kpis[1].metric(
    "Core IC IR",
    _format_decimal(core["annualized_ic_ir"], digits=2),
    border=True,
)
kpis[2].metric(
    "Positive IC",
    _format_percent(core["positive_ic_ratio"], digits=1),
    border=True,
)
kpis[3].metric(
    "T-B spread",
    _format_percent(core["mean_top_bottom_spread"], digits=2),
    border=True,
)
kpis[4].metric("LGBM features", f"{len(feature_importance)}", border=True)

with st.container(border=True):
    st.subheader("Model comparison")
    st.caption(
        "Canonical out-of-sample comparison across simple baselines, linear models, and LightGBM. "
        "Highlighted chart colors identify the three frozen ensemble components."
    )
    st.plotly_chart(
        model_comparison_figure(model_summary),
        width="stretch",
        config={"displayModeBar": False},
    )

    comparison_table = model_summary.loc[
        :,
        [
            "model_label",
            "months",
            "mean_ic",
            "annualized_ic_ir",
            "positive_ic_ratio",
            "mean_top_bottom_spread",
            "mean_top_quintile_precision",
        ],
    ].rename(
        columns={
            "model_label": "Model",
            "months": "OOS",
            "mean_ic": "Mean IC",
            "annualized_ic_ir": "IC IR",
            "positive_ic_ratio": "Positive IC",
            "mean_top_bottom_spread": "Spread",
            "mean_top_quintile_precision": "Top-Q hit",
        }
    )
    comparison_table["Mean IC"] = comparison_table["Mean IC"].map(_format_decimal)
    comparison_table["IC IR"] = comparison_table["IC IR"].map(_format_decimal)
    comparison_table["Positive IC"] = comparison_table["Positive IC"].map(_format_percent)
    comparison_table["Spread"] = comparison_table["Spread"].map(
        lambda value: _format_percent(value, digits=2)
    )
    comparison_table["Top-Q hit"] = comparison_table["Top-Q hit"].map(
        _format_percent
    )
    st.dataframe(
        comparison_table,
        width="stretch",
        hide_index=True,
        height=315,
        column_config={
            "Model": st.column_config.TextColumn(width="medium"),
            "OOS": st.column_config.TextColumn(width="small"),
            "Mean IC": st.column_config.TextColumn(width="small"),
            "IC IR": st.column_config.TextColumn(width="small"),
            "Positive IC": st.column_config.TextColumn(width="small"),
            "Spread": st.column_config.TextColumn(width="small"),
            "Top-Q hit": st.column_config.TextColumn(width="small"),
        },
    )
    st.caption(
        "Constant-model ranking metrics are undefined by construction. Elastic Net has 74 valid "
        "IC months; the other non-constant models have 77 in the canonical summary."
    )

st.subheader("Monthly OOS model diagnostics")
st.caption(
    "Monthly evidence for the three components used by the frozen ensemble: Technical Composite, "
    "Elastic Net, and LightGBM Ranker."
)
monthly_left, monthly_right = st.columns(2, gap="medium")
with monthly_left:
    with st.container(border=True):
        st.markdown("**Information coefficient history**")
        st.plotly_chart(
            monthly_research_figure(
                model_monthly,
                metric="ic",
                yaxis_title="Spearman IC",
                tickformat=".2f",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
with monthly_right:
    with st.container(border=True):
        st.markdown("**Top-bottom spread history**")
        st.plotly_chart(
            monthly_research_figure(
                model_monthly,
                metric="top_bottom_spread",
                yaxis_title="21-session excess-return spread",
                tickformat=".1%",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )

st.subheader("Stability diagnostics")
st.caption(
    "The canonical yearly and sector artifacts do not contain LightGBM Ranker rows. The dashboard "
    "shows only stored observations and does not infer missing ranker stability metrics."
)
with st.container(border=True):
    st.markdown("**Yearly mean IC**")
    st.plotly_chart(
        stability_heatmap_figure(model_yearly, height=300),
        width="stretch",
        config={"displayModeBar": False},
    )

with st.container(border=True):
    st.markdown("**Sector mean IC**")
    st.caption(
        f"Stored sector-stability artifact contains {len(model_sector.columns)} sectors; "
        "no values are inferred for sectors absent from that artifact."
    )
    st.plotly_chart(
        stability_heatmap_figure(model_sector, height=350),
        width="stretch",
        config={"displayModeBar": False},
    )

st.subheader("LightGBM Ranker feature importance")
feature_top_share = (
    float(top_features.iloc[0]["mean_gain_share"])
    if not top_features.empty
    else math.nan
)
feature_top_five_share = float(top_features.head(5)["mean_gain_share"].sum())
family_lookup = feature_families.set_index("family")["gain_share"].to_dict()
feature_kpis = st.columns(4, gap="small")
feature_kpis[0].metric(
    "Top feature share",
    _format_percent(feature_top_share, digits=1),
    border=True,
)
feature_kpis[1].metric(
    "Top 5 concentration",
    _format_percent(feature_top_five_share, digits=1),
    border=True,
)
feature_kpis[2].metric(
    "Technical gain share",
    _format_percent(family_lookup.get("Technical", math.nan), digits=1),
    border=True,
)
feature_kpis[3].metric(
    "Fundamental gain",
    _format_percent(family_lookup.get("Fundamental", math.nan), digits=1),
    border=True,
)

with st.container(border=True):
    st.caption("Top 15 predictors by mean LightGBM gain share across walk-forward folds.")
    st.plotly_chart(
        feature_importance_figure(top_features),
        width="stretch",
        config={"displayModeBar": False},
    )

with st.container(border=True):
    st.markdown("**Top feature detail**")
    feature_table = top_features.loc[
        :,
        [
            "feature_label",
            "family",
            "mean_gain_share",
            "folds_used",
        ],
    ].rename(
        columns={
            "feature_label": "Feature",
            "family": "Family",
            "mean_gain_share": "Mean gain",
            "folds_used": "Folds",
        }
    )
    feature_table["Mean gain"] = feature_table["Mean gain"].map(
        lambda value: _format_percent(value, digits=2)
    )
    st.dataframe(
        feature_table,
        width="stretch",
        hide_index=True,
        height=360,
        column_config={
            "Feature": st.column_config.TextColumn(width="large"),
            "Family": st.column_config.TextColumn(width="medium"),
            "Mean gain": st.column_config.TextColumn(width="small"),
            "Folds": st.column_config.NumberColumn(width="small", format="%d"),
        },
    )

st.subheader("Ensemble research")
st.caption(
    "Candidate comparison and cross-sectional signal correlations are presented as diagnostics, "
    "not as a retrospective model-selection control."
)
ensemble_left, ensemble_right = st.columns([1.1, 0.9], gap="medium")
with ensemble_left:
    with st.container(border=True):
        st.markdown("**Candidate evidence**")
        st.plotly_chart(
            ensemble_candidate_figure(ensemble_summary),
            width="stretch",
            config={"displayModeBar": False},
        )
with ensemble_right:
    with st.container(border=True):
        st.markdown("**Component signal correlation**")
        st.plotly_chart(
            ensemble_correlation_figure(ensemble_correlations),
            width="stretch",
            config={"displayModeBar": False},
        )

with st.container(border=True):
    ensemble_table = ensemble_summary.loc[
        :,
        [
            "model_label",
            "mean_ic",
            "annualized_ic_ir",
            "positive_ic_ratio",
            "mean_top_bottom_spread",
            "mean_top_quintile_precision",
            "mean_top_quintile_turnover",
        ],
    ].rename(
        columns={
            "model_label": "Candidate",
            "mean_ic": "Mean IC",
            "annualized_ic_ir": "IC IR",
            "positive_ic_ratio": "Positive IC",
            "mean_top_bottom_spread": "Spread",
            "mean_top_quintile_precision": "Top-Q hit",
            "mean_top_quintile_turnover": "Turnover",
        }
    )
    ensemble_table["Mean IC"] = ensemble_table["Mean IC"].map(_format_decimal)
    ensemble_table["IC IR"] = ensemble_table["IC IR"].map(_format_decimal)
    ensemble_table["Positive IC"] = ensemble_table["Positive IC"].map(_format_percent)
    ensemble_table["Spread"] = ensemble_table["Spread"].map(
        lambda value: _format_percent(value, digits=2)
    )
    ensemble_table["Top-Q hit"] = ensemble_table["Top-Q hit"].map(
        _format_percent
    )
    ensemble_table["Turnover"] = ensemble_table["Turnover"].map(
        _format_percent
    )
    st.dataframe(
        ensemble_table,
        width="stretch",
        hide_index=True,
        height=155,
        column_config={
            "Candidate": st.column_config.TextColumn(width="large"),
            "Mean IC": st.column_config.TextColumn(width="small"),
            "IC IR": st.column_config.TextColumn(width="small"),
            "Positive IC": st.column_config.TextColumn(width="small"),
            "Spread": st.column_config.TextColumn(width="small"),
            "Top-Q hit": st.column_config.TextColumn(width="small"),
            "Turnover": st.column_config.TextColumn(width="small"),
        },
    )

st.caption(
    "All reported model evidence is historical out-of-sample research evidence. Dashboard views "
    "must not be used to retune the frozen model on the same evaluation period."
)
