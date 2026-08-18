from __future__ import annotations

import pandas as pd

MODEL_LABELS: dict[str, str] = {
    "constant": "Constant",
    "momentum_3m": "Momentum 3M",
    "technical_equal_weight_composite": "Technical Composite",
    "ridge": "Ridge",
    "elastic_net": "Elastic Net",
    "lightgbm_regressor": "LightGBM Regressor",
    "lightgbm_ranker": "LightGBM Ranker",
    "core_percentile_ensemble": "Core Percentile Ensemble",
    "equal_percentile_ensemble": "Equal Percentile Ensemble",
    "validation_weighted_ensemble": "Validation-Weighted Ensemble",
}

MODEL_ORDER: tuple[str, ...] = (
    "constant",
    "momentum_3m",
    "technical_equal_weight_composite",
    "ridge",
    "elastic_net",
    "lightgbm_regressor",
    "lightgbm_ranker",
)

ENSEMBLE_COMPONENTS: tuple[str, ...] = (
    "technical_equal_weight_composite",
    "elastic_net",
    "lightgbm_ranker",
)

STABILITY_MODEL_ORDER: tuple[str, ...] = (
    "momentum_3m",
    "technical_equal_weight_composite",
    "ridge",
    "elastic_net",
    "lightgbm_regressor",
)

ENSEMBLE_ORDER: tuple[str, ...] = (
    "core_percentile_ensemble",
    "equal_percentile_ensemble",
    "validation_weighted_ensemble",
)

SIGNAL_LABELS: dict[str, str] = {
    "composite": "Technical Composite",
    "elastic_net": "Elastic Net",
    "lightgbm_ranker": "LightGBM Ranker",
}


def model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name.replace("_", " ").title())


def model_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    order = {name: index for index, name in enumerate(MODEL_ORDER)}
    frame["model_label"] = frame["model_name"].astype(str).map(model_label)
    frame["ensemble_component"] = frame["model_name"].astype(str).isin(ENSEMBLE_COMPONENTS)
    frame["_order"] = frame["model_name"].astype(str).map(order).fillna(len(order))
    return frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def ensemble_component_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    frame = monthly.copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame = frame[frame["model_name"].astype(str).isin(ENSEMBLE_COMPONENTS)].copy()
    frame["model_label"] = frame["model_name"].astype(str).map(model_label)
    return frame.sort_values(["as_of_date", "model_name"]).reset_index(drop=True)


def yearly_stability_matrix(yearly: pd.DataFrame) -> pd.DataFrame:
    frame = yearly.copy()
    frame = frame[frame["model_name"].astype(str).isin(STABILITY_MODEL_ORDER)]
    matrix = frame.pivot(index="model_name", columns="year", values="mean_ic")
    matrix = matrix.reindex(STABILITY_MODEL_ORDER)
    matrix.index = [model_label(str(name)) for name in matrix.index]
    return matrix


def sector_stability_matrix(sector: pd.DataFrame) -> pd.DataFrame:
    frame = sector.copy()
    frame = frame[frame["model_name"].astype(str).isin(STABILITY_MODEL_ORDER)]
    matrix = frame.pivot(index="model_name", columns="sector", values="mean_sector_ic")
    matrix = matrix.reindex(STABILITY_MODEL_ORDER)
    matrix.index = [model_label(str(name)) for name in matrix.index]
    return matrix


def feature_importance_table(
    feature_importance: pd.DataFrame,
    *,
    top_n: int = 15,
) -> pd.DataFrame:
    frame = feature_importance.copy()
    frame["family"] = frame["feature"].astype(str).map(_feature_family)
    frame["feature_label"] = frame["feature"].astype(str).map(_feature_label)
    frame = frame.sort_values("mean_gain_share", ascending=False).head(top_n)
    return frame.reset_index(drop=True)


def feature_family_summary(feature_importance: pd.DataFrame) -> pd.DataFrame:
    frame = feature_importance.copy()
    frame["family"] = frame["feature"].astype(str).map(_feature_family)
    grouped = (
        frame.groupby("family", as_index=False)["mean_gain_share"]
        .sum()
        .rename(columns={"mean_gain_share": "gain_share"})
    )
    return grouped.sort_values("gain_share", ascending=False).reset_index(drop=True)


def ensemble_candidate_table(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    order = {name: index for index, name in enumerate(ENSEMBLE_ORDER)}
    frame["model_label"] = frame["model_name"].astype(str).map(model_label)
    frame["_order"] = frame["model_name"].astype(str).map(order).fillna(len(order))
    return frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def ensemble_correlation_matrix(correlations: pd.DataFrame) -> pd.DataFrame:
    signal_order = ("composite", "elastic_net", "lightgbm_ranker")
    matrix = pd.DataFrame(1.0, index=signal_order, columns=signal_order, dtype=float)
    for row in correlations.itertuples(index=False):
        signal_a = str(row.signal_a)
        signal_b = str(row.signal_b)
        value = float(row.mean_spearman)
        if signal_a in matrix.index and signal_b in matrix.columns:
            matrix.loc[signal_a, signal_b] = value
            matrix.loc[signal_b, signal_a] = value
    matrix.index = [SIGNAL_LABELS.get(name, name) for name in matrix.index]
    matrix.columns = [SIGNAL_LABELS.get(name, name) for name in matrix.columns]
    return matrix


def _feature_family(feature: str) -> str:
    if feature.startswith("tech__"):
        return "Technical"
    if feature.startswith("fund__"):
        return "Fundamental"
    return "Other"


def _feature_label(feature: str) -> str:
    label = feature
    label = label.removeprefix("tech__").removeprefix("fund__")
    label = label.replace("_sector_neutral", "")
    label = label.replace("_sector_zscore", " · sector z-score")
    label = label.replace("_zscore", " · z-score")
    return label.replace("_", " ").title()
