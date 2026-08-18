from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OverviewSummary:
    strategy_name: str
    net_cagr: float
    net_sharpe_ratio: float
    net_sortino_ratio: float
    net_maximum_drawdown: float
    net_beta_vs_spy: float
    net_annualized_alpha_vs_spy: float
    mean_one_way_turnover: float
    total_transaction_cost: float


def strategy_summary(frame: pd.DataFrame, strategy_name: str) -> OverviewSummary:
    row = _single_strategy_row(frame, strategy_name, "strategy_name")
    return OverviewSummary(
        strategy_name=strategy_name,
        net_cagr=float(row["net_cagr"]),
        net_sharpe_ratio=float(row["net_sharpe_ratio"]),
        net_sortino_ratio=float(row["net_sortino_ratio"]),
        net_maximum_drawdown=float(row["net_maximum_drawdown"]),
        net_beta_vs_spy=float(row["net_beta_vs_spy"]),
        net_annualized_alpha_vs_spy=float(row["net_annualized_alpha_vs_spy"]),
        mean_one_way_turnover=float(row["mean_one_way_turnover"]),
        total_transaction_cost=float(row["total_transaction_cost"]),
    )


def build_performance_index(
    net_daily: pd.DataFrame,
    benchmark: pd.DataFrame,
    strategy_name: str,
    *,
    baseline_strategy: str = "top_n_equal_weight",
) -> pd.DataFrame:
    selected = _strategy_curve(net_daily, strategy_name)
    spy = _benchmark_curve(benchmark)

    merged = selected.merge(spy, on="date", how="inner")

    include_baseline = baseline_strategy != strategy_name
    if include_baseline:
        baseline = _strategy_curve(net_daily, baseline_strategy).rename(
            columns={"portfolio_value": "baseline_value"}
        )
        merged = merged.merge(baseline, on="date", how="inner")

    merged = merged.sort_values("date").reset_index(drop=True)
    if merged.empty:
        raise ValueError("No common dates exist for the selected performance series.")

    selected_index = _normalize_index(merged["portfolio_value"])
    spy_index = _normalize_index(merged["adjusted_close"])

    parts = [
        pd.DataFrame(
            {
                "date": merged["date"],
                "series": strategy_name,
                "role": "selected",
                "index_value": selected_index,
            }
        ),
        pd.DataFrame(
            {
                "date": merged["date"],
                "series": "SPY",
                "role": "benchmark",
                "index_value": spy_index,
            }
        ),
    ]

    if include_baseline:
        baseline_index = _normalize_index(merged["baseline_value"])
        parts.append(
            pd.DataFrame(
                {
                    "date": merged["date"],
                    "series": baseline_strategy,
                    "role": "baseline",
                    "index_value": baseline_index,
                }
            )
        )

    return pd.concat(parts, ignore_index=True)


def build_drawdown_series(performance_index: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "series", "role", "index_value"}
    missing = required - set(performance_index.columns)
    if missing:
        raise ValueError(f"Performance index is missing columns: {sorted(missing)}")

    parts: list[pd.DataFrame] = []
    for (series, role), group in performance_index.groupby(["series", "role"], sort=False):
        if role == "baseline":
            continue

        ordered = group.sort_values("date").copy()
        wealth = ordered["index_value"].astype(float)
        running_peak = wealth.cummax()
        ordered["drawdown"] = wealth / running_peak - 1.0
        ordered["series"] = series
        ordered["role"] = role
        parts.append(ordered[["date", "series", "role", "drawdown"]])

    if not parts:
        raise ValueError("No selected or benchmark drawdown series are available.")

    return pd.concat(parts, ignore_index=True)


def latest_portfolio_weights(
    target_weights: pd.DataFrame,
    strategy_name: str,
    *,
    minimum_weight: float = 1e-10,
) -> pd.DataFrame:
    subset = target_weights.loc[target_weights["method"] == strategy_name].copy()
    if subset.empty:
        raise ValueError(f"No target weights found for strategy {strategy_name!r}.")

    subset["as_of_date"] = pd.to_datetime(subset["as_of_date"], errors="coerce")
    latest = subset["as_of_date"].max()
    if pd.isna(latest):
        raise ValueError("Target weights contain no valid as-of dates.")

    snapshot = subset.loc[subset["as_of_date"] == latest].copy()
    snapshot = snapshot.loc[snapshot["weight"].astype(float) > minimum_weight]
    if snapshot.empty:
        raise ValueError("Latest target-weight snapshot contains no positive positions.")

    return snapshot.sort_values("weight", ascending=False).reset_index(drop=True)


def sector_exposure(portfolio_weights: pd.DataFrame) -> pd.DataFrame:
    if portfolio_weights.empty:
        raise ValueError("Portfolio weights are empty.")

    exposure = (
        portfolio_weights.groupby("sector", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "sector_weight"})
        .sort_values("sector_weight", ascending=False)
        .reset_index(drop=True)
    )
    return exposure


def latest_strategy_row(
    frame: pd.DataFrame,
    strategy_name: str,
    *,
    strategy_column: str,
    date_column: str | None = None,
) -> pd.Series:
    subset = frame.loc[frame[strategy_column] == strategy_name].copy()
    if subset.empty:
        raise ValueError(f"No rows found for strategy {strategy_name!r} in {strategy_column!r}.")

    if date_column is None:
        if len(subset) != 1:
            raise ValueError(f"Expected one row for {strategy_name!r}, found {len(subset)}.")
        return subset.iloc[0].copy()

    subset[date_column] = pd.to_datetime(subset[date_column], errors="coerce")
    latest = subset[date_column].max()
    latest_rows = subset.loc[subset[date_column] == latest]
    if len(latest_rows) != 1:
        raise ValueError(
            f"Expected one latest row for {strategy_name!r}, found {len(latest_rows)}."
        )
    return latest_rows.iloc[0].copy()


def _single_strategy_row(
    frame: pd.DataFrame,
    strategy_name: str,
    strategy_column: str,
) -> pd.Series:
    subset = frame.loc[frame[strategy_column] == strategy_name]
    if len(subset) != 1:
        raise ValueError(f"Expected one row for {strategy_name!r}, found {len(subset)}.")
    return subset.iloc[0].copy()


def _strategy_curve(frame: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    subset = frame.loc[
        frame["strategy_name"] == strategy_name,
        ["date", "portfolio_value"],
    ].copy()
    if subset.empty:
        raise ValueError(f"No daily performance found for strategy {strategy_name!r}.")

    subset["date"] = pd.to_datetime(subset["date"], errors="coerce")
    subset["portfolio_value"] = pd.to_numeric(subset["portfolio_value"], errors="coerce")
    subset = subset.dropna(subset=["date", "portfolio_value"])
    subset = subset.sort_values("date").drop_duplicates("date", keep="last")
    return subset


def _benchmark_curve(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame.copy()
    if "ticker" in subset.columns:
        subset = subset.loc[subset["ticker"].astype(str) == "SPY"]

    subset = subset[["date", "adjusted_close"]].copy()
    subset["date"] = pd.to_datetime(subset["date"], errors="coerce")
    subset["adjusted_close"] = pd.to_numeric(subset["adjusted_close"], errors="coerce")
    subset = subset.dropna(subset=["date", "adjusted_close"])
    subset = subset.sort_values("date").drop_duplicates("date", keep="last")
    return subset


def _normalize_index(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError("Performance series contains invalid numeric values.")

    first = float(numeric.iloc[0])
    if not np.isfinite(first) or first <= 0.0:
        raise ValueError("Performance series must start from a finite positive value.")

    return numeric.astype(float) / first * 100.0


def signal_dates(alpha_signal: pd.DataFrame) -> tuple[pd.Timestamp, ...]:
    dates = pd.to_datetime(alpha_signal["as_of_date"], errors="coerce").dropna()
    unique = sorted(pd.Timestamp(value) for value in dates.unique())
    return tuple(unique)


def build_alpha_snapshot(
    alpha_signal: pd.DataFrame,
    security_risk: pd.DataFrame,
    target_weights: pd.DataFrame,
    strategy_name: str,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    selected_date = pd.Timestamp(as_of_date).normalize()

    signal = alpha_signal.copy()
    signal["as_of_date"] = pd.to_datetime(signal["as_of_date"], errors="coerce")
    signal = signal.loc[signal["as_of_date"] == selected_date].copy()
    if signal.empty:
        raise ValueError(f"No alpha signal found for {selected_date.date()}.")

    risk = security_risk.copy()
    risk["as_of_date"] = pd.to_datetime(risk["as_of_date"], errors="coerce")
    risk = risk.loc[risk["as_of_date"] == selected_date].copy()
    if risk.empty:
        raise ValueError(f"No security risk found for {selected_date.date()}.")

    weights = target_weights.loc[target_weights["method"] == strategy_name].copy()
    weights["as_of_date"] = pd.to_datetime(weights["as_of_date"], errors="coerce")
    weights = weights.loc[weights["as_of_date"] == selected_date].copy()

    risk_columns = [
        "as_of_date",
        "ticker",
        "annualized_volatility",
        "annualized_downside_volatility",
        "beta_vs_spy",
        "correlation_vs_spy",
        "average_dollar_volume",
    ]
    weight_columns = ["as_of_date", "ticker", "weight"]

    snapshot = signal.merge(
        risk[risk_columns],
        on=["as_of_date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    snapshot = snapshot.merge(
        weights[weight_columns].rename(columns={"weight": "selected_weight"}),
        on=["as_of_date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    snapshot["selected_weight"] = snapshot["selected_weight"].fillna(0.0)

    if snapshot["ticker"].duplicated().any():
        raise ValueError("Alpha snapshot contains duplicate tickers.")
    if snapshot["rank"].duplicated().any():
        raise ValueError("Alpha snapshot contains duplicate ranks.")

    return snapshot.sort_values("rank").reset_index(drop=True)


def ensemble_weights(snapshot: pd.DataFrame) -> dict[str, float]:
    columns = {
        "Technical composite": "composite_weight",
        "Elastic Net": "elastic_net_weight",
        "LightGBM ranker": "lightgbm_ranker_weight",
    }
    values: dict[str, float] = {}

    for label, column in columns.items():
        numeric = pd.to_numeric(snapshot[column], errors="coerce").dropna()
        if numeric.empty:
            raise ValueError(f"No valid values found for {column!r}.")
        if not np.allclose(numeric.to_numpy(dtype=float), float(numeric.iloc[0])):
            raise ValueError(f"Ensemble weight {column!r} is not constant in the snapshot.")
        values[label] = float(numeric.iloc[0])

    if not np.isclose(sum(values.values()), 1.0, atol=1e-8):
        raise ValueError("Ensemble weights do not sum to one.")
    return values


def parse_model_contributions(value: str) -> pd.DataFrame:
    import json

    parsed = json.loads(str(value))
    order = (
        ("technical_composite", "Technical composite"),
        ("elastic_net", "Elastic Net"),
        ("lightgbm_ranker", "LightGBM ranker"),
    )
    rows = []
    for key, label in order:
        payload = parsed.get(key)
        if not isinstance(payload, dict):
            raise ValueError(f"Missing model contribution payload for {key!r}.")
        rows.append(
            {
                "component": label,
                "percentile": float(payload["percentile"]),
                "weight": float(payload["weight"]),
                "contribution": float(payload["contribution"]),
            }
        )
    return pd.DataFrame(rows)


def portfolio_dates(
    target_weights: pd.DataFrame,
    strategy_name: str,
) -> tuple[pd.Timestamp, ...]:
    subset = target_weights.loc[target_weights["method"] == strategy_name, "as_of_date"]
    values = pd.to_datetime(subset, errors="coerce").dropna().sort_values().unique()
    return tuple(pd.Timestamp(value) for value in values)


def portfolio_snapshot(
    target_weights: pd.DataFrame,
    strategy_name: str,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    selected_date = pd.Timestamp(as_of_date).normalize()
    method_rows = target_weights.loc[target_weights["method"] == strategy_name].copy()
    method_rows["as_of_date"] = pd.to_datetime(method_rows["as_of_date"], errors="coerce")

    current = method_rows.loc[method_rows["as_of_date"] == selected_date].copy()
    if current.empty:
        raise ValueError(
            f"No target weights found for {strategy_name!r} on {selected_date.date()}."
        )

    current["weight"] = pd.to_numeric(current["weight"], errors="raise")
    previous_dates = method_rows.loc[
        method_rows["as_of_date"] < selected_date,
        "as_of_date",
    ].dropna()

    if previous_dates.empty:
        current["previous_weight"] = pd.to_numeric(
            current["previous_weight"], errors="coerce"
        ).fillna(0.0)
        current["weight_delta"] = current["weight"] - current["previous_weight"]
        return current.sort_values("weight", ascending=False).reset_index(drop=True)

    previous_date = pd.Timestamp(previous_dates.max())
    previous = method_rows.loc[
        method_rows["as_of_date"] == previous_date,
        ["ticker", "sector", "weight"],
    ].copy()
    previous["weight"] = pd.to_numeric(previous["weight"], errors="raise")
    previous = previous.rename(
        columns={
            "sector": "previous_sector",
            "weight": "previous_weight",
        }
    )

    current = current.drop(columns=["previous_weight"], errors="ignore")
    snapshot = current.merge(previous, on="ticker", how="outer")

    snapshot["as_of_date"] = pd.to_datetime(snapshot["as_of_date"], errors="coerce").fillna(
        selected_date
    )
    snapshot["method"] = snapshot["method"].fillna(strategy_name)
    snapshot["sector"] = snapshot["sector"].combine_first(snapshot["previous_sector"])
    snapshot["weight"] = pd.to_numeric(snapshot["weight"], errors="coerce").fillna(0.0)
    snapshot["previous_weight"] = pd.to_numeric(
        snapshot["previous_weight"], errors="coerce"
    ).fillna(0.0)
    snapshot["weight_delta"] = snapshot["weight"] - snapshot["previous_weight"]

    snapshot = snapshot.drop(columns=["previous_sector"], errors="ignore")
    return snapshot.sort_values(
        ["weight", "previous_weight"],
        ascending=False,
    ).reset_index(drop=True)


def enrich_portfolio_snapshot(
    snapshot: pd.DataFrame,
    security_risk: pd.DataFrame,
) -> pd.DataFrame:
    if snapshot.empty:
        raise ValueError("Portfolio snapshot is empty.")

    enriched = snapshot.copy()
    enriched["as_of_date"] = pd.to_datetime(enriched["as_of_date"], errors="coerce")
    selected_date = pd.Timestamp(enriched["as_of_date"].dropna().iloc[0])

    risk = security_risk.copy()
    risk["as_of_date"] = pd.to_datetime(risk["as_of_date"], errors="coerce")
    risk = risk.loc[
        risk["as_of_date"] == selected_date,
        ["ticker", "beta_vs_spy", "average_dollar_volume"],
    ].copy()
    risk = risk.rename(
        columns={
            "beta_vs_spy": "risk_beta_vs_spy",
            "average_dollar_volume": "risk_average_dollar_volume",
        }
    )

    enriched = enriched.merge(risk, on="ticker", how="left", validate="one_to_one")

    beta = pd.to_numeric(enriched["beta_vs_spy"], errors="coerce")
    risk_beta = pd.to_numeric(enriched["risk_beta_vs_spy"], errors="coerce")
    enriched["beta_vs_spy"] = beta.combine_first(risk_beta)

    adv = pd.to_numeric(enriched["average_dollar_volume"], errors="coerce")
    risk_adv = pd.to_numeric(
        enriched["risk_average_dollar_volume"],
        errors="coerce",
    )
    enriched["average_dollar_volume"] = adv.combine_first(risk_adv)

    return enriched.drop(columns=["risk_beta_vs_spy", "risk_average_dollar_volume"])


def portfolio_sector_changes(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        raise ValueError("Portfolio snapshot is empty.")
    current = snapshot.groupby("sector", as_index=False)["weight"].sum()
    current = current.rename(columns={"weight": "current_weight"})
    previous = snapshot.groupby("sector", as_index=False)["previous_weight"].sum()
    previous = previous.rename(columns={"previous_weight": "previous_weight"})
    exposure = current.merge(previous, on="sector", how="outer").fillna(0.0)
    exposure["weight_delta"] = exposure["current_weight"] - exposure["previous_weight"]
    return exposure.sort_values("current_weight", ascending=False).reset_index(drop=True)


def portfolio_diagnostics_row(
    diagnostics: pd.DataFrame,
    strategy_name: str,
    as_of_date: pd.Timestamp,
) -> pd.Series:
    return _dated_method_row(diagnostics, strategy_name, as_of_date)


def portfolio_risk_row(
    risk_summary: pd.DataFrame,
    strategy_name: str,
    as_of_date: pd.Timestamp,
) -> pd.Series:
    return _dated_method_row(risk_summary, strategy_name, as_of_date)


def realized_positions_for_signal(
    positions: pd.DataFrame,
    strategy_name: str,
    signal_date: pd.Timestamp,
) -> pd.DataFrame:
    selected_signal = pd.Timestamp(signal_date).normalize()
    subset = positions.loc[positions["strategy_name"] == strategy_name].copy()
    subset["date"] = pd.to_datetime(subset["date"], errors="coerce")
    subset["active_signal_date"] = pd.to_datetime(subset["active_signal_date"], errors="coerce")
    subset = subset.loc[subset["active_signal_date"] == selected_signal].copy()
    if subset.empty:
        raise ValueError(f"No realized positions found for signal date {selected_signal.date()}.")

    latest_date = subset["date"].max()
    snapshot = subset.loc[subset["date"] == latest_date].copy()
    snapshot = snapshot.loc[snapshot["actual_weight"].astype(float) > 1e-10]
    if snapshot.empty:
        raise ValueError("Realized position snapshot contains no positive holdings.")
    return snapshot.sort_values("actual_weight", ascending=False).reset_index(drop=True)


def turnover_history(
    diagnostics: pd.DataFrame,
    strategy_name: str,
    *,
    baseline_strategy: str = "top_n_equal_weight",
) -> pd.DataFrame:
    methods = [strategy_name]
    if strategy_name != baseline_strategy:
        methods.append(baseline_strategy)

    subset = diagnostics.loc[diagnostics["method"].isin(methods)].copy()
    subset["as_of_date"] = pd.to_datetime(subset["as_of_date"], errors="coerce")
    subset = subset.dropna(subset=["as_of_date", "one_way_turnover"])
    subset["role"] = np.where(
        subset["method"] == strategy_name,
        "selected",
        "baseline",
    )
    return subset.sort_values(["method", "as_of_date"]).reset_index(drop=True)


def portfolio_method_comparison(
    diagnostics: pd.DataFrame,
    risk_summary: pd.DataFrame,
    strategy_name: str,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    selected_date = pd.Timestamp(as_of_date).normalize()

    diagnostics_copy = diagnostics.copy()
    diagnostics_copy["as_of_date"] = pd.to_datetime(diagnostics_copy["as_of_date"], errors="coerce")
    diagnostics_copy = diagnostics_copy.loc[diagnostics_copy["as_of_date"] == selected_date]

    risk_copy = risk_summary.copy()
    risk_copy["as_of_date"] = pd.to_datetime(risk_copy["as_of_date"], errors="coerce")
    risk_copy = risk_copy.loc[risk_copy["as_of_date"] == selected_date]

    columns = [
        "as_of_date",
        "method",
        "predicted_volatility",
        "portfolio_beta_vs_spy",
        "maximum_liquidation_days",
    ]
    comparison = diagnostics_copy.merge(
        risk_copy[columns],
        on=["as_of_date", "method"],
        how="inner",
        validate="one_to_one",
    )
    if comparison.empty:
        raise ValueError(f"No method comparison found for {selected_date.date()}.")

    comparison["role"] = np.where(
        comparison["method"] == strategy_name,
        "selected",
        "other",
    )
    comparison["label"] = comparison["method"].map(lambda method: method.replace("_", " ").title())
    return comparison.sort_values(["role", "method"], ascending=[False, True])


def _dated_method_row(
    frame: pd.DataFrame,
    strategy_name: str,
    as_of_date: pd.Timestamp,
) -> pd.Series:
    selected_date = pd.Timestamp(as_of_date).normalize()
    subset = frame.loc[frame["method"] == strategy_name].copy()
    subset["as_of_date"] = pd.to_datetime(subset["as_of_date"], errors="coerce")
    subset = subset.loc[subset["as_of_date"] == selected_date]
    if len(subset) != 1:
        raise ValueError(
            f"Expected one row for {strategy_name!r} on {selected_date.date()}, "
            f"found {len(subset)}."
        )
    return subset.iloc[0].copy()
