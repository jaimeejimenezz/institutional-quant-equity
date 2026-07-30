"""Finalize and freeze the technical-feature selection from Step 6."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_equity.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from quant_equity.features import (
    SELECTED_TECHNICAL_FEATURE_COLUMNS,
    TECHNICAL_MODEL_FEATURE_COLUMNS,
)

PANEL_PATH = PROCESSED_DATA_DIR / "technical_factor_research_panel.parquet"

SELECTION_TABLE_PATH = REPORTS_DIR / "tables" / "technical_feature_selection.csv"

REPORT_PATH = REPORTS_DIR / "research" / "technical_factor_report.md"

REPORT_MARKER = "## Final technical feature selection"


FEATURE_DECISIONS: dict[str, dict[str, str]] = {
    "amihud_illiquidity_20d_sector_neutral": {
        "family": "liquidity",
        "economic_hypothesis": (
            "Relative illiquidity may contain information about "
            "liquidity premia and trading frictions."
        ),
        "selection_reason": (
            "Highest absolute mean IC, broad sector coverage and reasonable turnover."
        ),
        "main_risk": ("Performance differs between temporal subperiods."),
    },
    "reversal_1m_sector_neutral": {
        "family": "short_term_reversal",
        "economic_hypothesis": (
            "Stocks with weak recent one-month performance may experience a short-term rebound."
        ),
        "selection_reason": (
            "Strong IC, positive results in both subperiods and broad sector evidence."
        ),
        "main_risk": ("Very high ranking turnover and potential trading costs."),
    },
    "volatility_60d_sector_neutral": {
        "family": "risk",
        "economic_hypothesis": (
            "Relative volatility may capture the low-risk anomaly "
            "and differences in required returns."
        ),
        "selection_reason": ("High directional spread combined with low turnover."),
        "main_risk": ("Limited consistency across individual years and sectors."),
    },
    "distance_sma_50d_sector_neutral": {
        "family": "short_term_trend",
        "economic_hypothesis": (
            "Distance from the 50-day moving average captures "
            "short-term trend strength or price overextension."
        ),
        "selection_reason": ("Useful IC and reasonable temporal and sector stability."),
        "main_risk": ("High turnover and partial overlap with one-month reversal."),
    },
    "beta_60d_market_sector_neutral": {
        "family": "market_risk",
        "economic_hypothesis": (
            "Market beta captures differences in systematic risk "
            "and exposure to broad market movements."
        ),
        "selection_reason": (
            "Largest directional spread, moderate turnover and positive results in both subperiods."
        ),
        "main_risk": ("Partial correlation with total volatility."),
    },
    "return_3m_sector_neutral": {
        "family": "momentum",
        "economic_hypothesis": (
            "Recent medium-term winners may continue outperforming "
            "because information is incorporated gradually."
        ),
        "selection_reason": (
            "Positive results in both subperiods and useful exposure to the momentum family."
        ),
        "main_risk": ("Predictive evidence is concentrated in relatively few sectors."),
    },
    "max_drawdown_126d_sector_neutral": {
        "family": "deep_reversal",
        "economic_hypothesis": (
            "Stocks experiencing deep medium-term drawdowns may subsequently recover."
        ),
        "selection_reason": ("Large directional spread and relatively low turnover."),
        "main_risk": ("Evidence is dependent on the market regime."),
    },
    "average_dollar_volume_20d_sector_neutral": {
        "family": "liquidity_capacity",
        "economic_hypothesis": (
            "Dollar trading volume captures relative liquidity, "
            "market attention and implementation capacity."
        ),
        "selection_reason": ("Low turnover and complementary liquidity information."),
        "main_risk": ("Weak individual predictive strength."),
    },
    "return_1m_sector_neutral": {
        "family": "short_term_return",
        "economic_hypothesis": ("Recent one-month returns may exhibit continuation or reversal."),
        "selection_reason": (
            "Excluded because it is the exact inverse representation of reversal_1m."
        ),
        "main_risk": ("Perfect redundancy with the selected reversal signal."),
    },
    "distance_sma_200d_sector_neutral": {
        "family": "long_term_trend",
        "economic_hypothesis": (
            "Distance from the 200-day moving average captures long-term trend strength."
        ),
        "selection_reason": (
            "Excluded because IC and quintile-spread evidence point in conflicting directions."
        ),
        "main_risk": ("Unstable and economically inconsistent ranking behaviour."),
    },
    "sma_50_200_spread_sector_neutral": {
        "family": "moving_average_trend",
        "economic_hypothesis": (
            "The spread between short and long moving averages "
            "captures medium-term trend direction."
        ),
        "selection_reason": (
            "Excluded because it is weak and highly redundant with other trend variables."
        ),
        "main_risk": ("High correlation with momentum_6_1 and limited stability."),
    },
    "volatility_20d_sector_neutral": {
        "family": "short_term_risk",
        "economic_hypothesis": (
            "Short-term volatility may capture changes in risk and investor uncertainty."
        ),
        "selection_reason": (
            "Excluded in favour of volatility_60d, which provides better IC and lower turnover."
        ),
        "main_risk": ("High turnover and limited sector breadth."),
    },
    "momentum_6_1_sector_neutral": {
        "family": "medium_term_momentum",
        "economic_hypothesis": (
            "Returns over months six to one may persist because of gradual information diffusion."
        ),
        "selection_reason": (
            "Excluded because the evidence is weak and redundant with other trend variables."
        ),
        "main_risk": ("High correlation with the moving-average trend signal."),
    },
    "return_1w_sector_neutral": {
        "family": "very_short_term_return",
        "economic_hypothesis": (
            "One-week returns may capture very short-term continuation or reversal."
        ),
        "selection_reason": (
            "Excluded because predictive strength is small relative to its turnover."
        ),
        "main_risk": ("Extremely high trading intensity."),
    },
    "downside_volatility_60d_sector_neutral": {
        "family": "downside_risk",
        "economic_hypothesis": ("Downside volatility isolates adverse price variation."),
        "selection_reason": (
            "Excluded because it is weak and largely overlaps with volatility_60d."
        ),
        "main_risk": ("Low temporal stability and substantial redundancy."),
    },
    "dollar_volume_change_20d_60d_sector_neutral": {
        "family": "liquidity_change",
        "economic_hypothesis": (
            "Changes in dollar trading volume may indicate shifts in investor attention."
        ),
        "selection_reason": ("Excluded because its mean IC is close to zero."),
        "main_risk": ("Very high turnover without meaningful predictive evidence."),
    },
    "momentum_12_1_sector_neutral": {
        "family": "long_term_momentum",
        "economic_hypothesis": ("Returns over months twelve to one may exhibit persistence."),
        "selection_reason": (
            "Excluded because its mean IC is close to zero in the research sample."
        ),
        "main_risk": ("No meaningful individual predictive evidence."),
    },
    "positive_day_ratio_60d_sector_neutral": {
        "family": "trend_breadth",
        "economic_hypothesis": (
            "The proportion of positive sessions may measure the breadth of a price trend."
        ),
        "selection_reason": ("Excluded because its mean IC is effectively zero."),
        "main_risk": ("Does not meaningfully distinguish future winners and losers."),
    },
    "zero_volume_ratio_60d_sector_neutral": {
        "family": "trading_activity",
        "economic_hypothesis": (
            "The frequency of zero-volume sessions may identify illiquid securities."
        ),
        "selection_reason": ("Excluded because it has no variation in the current universe."),
        "main_risk": ("Cannot rank the large and liquid companies in the universe."),
    },
}


def _format_value(value: Any) -> str:
    """Format a value for a Markdown table."""
    if value is None or pd.isna(value):
        return ""

    return str(value).replace("|", "\\|")


def _to_markdown(frame: pd.DataFrame) -> str:
    """Convert a dataframe to a Markdown table without extra dependencies."""
    if frame.empty:
        return "_No observations._"

    columns = [str(column) for column in frame.columns]

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in frame.itertuples(index=False, name=None):
        values = [_format_value(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _validate_selection(panel: pd.DataFrame) -> None:
    """Validate the frozen technical-feature selection."""
    full_features = tuple(TECHNICAL_MODEL_FEATURE_COLUMNS)
    selected_features = tuple(SELECTED_TECHNICAL_FEATURE_COLUMNS)

    if len(full_features) != 19:
        raise ValueError(f"Expected 19 technical model features, but found {len(full_features)}.")

    if not 8 <= len(selected_features) <= 15:
        raise ValueError("The selected technical feature count must be between 8 and 15.")

    if len(selected_features) != len(set(selected_features)):
        raise ValueError("The selected technical features contain duplicates.")

    unknown_selected = sorted(set(selected_features).difference(full_features))

    if unknown_selected:
        raise ValueError(
            "Selected features not present in the full feature set: " + ", ".join(unknown_selected)
        )

    missing_panel_columns = sorted(set(full_features).difference(panel.columns))

    if missing_panel_columns:
        raise ValueError(
            "The research panel is missing technical features: " + ", ".join(missing_panel_columns)
        )

    missing_decisions = sorted(set(full_features).difference(FEATURE_DECISIONS))

    unexpected_decisions = sorted(set(FEATURE_DECISIONS).difference(full_features))

    if missing_decisions:
        raise ValueError("Missing decision metadata for: " + ", ".join(missing_decisions))

    if unexpected_decisions:
        raise ValueError(
            "Decision metadata contains unknown features: " + ", ".join(unexpected_decisions)
        )


def _build_selection_table() -> pd.DataFrame:
    """Build the final selected and excluded feature table."""
    selected_order = {
        signal: position
        for position, signal in enumerate(
            SELECTED_TECHNICAL_FEATURE_COLUMNS,
            start=1,
        )
    }

    rows: list[dict[str, Any]] = []

    for signal in TECHNICAL_MODEL_FEATURE_COLUMNS:
        metadata = FEATURE_DECISIONS[signal]
        is_selected = signal in selected_order

        rows.append(
            {
                "signal": signal,
                "decision": ("selected" if is_selected else "excluded"),
                "selected_order": (selected_order[signal] if is_selected else pd.NA),
                "family": metadata["family"],
                "economic_hypothesis": (metadata["economic_hypothesis"]),
                "selection_reason": (metadata["selection_reason"]),
                "main_risk": metadata["main_risk"],
                "research_start": "2014-01-01",
                "research_end": "2019-12-31",
                "selection_status": "frozen",
            }
        )

    selection_table = pd.DataFrame(rows)

    selection_table["selected_order"] = selection_table["selected_order"].astype("Int64")

    return selection_table.sort_values(
        ["decision", "selected_order", "signal"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _write_selection_table(
    selection_table: pd.DataFrame,
) -> Path:
    """Write the final feature-selection table."""
    SELECTION_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_table.to_csv(
        SELECTION_TABLE_PATH,
        index=False,
    )

    return SELECTION_TABLE_PATH


def _append_selection_report(
    selection_table: pd.DataFrame,
) -> Path:
    """Append the frozen final selection to the research report."""
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_report = (
        REPORT_PATH.read_text(encoding="utf-8")
        if REPORT_PATH.exists()
        else "# Technical Factor Research Report\n"
    )

    base_report = existing_report.split(REPORT_MARKER)[0].rstrip()

    selected = selection_table.loc[
        selection_table["decision"].eq("selected"),
        [
            "selected_order",
            "signal",
            "family",
            "selection_reason",
            "main_risk",
        ],
    ].sort_values("selected_order")

    excluded = selection_table.loc[
        selection_table["decision"].eq("excluded"),
        [
            "signal",
            "family",
            "selection_reason",
        ],
    ].sort_values("signal")

    lines = [
        base_report,
        "",
        REPORT_MARKER,
        "",
        "### Frozen decision",
        "",
        ("The technical feature set was selected exclusively using the 2014-2019 research sample."),
        "",
        (
            "No observations from 2020 onward were used to choose, "
            "remove or replace these variables."
        ),
        "",
        (
            "The selection is frozen before any out-of-sample model "
            "evaluation. Future results may be used to evaluate the "
            "selection, but not to rewrite this historical decision."
        ),
        "",
        "### Selected features",
        "",
        _to_markdown(selected),
        "",
        "### Excluded features",
        "",
        _to_markdown(excluded),
        "",
        "### Final counts",
        "",
        f"- Signals evaluated: `{len(selection_table)}`",
        (f"- Signals selected: `{selection_table['decision'].eq('selected').sum()}`"),
        (f"- Signals excluded: `{selection_table['decision'].eq('excluded').sum()}`"),
        "- Research period: `2014-2019`",
        "- Selection status: `FROZEN`",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return REPORT_PATH


def main() -> None:
    """Finalize the Step 6 technical-feature selection."""
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"Technical factor research panel was not found: {PANEL_PATH}")

    panel = pd.read_parquet(PANEL_PATH)

    _validate_selection(panel)

    selection_table = _build_selection_table()

    table_path = _write_selection_table(selection_table)

    report_path = _append_selection_report(selection_table)

    selected_count = int(selection_table["decision"].eq("selected").sum())

    excluded_count = int(selection_table["decision"].eq("excluded").sum())

    print()
    print("Institutional Quant Equity Research Platform")
    print("Technical feature selection - Step 6C")
    print("-" * 48)
    print(f"Signals evaluated: {len(selection_table)}")
    print(f"Signals selected: {selected_count}")
    print(f"Signals excluded: {excluded_count}")
    print("Research period: 2014-2019")
    print("Out-of-sample period inspected: No")
    print("Selection status: FROZEN")
    print()
    print(f"Selection table: {table_path}")
    print(f"Research report: {report_path}")


if __name__ == "__main__":
    main()
