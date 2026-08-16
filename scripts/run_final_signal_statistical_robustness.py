"""Evaluate final-signal IC, spread and sector robustness."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_equity.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from quant_equity.logging_config import configure_logging

FINAL_SIGNAL_PATH = PROCESSED_DATA_DIR / "final_alpha_signal.parquet"

MODELING_PANEL_PATH = PROCESSED_DATA_DIR / "modeling_panel.parquet"

TABLES_DIR = REPORTS_DIR / "tables"

MONTHLY_PATH = TABLES_DIR / "robustness_final_signal_monthly.csv"

BOOTSTRAP_PATH = TABLES_DIR / "robustness_final_signal_bootstrap.csv"

YEARLY_PATH = TABLES_DIR / "robustness_final_signal_yearly.csv"

SECTOR_PATH = TABLES_DIR / "robustness_final_signal_sector.csv"

CHECKS_PATH = TABLES_DIR / "robustness_final_signal_checks.csv"

REPORT_PATH = REPORTS_DIR / "robustness" / "final_signal_statistical_robustness.md"

TARGET_COLUMN = "target_21d_excess"

BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_SEED = 42
BOOTSTRAP_BLOCK_LENGTH = 3
CONFIDENCE_LEVEL = 0.95

MINIMUM_MONTHS = 60
MINIMUM_SECTOR_COMPANIES = 3
MINIMUM_SECTOR_MONTHS = 12


def _write_csv(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write one CSV table."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        path,
        index=False,
    )


def _require_columns(
    data: pd.DataFrame,
    columns: set[str],
    *,
    dataset_name: str,
) -> None:
    """Require a set of dataframe columns."""
    missing = sorted(columns.difference(data.columns))

    if missing:
        raise ValueError(f"{dataset_name} is missing columns: " + ", ".join(missing) + ".")


def _prepare_research_panel(
    final_signal: pd.DataFrame,
    modeling_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Join the production-safe final signal to realized OOS targets."""
    _require_columns(
        final_signal,
        {
            "as_of_date",
            "ticker",
            "sector",
            "percentile_score",
            "rank",
        },
        dataset_name="Final alpha signal",
    )

    _require_columns(
        modeling_panel,
        {
            "as_of_date",
            "ticker",
            TARGET_COLUMN,
        },
        dataset_name="Modeling panel",
    )

    signal = final_signal.loc[
        :,
        [
            "as_of_date",
            "ticker",
            "sector",
            "percentile_score",
            "rank",
        ],
    ].copy()

    signal["as_of_date"] = pd.to_datetime(
        signal["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    signal["ticker"] = signal["ticker"].astype("string").str.strip()

    signal["sector"] = signal["sector"].astype("string").str.strip()

    for column in (
        "percentile_score",
        "rank",
    ):
        signal[column] = pd.to_numeric(
            signal[column],
            errors="coerce",
        )

    if (
        signal["as_of_date"].isna().any()
        or signal["ticker"].isna().any()
        or signal["sector"].isna().any()
        or signal[
            [
                "percentile_score",
                "rank",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Final alpha signal contains invalid research keys or scores.")

    if signal.duplicated(
        [
            "as_of_date",
            "ticker",
        ]
    ).any():
        raise ValueError("Final alpha signal contains duplicated date-ticker rows.")

    targets = modeling_panel.loc[
        :,
        [
            "as_of_date",
            "ticker",
            TARGET_COLUMN,
        ],
    ].copy()

    targets["as_of_date"] = pd.to_datetime(
        targets["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    targets["ticker"] = targets["ticker"].astype("string").str.strip()

    targets[TARGET_COLUMN] = pd.to_numeric(
        targets[TARGET_COLUMN],
        errors="coerce",
    )

    if targets.duplicated(
        [
            "as_of_date",
            "ticker",
        ]
    ).any():
        raise ValueError("Modeling panel contains duplicated date-ticker rows.")

    research = signal.merge(
        targets,
        on=[
            "as_of_date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    missing_target = research[TARGET_COLUMN].isna()

    if missing_target.any():
        sample = research.loc[
            missing_target,
            [
                "as_of_date",
                "ticker",
            ],
        ].iloc[0]

        raise ValueError(
            f"Missing realized OOS target for {sample['ticker']} on {sample['as_of_date'].date()}."
        )

    if not np.isfinite(
        research[
            [
                "percentile_score",
                "rank",
                TARGET_COLUMN,
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Final-signal research panel contains non-finite values.")

    return research.sort_values(
        [
            "as_of_date",
            "rank",
            "ticker",
        ]
    ).reset_index(drop=True)


def _spearman(
    left: pd.Series,
    right: pd.Series,
) -> float:
    """Calculate Spearman correlation through rank correlation."""
    if len(left) < 3:
        return float("nan")

    left_rank = pd.to_numeric(
        left,
        errors="coerce",
    ).rank(method="average")

    right_rank = pd.to_numeric(
        right,
        errors="coerce",
    ).rank(method="average")

    if left_rank.nunique() < 2 or right_rank.nunique() < 2:
        return float("nan")

    return float(left_rank.corr(right_rank))


def _monthly_metrics(
    research: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate final-signal OOS ranking metrics by month."""
    rows = []

    for as_of_date, group in research.groupby(
        "as_of_date",
        sort=True,
    ):
        group = group.sort_values(
            [
                "rank",
                "ticker",
            ]
        ).reset_index(drop=True)

        cross_section_size = len(group)

        if cross_section_size < 5:
            raise ValueError("Final-signal monthly cross-section is too small.")

        quintile_size = max(
            1,
            int(np.floor(cross_section_size * 0.20)),
        )

        top = group.nsmallest(
            quintile_size,
            "rank",
        )

        bottom = group.nlargest(
            quintile_size,
            "rank",
        )

        actual_top = set(
            group.nlargest(
                quintile_size,
                TARGET_COLUMN,
            )["ticker"].astype(str)
        )

        predicted_top = set(top["ticker"].astype(str))

        information_coefficient = _spearman(
            group["percentile_score"],
            group[TARGET_COLUMN],
        )

        top_bottom_spread = float(top[TARGET_COLUMN].mean() - bottom[TARGET_COLUMN].mean())

        precision = float(len(predicted_top.intersection(actual_top)) / quintile_size)

        rows.append(
            {
                "as_of_date": pd.Timestamp(as_of_date),
                "cross_section_size": int(cross_section_size),
                "quintile_size": int(quintile_size),
                "information_coefficient": (information_coefficient),
                "top_bottom_spread": (top_bottom_spread),
                "top_quintile_precision": (precision),
                "top_quintile_mean_target": float(top[TARGET_COLUMN].mean()),
                "bottom_quintile_mean_target": float(bottom[TARGET_COLUMN].mean()),
            }
        )

    return pd.DataFrame(rows)


def _annualized_ic_ir(
    values: np.ndarray,
) -> float:
    """Calculate annualized information-coefficient information ratio."""
    finite = values[np.isfinite(values)]

    if len(finite) < 2:
        return float("nan")

    standard_deviation = float(
        np.std(
            finite,
            ddof=1,
        )
    )

    if standard_deviation <= 0.0:
        return float("nan")

    return float(np.mean(finite) / standard_deviation * np.sqrt(12.0))


def _yearly_metrics(
    monthly: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize final-signal ranking quality by calendar year."""
    data = monthly.copy()

    data["year"] = data["as_of_date"].dt.year

    rows = []

    for year, group in data.groupby(
        "year",
        sort=True,
    ):
        ic_values = group["information_coefficient"].to_numpy(dtype=float)

        spread_values = group["top_bottom_spread"].to_numpy(dtype=float)

        rows.append(
            {
                "year": int(year),
                "months": int(len(group)),
                "mean_ic": float(np.nanmean(ic_values)),
                "median_ic": float(np.nanmedian(ic_values)),
                "annualized_ic_ir": (_annualized_ic_ir(ic_values)),
                "positive_ic_ratio": float(np.mean(ic_values > 0.0)),
                "mean_top_bottom_spread": float(np.mean(spread_values)),
                "median_top_bottom_spread": float(np.median(spread_values)),
                "positive_spread_ratio": float(np.mean(spread_values > 0.0)),
                "mean_top_quintile_precision": float(group["top_quintile_precision"].mean()),
            }
        )

    return pd.DataFrame(rows)


def _sector_metrics(
    research: pd.DataFrame,
) -> pd.DataFrame:
    """Measure whether final-signal IC depends on individual sectors."""
    monthly_rows = []

    for (
        as_of_date,
        sector,
    ), group in research.groupby(
        [
            "as_of_date",
            "sector",
        ],
        sort=True,
    ):
        if len(group) < MINIMUM_SECTOR_COMPANIES:
            continue

        ic = _spearman(
            group["percentile_score"],
            group[TARGET_COLUMN],
        )

        if not np.isfinite(ic):
            continue

        monthly_rows.append(
            {
                "as_of_date": pd.Timestamp(as_of_date),
                "sector": str(sector),
                "companies": int(len(group)),
                "information_coefficient": float(ic),
            }
        )

    sector_monthly = pd.DataFrame(monthly_rows)

    if sector_monthly.empty:
        raise ValueError("No sector-level IC observations could be calculated.")

    rows = []

    for sector, group in sector_monthly.groupby(
        "sector",
        sort=True,
    ):
        values = group["information_coefficient"].to_numpy(dtype=float)

        rows.append(
            {
                "sector": str(sector),
                "valid_months": int(len(group)),
                "mean_companies_per_month": float(group["companies"].mean()),
                "mean_ic": float(np.mean(values)),
                "median_ic": float(np.median(values)),
                "std_ic": float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )
                if len(values) > 1
                else 0.0,
                "annualized_ic_ir": (_annualized_ic_ir(values)),
                "positive_ic_ratio": float(np.mean(values > 0.0)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "mean_ic",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def _circular_block_bootstrap_indices(
    *,
    observations: int,
    replications: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    """Generate circular block-bootstrap indices."""
    if observations < block_length:
        raise ValueError("Block length cannot exceed the number of observations.")

    rng = np.random.default_rng(seed)

    blocks_needed = int(np.ceil(observations / block_length))

    starts = rng.integers(
        0,
        observations,
        size=(
            replications,
            blocks_needed,
        ),
    )

    offsets = np.arange(
        block_length,
        dtype=int,
    )

    indices = (
        starts[
            :,
            :,
            None,
        ]
        + offsets[
            None,
            None,
            :,
        ]
    ) % observations

    indices = indices.reshape(
        replications,
        -1,
    )

    return indices[
        :,
        :observations,
    ]


def _confidence_interval(
    values: np.ndarray,
) -> tuple[
    float,
    float,
]:
    """Return an equal-tailed confidence interval."""
    alpha = 1.0 - CONFIDENCE_LEVEL

    return (
        float(
            np.quantile(
                values,
                alpha / 2.0,
            )
        ),
        float(
            np.quantile(
                values,
                1.0 - alpha / 2.0,
            )
        ),
    )


def _bootstrap_summary(
    monthly: pd.DataFrame,
) -> pd.DataFrame:
    """Bootstrap mean IC, spread and ranking precision by month."""
    data = monthly.loc[
        monthly[
            [
                "information_coefficient",
                "top_bottom_spread",
                "top_quintile_precision",
            ]
        ]
        .notna()
        .all(axis=1)
    ].copy()

    if len(data) < MINIMUM_MONTHS:
        raise ValueError("Too few valid monthly observations for final-signal bootstrap.")

    ic = data["information_coefficient"].to_numpy(dtype=float)

    spread = data["top_bottom_spread"].to_numpy(dtype=float)

    precision = data["top_quintile_precision"].to_numpy(dtype=float)

    indices = _circular_block_bootstrap_indices(
        observations=len(data),
        replications=(BOOTSTRAP_REPLICATIONS),
        block_length=(BOOTSTRAP_BLOCK_LENGTH),
        seed=BOOTSTRAP_SEED,
    )

    sampled_ic = ic[indices]

    sampled_spread = spread[indices]

    sampled_precision = precision[indices]

    mean_ic_draws = np.mean(
        sampled_ic,
        axis=1,
    )

    spread_draws = np.mean(
        sampled_spread,
        axis=1,
    )

    precision_draws = np.mean(
        sampled_precision,
        axis=1,
    )

    ic_ir_draws = (
        np.mean(
            sampled_ic,
            axis=1,
        )
        / np.std(
            sampled_ic,
            axis=1,
            ddof=1,
        )
        * np.sqrt(12.0)
    )

    if not np.isfinite(ic_ir_draws).all():
        raise ValueError("Bootstrap produced non-finite IC information ratios.")

    mean_ic_ci = _confidence_interval(mean_ic_draws)

    ic_ir_ci = _confidence_interval(ic_ir_draws)

    spread_ci = _confidence_interval(spread_draws)

    precision_ci = _confidence_interval(precision_draws)

    observed_mean_ic = float(np.mean(ic))

    observed_spread = float(np.mean(spread))

    observed_precision = float(np.mean(precision))

    return pd.DataFrame(
        [
            {
                "months": int(len(data)),
                "bootstrap_replications": (BOOTSTRAP_REPLICATIONS),
                "block_length_months": (BOOTSTRAP_BLOCK_LENGTH),
                "confidence_level": (CONFIDENCE_LEVEL),
                "observed_mean_ic": (observed_mean_ic),
                "mean_ic_ci_lower": (mean_ic_ci[0]),
                "mean_ic_ci_upper": (mean_ic_ci[1]),
                "probability_mean_ic_positive": float(np.mean(mean_ic_draws > 0.0)),
                "observed_annualized_ic_ir": (_annualized_ic_ir(ic)),
                "annualized_ic_ir_ci_lower": (ic_ir_ci[0]),
                "annualized_ic_ir_ci_upper": (ic_ir_ci[1]),
                "observed_positive_ic_ratio": float(np.mean(ic > 0.0)),
                "observed_mean_top_bottom_spread": (observed_spread),
                "top_bottom_spread_ci_lower": (spread_ci[0]),
                "top_bottom_spread_ci_upper": (spread_ci[1]),
                "probability_mean_spread_positive": float(np.mean(spread_draws > 0.0)),
                "observed_positive_spread_ratio": float(np.mean(spread > 0.0)),
                "observed_mean_top_quintile_precision": (observed_precision),
                "top_quintile_precision_ci_lower": (precision_ci[0]),
                "top_quintile_precision_ci_upper": (precision_ci[1]),
                "probability_both_ic_and_spread_positive": float(
                    np.mean((mean_ic_draws > 0.0) & (spread_draws > 0.0))
                ),
            }
        ]
    )


def _build_checks(
    research: pd.DataFrame,
    monthly: pd.DataFrame,
    bootstrap: pd.DataFrame,
    yearly: pd.DataFrame,
    sectors: pd.DataFrame,
) -> pd.DataFrame:
    """Audit final-signal statistical robustness outputs."""
    monthly_finite = (
        monthly[
            [
                "information_coefficient",
                "top_bottom_spread",
                "top_quintile_precision",
            ]
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .notna()
        .all(axis=1)
    )

    probability_columns = [
        "probability_mean_ic_positive",
        "probability_mean_spread_positive",
        "probability_both_ic_and_spread_positive",
    ]

    probabilities_valid = all(
        bootstrap.loc[
            0,
            column,
        ]
        >= 0.0
        and bootstrap.loc[
            0,
            column,
        ]
        <= 1.0
        for column in probability_columns
    )

    interval_order = (
        bootstrap.loc[
            0,
            "mean_ic_ci_lower",
        ]
        <= bootstrap.loc[
            0,
            "mean_ic_ci_upper",
        ]
        and bootstrap.loc[
            0,
            "annualized_ic_ir_ci_lower",
        ]
        <= bootstrap.loc[
            0,
            "annualized_ic_ir_ci_upper",
        ]
        and bootstrap.loc[
            0,
            "top_bottom_spread_ci_lower",
        ]
        <= bootstrap.loc[
            0,
            "top_bottom_spread_ci_upper",
        ]
    )

    usable_sectors = int(sectors["valid_months"].ge(MINIMUM_SECTOR_MONTHS).sum())

    checks = [
        (
            "research_rows_match_final_signal",
            int(len(research) != 3850),
            (
                "The current frozen OOS final-signal research panel "
                "should contain 77 dates × 50 securities."
            ),
        ),
        (
            "expected_oos_months",
            int(monthly["as_of_date"].nunique() != 77),
            "The current frozen OOS evaluation should contain 77 monthly dates.",
        ),
        (
            "minimum_valid_months",
            int(monthly_finite.sum() < MINIMUM_MONTHS),
            "Final-signal statistical robustness requires at least 60 valid months.",
        ),
        (
            "finite_monthly_metrics",
            int((~monthly_finite).sum()),
            "IC, spread and precision must be finite for every evaluated month.",
        ),
        (
            "valid_bootstrap_probabilities",
            int(not probabilities_valid),
            "Bootstrap probabilities must lie between zero and one.",
        ),
        (
            "bootstrap_interval_order",
            int(not interval_order),
            "Bootstrap confidence-interval lower bounds must not exceed upper bounds.",
        ),
        (
            "yearly_coverage",
            int(yearly["year"].nunique() != 7),
            "The current OOS sample should cover calendar years 2020 through 2026.",
        ),
        (
            "sector_coverage",
            int(usable_sectors < 5),
            (
                "At least five sectors should have enough monthly "
                "cross-sectional observations for sector IC diagnostics."
            ),
        ),
    ]

    return pd.DataFrame(
        [
            {
                "check": name,
                "status": ("PASS" if violations == 0 else "FAIL"),
                "violations": int(violations),
                "description": description,
            }
            for (
                name,
                violations,
                description,
            ) in checks
        ]
    )


def _format_value(
    value: Any,
) -> str:
    """Format values for Markdown output."""
    if value is None or pd.isna(value):
        return ""

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.date().isoformat()

    if isinstance(
        value,
        float,
    ):
        return f"{value:.6f}"

    return str(value).replace(
        "|",
        "\\|",
    )


def _to_markdown(
    data: pd.DataFrame,
) -> str:
    """Convert a dataframe to a simple Markdown table."""
    if data.empty:
        return "_No observations._"

    columns = [str(column) for column in data.columns]

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in data.itertuples(
        index=False,
        name=None,
    ):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")

    return "\n".join(lines)


def _build_report(
    bootstrap: pd.DataFrame,
    yearly: pd.DataFrame,
    sectors: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    """Build the final-signal statistical robustness report."""
    sector_display = sectors.loc[
        sectors["valid_months"].ge(MINIMUM_SECTOR_MONTHS),
        [
            "sector",
            "valid_months",
            "mean_companies_per_month",
            "mean_ic",
            "annualized_ic_ir",
            "positive_ic_ratio",
        ],
    ]

    return "\n".join(
        [
            "# Final Signal Statistical Robustness",
            "",
            "## Methodology",
            "",
            (
                "- The production-safe final alpha signal is joined back "
                "to realized 21-session OOS excess returns only inside "
                "this research diagnostic."
            ),
            (
                "- Monthly IC is the Spearman cross-sectional correlation "
                "between final percentile score and realized target."
            ),
            (
                "- Top-bottom spread is the mean realized target of the "
                "predicted top quintile minus the predicted bottom quintile."
            ),
            (
                f"- Confidence intervals use `{BOOTSTRAP_REPLICATIONS:,}` "
                f"circular block-bootstrap replications with "
                f"`{BOOTSTRAP_BLOCK_LENGTH}`-month blocks."
            ),
            (
                "- Sector IC is diagnostic because individual sector "
                "cross-sections are much smaller than the full universe."
            ),
            "",
            "## Bootstrap summary",
            "",
            _to_markdown(bootstrap),
            "",
            "## Calendar-year stability",
            "",
            _to_markdown(yearly),
            "",
            "## Sector stability",
            "",
            _to_markdown(sector_display),
            "",
            "## Readiness checks",
            "",
            _to_markdown(checks),
            "",
        ]
    )


def main() -> None:
    """Run final-signal statistical robustness diagnostics."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    required_paths = (
        FINAL_SIGNAL_PATH,
        MODELING_PANEL_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    final_signal = pd.read_parquet(FINAL_SIGNAL_PATH)

    modeling_panel = pd.read_parquet(MODELING_PANEL_PATH)

    logger.info("Building final-signal OOS research panel.")

    research = _prepare_research_panel(
        final_signal,
        modeling_panel,
    )

    logger.info("Calculating monthly IC and spread stability.")

    monthly = _monthly_metrics(research)

    yearly = _yearly_metrics(monthly)

    logger.info("Running circular block bootstrap.")

    bootstrap = _bootstrap_summary(monthly)

    logger.info("Calculating sector-level signal stability.")

    sectors = _sector_metrics(research)

    checks = _build_checks(
        research,
        monthly,
        bootstrap,
        yearly,
        sectors,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    _write_csv(
        monthly,
        MONTHLY_PATH,
    )

    _write_csv(
        bootstrap,
        BOOTSTRAP_PATH,
    )

    _write_csv(
        yearly,
        YEARLY_PATH,
    )

    _write_csv(
        sectors,
        SECTOR_PATH,
    )

    _write_csv(
        checks,
        CHECKS_PATH,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        _build_report(
            bootstrap,
            yearly,
            sectors,
            checks,
        ),
        encoding="utf-8",
    )

    if failed_checks:
        raise ValueError(
            "Final-signal statistical robustness validation "
            f"failed with {failed_checks} failed checks."
        )

    logger.info("Final-signal statistical robustness analysis completed.")

    print()

    print("Institutional Quant Equity Research Platform")

    print("Final signal statistical robustness")

    print("------------------------------------------------")

    print(f"rows: {len(research)}")

    print(f"months: {monthly['as_of_date'].nunique()}")

    print(f"years: {yearly['year'].nunique()}")

    print(f"sectors: {sectors['sector'].nunique()}")

    print(f"bootstrap_replications: {BOOTSTRAP_REPLICATIONS}")

    print()

    print("Bootstrap summary:")

    print(bootstrap.to_string(index=False))

    print()

    print("Yearly signal stability:")

    print(yearly.to_string(index=False))

    print()

    print("Sector signal stability:")

    print(sectors.to_string(index=False))

    print()

    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()

    print(f"Monthly table: {MONTHLY_PATH}")

    print(f"Bootstrap table: {BOOTSTRAP_PATH}")

    print(f"Yearly table: {YEARLY_PATH}")

    print(f"Sector table: {SECTOR_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
