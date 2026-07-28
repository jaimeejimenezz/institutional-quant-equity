"""Monthly rebalance calendars and forward-return labels."""

from __future__ import annotations

import math

import pandas as pd

REQUIRED_MARKET_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "adjusted_close",
)

REBALANCE_CALENDAR_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "rebalance_month",
    "first_future_date",
    "target_end_date",
    "horizon_sessions",
    "has_full_horizon",
)

MONTHLY_LABEL_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "ticker",
    "first_future_date",
    "target_end_date",
    "horizon_sessions",
    "as_of_adjusted_close",
    "target_end_adjusted_close",
    "target_21d",
    "cross_sectional_median_21d",
    "target_21d_excess",
    "target_rank",
    "target_percentile",
    "cross_section_size",
    "label_top_quintile",
)


class MonthlyLabelError(ValueError):
    """Raised when monthly labels cannot be constructed safely."""


def _prepare_market_data(
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and standardize market data required by labels.

    Missing or non-positive adjusted prices are retained at this
    stage so they do not remove valid market sessions from the
    rebalance calendar. They are filtered later when labels are
    constructed.
    """
    missing_columns = sorted(set(REQUIRED_MARKET_COLUMNS).difference(market_data.columns))

    if missing_columns:
        missing = ", ".join(missing_columns)

        raise MonthlyLabelError(f"Missing required market columns: {missing}")

    if market_data.empty:
        raise MonthlyLabelError("The market dataset is empty.")

    data = market_data.loc[
        :,
        REQUIRED_MARKET_COLUMNS,
    ].copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    ).dt.normalize()

    data["ticker"] = data["ticker"].astype("string").str.strip().str.upper()

    data["adjusted_close"] = pd.to_numeric(
        data["adjusted_close"],
        errors="coerce",
    )

    invalid_date_rows = int(data["date"].isna().sum())

    if invalid_date_rows:
        raise MonthlyLabelError(f"{invalid_date_rows} market rows contain invalid dates.")

    invalid_ticker_rows = int(data["ticker"].isna().sum()) + int(data["ticker"].eq("").sum())

    if invalid_ticker_rows:
        raise MonthlyLabelError(f"{invalid_ticker_rows} market rows contain invalid tickers.")

    duplicate_rows = int(
        data.duplicated(
            subset=["date", "ticker"],
            keep=False,
        ).sum()
    )

    if duplicate_rows:
        raise MonthlyLabelError(f"{duplicate_rows} duplicated date-ticker rows were found.")

    return data.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_rebalance_calendar(
    market_data: pd.DataFrame,
    *,
    horizon_sessions: int = 21,
) -> pd.DataFrame:
    """Build the month-end rebalance calendar.

    The as-of date is the final observed market session
    in each calendar month.

    The target end date is exactly ``horizon_sessions``
    market sessions after the as-of date.
    """
    if horizon_sessions < 1:
        raise MonthlyLabelError("horizon_sessions must be at least 1.")

    data = _prepare_market_data(market_data)

    session_dates = pd.DatetimeIndex(
        data["date"].drop_duplicates().sort_values(),
        name="date",
    )

    session_frame = pd.DataFrame(
        {
            "date": session_dates,
        }
    )

    session_frame["rebalance_month"] = session_frame["date"].dt.to_period("M").astype("string")

    calendar = (
        session_frame.groupby(
            "rebalance_month",
            as_index=False,
            sort=True,
        )
        .agg(
            as_of_date=(
                "date",
                "max",
            )
        )
        .sort_values("as_of_date")
        .reset_index(drop=True)
    )

    session_position = pd.Series(
        range(len(session_dates)),
        index=session_dates,
        dtype="int64",
    )

    as_of_positions = calendar["as_of_date"].map(session_position).astype("int64")

    first_future_dates: list[pd.Timestamp | pd.NaT] = []
    target_end_dates: list[pd.Timestamp | pd.NaT] = []

    for position in as_of_positions:
        first_future_position = position + 1

        target_end_position = position + horizon_sessions

        if first_future_position < len(session_dates):
            first_future_dates.append(session_dates[first_future_position])
        else:
            first_future_dates.append(pd.NaT)

        if target_end_position < len(session_dates):
            target_end_dates.append(session_dates[target_end_position])
        else:
            target_end_dates.append(pd.NaT)

    calendar["first_future_date"] = pd.to_datetime(first_future_dates)

    calendar["target_end_date"] = pd.to_datetime(target_end_dates)

    calendar["horizon_sessions"] = horizon_sessions

    calendar["has_full_horizon"] = calendar["target_end_date"].notna()

    return calendar.loc[
        :,
        REBALANCE_CALENDAR_COLUMNS,
    ]


def _validate_rebalance_calendar(
    rebalance_calendar: pd.DataFrame,
    *,
    horizon_sessions: int,
) -> pd.DataFrame:
    """Validate a calendar supplied to the label builder."""
    required_columns = {
        "as_of_date",
        "first_future_date",
        "target_end_date",
        "horizon_sessions",
        "has_full_horizon",
    }

    missing_columns = sorted(required_columns.difference(rebalance_calendar.columns))

    if missing_columns:
        missing = ", ".join(missing_columns)

        raise MonthlyLabelError(f"Missing rebalance calendar columns: {missing}")

    calendar = rebalance_calendar.copy()

    for column in (
        "as_of_date",
        "first_future_date",
        "target_end_date",
    ):
        calendar[column] = pd.to_datetime(
            calendar[column],
            errors="coerce",
        ).dt.normalize()

    if calendar["as_of_date"].isna().any():
        raise MonthlyLabelError("The rebalance calendar contains invalid as-of dates.")

    calendar["horizon_sessions"] = pd.to_numeric(
        calendar["horizon_sessions"],
        errors="coerce",
    )

    calendar["has_full_horizon"] = calendar["has_full_horizon"].astype(bool)

    duplicate_dates = int(calendar["as_of_date"].duplicated(keep=False).sum())

    if duplicate_dates:
        raise MonthlyLabelError(
            f"{duplicate_dates} duplicated as-of dates were found in the calendar."
        )

    horizon_mismatch = calendar["horizon_sessions"].ne(horizon_sessions).any()

    if horizon_mismatch:
        raise MonthlyLabelError(
            "The rebalance calendar horizon does not match the requested horizon."
        )

    eligible_calendar = calendar.loc[calendar["has_full_horizon"]]

    missing_future_dates = (
        eligible_calendar[
            [
                "first_future_date",
                "target_end_date",
            ]
        ]
        .isna()
        .any()
        .any()
    )

    if missing_future_dates:
        raise MonthlyLabelError("An eligible calendar row contains missing future dates.")

    invalid_temporal_order = (
        eligible_calendar["first_future_date"].le(eligible_calendar["as_of_date"])
        | eligible_calendar["target_end_date"].lt(eligible_calendar["first_future_date"])
    ).any()

    if invalid_temporal_order:
        raise MonthlyLabelError("The rebalance calendar contains invalid future-date ordering.")

    return calendar.sort_values("as_of_date").reset_index(drop=True)


def build_forward_return_labels(
    market_data: pd.DataFrame,
    *,
    horizon_sessions: int = 21,
    relative_to: str | None = ("cross_sectional_median"),
    top_quantile_fraction: float = 0.20,
    rebalance_calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build point-in-time monthly forward-return labels.

    ``target_21d`` is the compounded close-to-close
    return from the as-of adjusted close through the
    adjusted close exactly 21 market sessions later.

    The first realized daily return belongs to the
    first market session strictly after ``as_of_date``.
    """
    if horizon_sessions < 1:
        raise MonthlyLabelError("horizon_sessions must be at least 1.")

    if not 0 < top_quantile_fraction <= 1:
        raise MonthlyLabelError("top_quantile_fraction must be greater than 0 and at most 1.")

    if relative_to not in {
        None,
        "cross_sectional_median",
    }:
        raise MonthlyLabelError("relative_to must be None or 'cross_sectional_median'.")

    data = _prepare_market_data(market_data)

    valid_price_mask = data["adjusted_close"].notna() & data["adjusted_close"].gt(0)

    price_data = data.loc[valid_price_mask].copy().reset_index(drop=True)

    if price_data.empty:
        raise MonthlyLabelError(
            "No valid positive adjusted prices are available for label construction."
        )

    if rebalance_calendar is None:
        calendar = build_rebalance_calendar(
            data,
            horizon_sessions=(horizon_sessions),
        )
    else:
        calendar = _validate_rebalance_calendar(
            rebalance_calendar,
            horizon_sessions=(horizon_sessions),
        )

    eligible_calendar = calendar.loc[
        calendar["has_full_horizon"],
        [
            "as_of_date",
            "first_future_date",
            "target_end_date",
            "horizon_sessions",
        ],
    ].copy()

    if eligible_calendar.empty:
        return pd.DataFrame(columns=MONTHLY_LABEL_COLUMNS)

    as_of_prices = price_data.rename(
        columns={
            "date": "as_of_date",
            "adjusted_close": ("as_of_adjusted_close"),
        }
    )

    target_prices = price_data.rename(
        columns={
            "date": "target_end_date",
            "adjusted_close": ("target_end_adjusted_close"),
        }
    )

    labels = eligible_calendar.merge(
        as_of_prices,
        on="as_of_date",
        how="inner",
        validate="one_to_many",
    )

    labels = labels.merge(
        target_prices,
        on=[
            "target_end_date",
            "ticker",
        ],
        how="inner",
        validate="one_to_one",
    )

    if labels.empty:
        return pd.DataFrame(columns=MONTHLY_LABEL_COLUMNS)

    labels["target_21d"] = (
        labels["target_end_adjusted_close"] / labels["as_of_adjusted_close"] - 1.0
    )

    labels["cross_sectional_median_21d"] = labels.groupby(
        "as_of_date",
        sort=False,
    )["target_21d"].transform("median")

    if relative_to == "cross_sectional_median":
        labels["target_21d_excess"] = labels["target_21d"] - labels["cross_sectional_median_21d"]
    else:
        labels["target_21d_excess"] = labels["target_21d"]

    labels = labels.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)

    labels["cross_section_size"] = labels.groupby(
        "as_of_date",
        sort=False,
    )["ticker"].transform("size")

    labels["target_rank"] = (
        labels.groupby(
            "as_of_date",
            sort=False,
        )["target_21d"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype("int64")
    )

    labels["target_percentile"] = (
        labels["cross_section_size"] - labels["target_rank"] + 1
    ) / labels["cross_section_size"]

    top_counts = labels["cross_section_size"].map(
        lambda size: max(
            1,
            math.ceil(size * top_quantile_fraction),
        )
    )

    labels["label_top_quintile"] = labels["target_rank"].le(top_counts).astype("int8")

    labels["horizon_sessions"] = labels["horizon_sessions"].astype("int16")

    labels["cross_section_size"] = labels["cross_section_size"].astype("int16")

    invalid_future_dates = (
        labels["first_future_date"].le(labels["as_of_date"])
        | labels["target_end_date"].lt(labels["first_future_date"])
    ).any()

    if invalid_future_dates:
        raise MonthlyLabelError("One or more labels use dates that are not strictly in the future.")

    return labels.loc[
        :,
        MONTHLY_LABEL_COLUMNS,
    ]
