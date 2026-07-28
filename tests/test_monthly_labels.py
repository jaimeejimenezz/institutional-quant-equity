"""Tests for monthly rebalance dates and forward labels."""

import pandas as pd
import pytest

from quant_equity.labels import (
    MonthlyLabelError,
    build_forward_return_labels,
    build_rebalance_calendar,
)


def make_market_data(
    dates: list[str],
    ticker_prices: dict[
        str,
        list[float],
    ],
) -> pd.DataFrame:
    """Create synthetic adjusted market data."""
    rows: list[dict[str, object]] = []

    for ticker, prices in ticker_prices.items():
        for date, price in zip(
            dates,
            prices,
            strict=True,
        ):
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "ticker": ticker,
                    "adjusted_close": price,
                }
            )

    return pd.DataFrame(rows)


def test_rebalance_calendar_uses_last_session_of_month() -> None:
    """Each rebalance date should be the last observed monthly session."""
    market_data = make_market_data(
        [
            "2024-01-30",
            "2024-01-31",
            "2024-02-01",
            "2024-02-28",
            "2024-02-29",
            "2024-03-01",
            "2024-03-04",
        ],
        {
            "AAA": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
            ],
        },
    )

    calendar = build_rebalance_calendar(
        market_data,
        horizon_sessions=2,
    )

    assert calendar["as_of_date"].tolist() == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-29"),
        pd.Timestamp("2024-03-04"),
    ]

    assert calendar.loc[
        0,
        "first_future_date",
    ] == pd.Timestamp("2024-02-01")

    assert calendar.loc[
        0,
        "target_end_date",
    ] == pd.Timestamp("2024-02-28")

    assert bool(
        calendar.loc[
            0,
            "has_full_horizon",
        ]
    )

    assert not bool(
        calendar.loc[
            2,
            "has_full_horizon",
        ]
    )


def test_forward_return_starts_after_as_of_date() -> None:
    """The first target session must be strictly after the signal."""
    market_data = make_market_data(
        [
            "2024-01-29",
            "2024-01-30",
            "2024-01-31",
            "2024-02-01",
            "2024-02-02",
        ],
        {
            "AAA": [
                90.0,
                95.0,
                100.0,
                110.0,
                121.0,
            ],
        },
    )

    labels = build_forward_return_labels(
        market_data,
        horizon_sessions=2,
    )

    row = labels.iloc[0]

    assert row["as_of_date"] == pd.Timestamp("2024-01-31")

    assert row["first_future_date"] == pd.Timestamp("2024-02-01")

    assert row["target_end_date"] == pd.Timestamp("2024-02-02")

    assert row["target_21d"] == pytest.approx(0.21)

    assert row["first_future_date"] > row["as_of_date"]


def test_relative_target_uses_cross_sectional_median() -> None:
    """Relative returns should subtract the monthly median."""
    market_data = make_market_data(
        [
            "2024-01-30",
            "2024-01-31",
            "2024-02-01",
        ],
        {
            "AAA": [
                100.0,
                100.0,
                110.0,
            ],
            "BBB": [
                100.0,
                100.0,
                120.0,
            ],
            "CCC": [
                100.0,
                100.0,
                130.0,
            ],
        },
    )

    labels = build_forward_return_labels(
        market_data,
        horizon_sessions=1,
    ).set_index("ticker")

    assert labels.loc[
        "AAA",
        "target_21d",
    ] == pytest.approx(0.10)

    assert labels.loc[
        "BBB",
        "target_21d",
    ] == pytest.approx(0.20)

    assert labels.loc[
        "CCC",
        "target_21d",
    ] == pytest.approx(0.30)

    assert labels.loc[
        "AAA",
        "target_21d_excess",
    ] == pytest.approx(-0.10)

    assert labels.loc[
        "BBB",
        "target_21d_excess",
    ] == pytest.approx(0.0)

    assert labels.loc[
        "CCC",
        "target_21d_excess",
    ] == pytest.approx(0.10)


def test_top_quintile_contains_twenty_percent() -> None:
    """Exactly two of ten securities should enter the top quintile."""
    ticker_prices = {
        f"T{index:02d}": [
            100.0,
            100.0 + index,
        ]
        for index in range(
            1,
            11,
        )
    }

    market_data = make_market_data(
        [
            "2024-01-31",
            "2024-02-01",
        ],
        ticker_prices,
    )

    labels = build_forward_return_labels(
        market_data,
        horizon_sessions=1,
        top_quantile_fraction=0.20,
    )

    selected = labels.loc[
        labels["label_top_quintile"].eq(1),
        "ticker",
    ].tolist()

    assert len(selected) == 2
    assert selected == [
        "T09",
        "T10",
    ]


def test_month_without_full_horizon_is_not_labeled() -> None:
    """The final incomplete month should remain outside the labels."""
    market_data = make_market_data(
        [
            "2024-01-31",
            "2024-02-01",
            "2024-02-29",
        ],
        {
            "AAA": [
                100.0,
                101.0,
                102.0,
            ],
        },
    )

    labels = build_forward_return_labels(
        market_data,
        horizon_sessions=1,
    )

    assert labels["as_of_date"].tolist() == [pd.Timestamp("2024-01-31")]


def test_missing_target_end_price_excludes_ticker() -> None:
    """An invalid exact target-end price must exclude the ticker."""
    market_data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-31",
                    "2024-02-01",
                    "2024-01-31",
                    "2024-02-01",
                ]
            ),
            "ticker": [
                "AAA",
                "AAA",
                "BBB",
                "BBB",
            ],
            "adjusted_close": [
                100.0,
                110.0,
                100.0,
                float("nan"),
            ],
        }
    )

    calendar = build_rebalance_calendar(
        market_data,
        horizon_sessions=1,
    )

    labels = build_forward_return_labels(
        market_data,
        horizon_sessions=1,
    )

    assert calendar.loc[
        0,
        "as_of_date",
    ] == pd.Timestamp("2024-01-31")

    assert labels["ticker"].tolist() == ["AAA"]


def test_duplicate_date_ticker_rows_raise() -> None:
    """Duplicate observations must block calendar construction."""
    market_data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-31",
                    "2024-01-31",
                ]
            ),
            "ticker": [
                "AAA",
                "AAA",
            ],
            "adjusted_close": [
                100.0,
                100.0,
            ],
        }
    )

    with pytest.raises(
        MonthlyLabelError,
        match="duplicated date-ticker",
    ):
        build_rebalance_calendar(market_data)


def test_invalid_horizon_raises() -> None:
    """A non-positive forecast horizon must be rejected."""
    market_data = make_market_data(
        [
            "2024-01-31",
            "2024-02-01",
        ],
        {
            "AAA": [
                100.0,
                101.0,
            ],
        },
    )

    with pytest.raises(
        MonthlyLabelError,
        match="at least 1",
    ):
        build_forward_return_labels(
            market_data,
            horizon_sessions=0,
        )
