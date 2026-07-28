"""Tests for monthly-label storage and validation."""

from pathlib import Path

import pandas as pd

from quant_equity.labels import (
    build_forward_return_labels,
    build_rebalance_calendar,
    validate_monthly_labels,
    write_monthly_labels,
    write_monthly_labels_report,
    write_rebalance_calendar,
)


def make_valid_monthly_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create a valid calendar and labels dataset."""
    dates = [
        "2024-01-31",
        "2024-02-01",
        "2024-02-29",
        "2024-03-01",
    ]

    rows: list[dict[str, object]] = []

    tickers = [
        "AAA",
        "BBB",
        "CCC",
        "DDD",
        "EEE",
    ]

    for ticker_position, ticker in enumerate(
        tickers,
        start=1,
    ):
        prices = [
            100.0,
            100.0 + ticker_position,
            110.0,
            110.0 + ticker_position,
        ]

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

    market_data = pd.DataFrame(rows)

    calendar = build_rebalance_calendar(
        market_data,
        horizon_sessions=1,
    )

    labels = build_forward_return_labels(
        market_data,
        horizon_sessions=1,
        top_quantile_fraction=0.20,
        rebalance_calendar=calendar,
    )

    return calendar, labels


def run_validation(
    calendar: pd.DataFrame,
    labels: pd.DataFrame,
):
    """Validate synthetic monthly data."""
    return validate_monthly_labels(
        calendar,
        labels,
        expected_tickers=[
            "AAA",
            "BBB",
            "CCC",
            "DDD",
            "EEE",
        ],
        horizon_sessions=1,
        top_quantile_fraction=0.20,
    )


def test_valid_monthly_labels_pass_validation() -> None:
    """A correct monthly dataset should pass."""
    calendar, labels = make_valid_monthly_data()

    result = run_validation(
        calendar,
        labels,
    )

    assert result.is_valid
    assert not result.issues
    assert result.summary["label_dates"] == 2
    assert result.summary["tickers"] == 5


def test_duplicate_monthly_label_fails_validation() -> None:
    """A duplicated date-ticker label should fail."""
    calendar, labels = make_valid_monthly_data()

    duplicated_labels = pd.concat(
        [
            labels,
            labels.iloc[[0]],
        ],
        ignore_index=True,
    )

    result = run_validation(
        calendar,
        duplicated_labels,
    )

    assert not result.is_valid

    assert any("duplicated date-ticker" in issue for issue in result.issues)


def test_incorrect_target_fails_validation() -> None:
    """A modified future return should be detected."""
    calendar, labels = make_valid_monthly_data()

    labels = labels.copy()

    labels.loc[
        labels.index[0],
        "target_21d",
    ] += 0.01

    result = run_validation(
        calendar,
        labels,
    )

    assert not result.is_valid

    assert any("incorrectly calculated returns" in issue for issue in result.issues)


def test_incorrect_top_count_fails_validation() -> None:
    """The selected top-quintile count should be audited."""
    calendar, labels = make_valid_monthly_data()

    labels = labels.copy()

    first_date = labels["as_of_date"].min()

    labels.loc[
        labels["as_of_date"].eq(first_date),
        "label_top_quintile",
    ] = 0

    result = run_validation(
        calendar,
        labels,
    )

    assert not result.is_valid

    assert any("incorrect top-quintile count" in issue for issue in result.issues)


def test_monthly_parquet_writers_sort_data(
    tmp_path: Path,
) -> None:
    """Stored monthly datasets should be sorted."""
    calendar, labels = make_valid_monthly_data()

    calendar_path = tmp_path / "calendar.parquet"

    labels_path = tmp_path / "labels.parquet"

    write_rebalance_calendar(
        calendar.iloc[::-1],
        calendar_path,
    )

    write_monthly_labels(
        labels.iloc[::-1],
        labels_path,
    )

    stored_calendar = pd.read_parquet(calendar_path)

    stored_labels = pd.read_parquet(labels_path)

    assert stored_calendar["as_of_date"].is_monotonic_increasing

    expected_order = stored_labels.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        stored_labels,
        expected_order,
    )


def test_monthly_report_is_written(
    tmp_path: Path,
) -> None:
    """The Markdown quality report should be created."""
    calendar, labels = make_valid_monthly_data()

    result = run_validation(
        calendar,
        labels,
    )

    report_path = tmp_path / "monthly_labels_report.md"

    written_path = write_monthly_labels_report(
        result,
        report_path,
        horizon_sessions=1,
        top_quantile_fraction=0.20,
    )

    report_text = written_path.read_text(encoding="utf-8")

    assert written_path.exists()
    assert "# Monthly Labels Quality Report" in report_text
    assert "Status: **PASS**" in report_text
