"""Tests for market-data storage and quality validation."""

from pathlib import Path

import pandas as pd
import pytest

from quant_equity.data import (
    get_raw_market_path,
    load_raw_market_snapshot,
    normalize_market_data,
    save_raw_market_snapshot,
    validate_market_data,
    write_market_data_report,
)


def make_provider_frame() -> pd.DataFrame:
    """Create a provider-shaped market-data table."""
    index = pd.DatetimeIndex(
        [
            "2024-01-02 00:00:00-05:00",
            "2024-01-03 00:00:00-05:00",
            "2024-01-04 00:00:00-05:00",
        ],
        name="Date",
    )

    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [101.0, 102.0, 103.0],
            "Adj Close": [101.0, 102.0, 103.0],
            "Volume": [1_000, 1_100, 1_200],
            "Dividends": [0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        },
        index=index,
    )


def make_valid_market_dataset() -> pd.DataFrame:
    """Create valid canonical data for two tickers."""
    frames: list[pd.DataFrame] = []

    for ticker, offset in [
        ("AAA", 0.0),
        ("BBB", 10.0),
    ]:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-04",
                    ]
                ),
                "ticker": ticker,
                "open": [
                    100.0 + offset,
                    101.0 + offset,
                    102.0 + offset,
                ],
                "high": [
                    102.0 + offset,
                    103.0 + offset,
                    104.0 + offset,
                ],
                "low": [
                    99.0 + offset,
                    100.0 + offset,
                    101.0 + offset,
                ],
                "close": [
                    101.0 + offset,
                    102.0 + offset,
                    103.0 + offset,
                ],
                "adjusted_close": [
                    101.0 + offset,
                    102.0 + offset,
                    103.0 + offset,
                ],
                "volume": [
                    1_000,
                    1_100,
                    1_200,
                ],
                "dividends": [
                    0.0,
                    0.0,
                    0.0,
                ],
                "stock_splits": [
                    0.0,
                    0.0,
                    0.0,
                ],
            }
        )

        frames.append(frame)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def run_validation(
    market_data: pd.DataFrame,
):
    """Run validation with small synthetic thresholds."""
    return validate_market_data(
        market_data,
        expected_tickers=["AAA", "BBB"],
        start_date="2024-01-01",
        end_date="2024-02-01",
        minimum_observations=3,
        minimum_coverage_ratio=1.0,
        extreme_return_threshold=0.35,
        maximum_missing_ratio=0.0,
        adjustment_factor_jump_threshold=0.05,
    )


def test_raw_snapshot_roundtrip(
    tmp_path: Path,
) -> None:
    """A provider response should survive raw Parquet storage."""
    path = tmp_path / "AAPL.parquet"

    save_raw_market_snapshot(
        make_provider_frame(),
        path,
    )

    loaded_data = load_raw_market_snapshot(path)

    normalized_data = normalize_market_data(
        loaded_data,
        ticker="AAPL",
    )

    assert path.exists()
    assert len(normalized_data) == 3
    assert normalized_data["ticker"].eq("AAPL").all()


def test_raw_snapshot_cannot_be_overwritten(
    tmp_path: Path,
) -> None:
    """Raw snapshots should be immutable."""
    path = tmp_path / "AAPL.parquet"

    save_raw_market_snapshot(
        make_provider_frame(),
        path,
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        save_raw_market_snapshot(
            make_provider_frame(),
            path,
        )


def test_raw_market_path_is_deterministic() -> None:
    """The same request should produce the same cache path."""
    first_path = get_raw_market_path(
        ticker="brk-b",
        start_date="2014-01-01",
        end_date="2026-07-28",
    )

    second_path = get_raw_market_path(
        ticker="BRK-B",
        start_date="2014-01-01",
        end_date="2026-07-28",
    )

    assert first_path == second_path
    assert first_path.name.startswith("BRK-B__2014-01-01")


def test_valid_market_dataset_passes() -> None:
    """A complete synthetic dataset should pass validation."""
    result = run_validation(make_valid_market_dataset())

    assert result.is_valid
    assert not result.issues
    assert result.summary["tickers"] == 2
    assert result.summary["duplicate_rows"] == 0


def test_duplicate_market_rows_fail() -> None:
    """Duplicate date-ticker rows should block validation."""
    market_data = make_valid_market_dataset()

    duplicated_data = pd.concat(
        [
            market_data,
            market_data.iloc[[0]],
        ],
        ignore_index=True,
    )

    result = run_validation(duplicated_data)

    assert not result.is_valid
    assert any("duplicated" in issue for issue in result.issues)


def test_extreme_return_is_reported_as_warning() -> None:
    """Large returns should be reviewed without always failing."""
    market_data = make_valid_market_dataset()

    target_mask = market_data["ticker"].eq("AAA") & market_data["date"].eq(
        pd.Timestamp("2024-01-04")
    )

    market_data.loc[
        target_mask,
        [
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
        ],
    ] = [
        199.0,
        201.0,
        198.0,
        200.0,
        200.0,
    ]

    result = run_validation(market_data)

    assert result.is_valid
    assert len(result.extreme_returns) == 1
    assert result.warnings


def test_quality_report_is_written(
    tmp_path: Path,
) -> None:
    """The quality report should contain its final status."""
    result = run_validation(make_valid_market_dataset())

    report_path = tmp_path / "market_data_report.md"

    write_market_data_report(
        result,
        report_path,
        provider_name="test-provider",
        requested_start_date="2024-01-01",
        requested_end_date="2024-02-01",
    )

    report_text = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "Market Data Quality Report" in report_text
    assert "PASS" in report_text
