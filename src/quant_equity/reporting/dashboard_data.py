from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_equity.reporting.dashboard_catalog import get_source, source_path


def read_dashboard_source(source_id: str) -> pd.DataFrame:
    get_source(source_id)
    return read_table(source_path(source_id))


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dashboard source format: {path}")


def latest_date(frame: pd.DataFrame, column: str) -> pd.Timestamp:
    if column not in frame.columns:
        raise KeyError(f"Column {column!r} is not available.")

    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"Column {column!r} has no valid dates.")

    return pd.Timestamp(values.max())
