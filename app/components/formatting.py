from __future__ import annotations

import math


def format_percent(value: float, *, digits: int = 1, signed: bool = False) -> str:
    if not math.isfinite(float(value)):
        return "—"
    sign = "+" if signed else ""
    return f"{float(value):{sign}.{digits}%}"


def format_ratio(value: float, *, digits: int = 2) -> str:
    if not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def format_currency(value: float) -> str:
    if not math.isfinite(float(value)):
        return "—"
    magnitude = abs(float(value))
    if magnitude >= 1_000_000:
        return f"${float(value) / 1_000_000:.2f}M"
    if magnitude >= 1_000:
        return f"${float(value) / 1_000:.1f}K"
    return f"${float(value):,.0f}"


def format_bps(value: float, *, digits: int = 2) -> str:
    if not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f} bps"
