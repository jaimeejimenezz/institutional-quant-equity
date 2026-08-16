"""Tests for purged walk-forward validation."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_equity.validation import (
    WalkForwardConfig,
    WalkForwardValidationError,
    build_walk_forward_folds,
    split_panel_by_fold,
)


def make_panel(
    *,
    periods: int = 20,
    tickers: tuple[str, ...] = (
        "AAA",
        "BBB",
    ),
) -> pd.DataFrame:
    """Create a synthetic monthly modeling panel."""
    dates = pd.date_range(
        "2014-01-31",
        periods=periods,
        freq="ME",
    )

    rows: list[dict[str, object]] = []

    for as_of_date in dates:
        for ticker in tickers:
            rows.append(
                {
                    "as_of_date": as_of_date,
                    "ticker": ticker,
                    "target_end_date": (as_of_date + pd.Timedelta(days=20)),
                    "has_target": 1,
                }
            )

    return pd.DataFrame(rows)


def test_builds_date_grouped_folds() -> None:
    """All securities from one month must remain together."""
    panel = make_panel()

    config = WalkForwardConfig(
        min_train_dates=5,
        validation_dates=2,
    )

    folds = build_walk_forward_folds(
        panel,
        config=config,
    )

    assert folds

    train, validation, test = split_panel_by_fold(
        panel,
        folds[0],
    )

    assert train.groupby("as_of_date")["ticker"].nunique().eq(2).all()

    assert validation.groupby("as_of_date")["ticker"].nunique().eq(2).all()

    assert test["ticker"].nunique() == 2


def test_first_fold_respects_window_sizes() -> None:
    """The first fold must have minimum train and validation history."""
    panel = make_panel()

    config = WalkForwardConfig(
        min_train_dates=5,
        validation_dates=2,
    )

    fold = build_walk_forward_folds(
        panel,
        config=config,
    )[0]

    assert len(fold.train_dates) == 5

    assert len(fold.validation_dates) == 2


def test_splits_do_not_overlap() -> None:
    """Train, validation and test dates must be disjoint."""
    panel = make_panel()

    config = WalkForwardConfig(
        min_train_dates=5,
        validation_dates=2,
    )

    folds = build_walk_forward_folds(
        panel,
        config=config,
    )

    for fold in folds:
        train_dates = set(fold.train_dates)

        validation_dates = set(fold.validation_dates)

        test_dates = {fold.test_date}

        assert train_dates.isdisjoint(validation_dates)

        assert train_dates.isdisjoint(test_dates)

        assert validation_dates.isdisjoint(test_dates)


def test_labels_must_be_known_before_test() -> None:
    """Train and validation labels must have matured by test date."""
    panel = make_panel()

    config = WalkForwardConfig(
        min_train_dates=5,
        validation_dates=2,
    )

    folds = build_walk_forward_folds(
        panel,
        config=config,
    )

    for fold in folds:
        train, validation, _ = split_panel_by_fold(
            panel,
            fold,
        )

        assert (train["target_end_date"] <= fold.test_date).all()

        assert (validation["target_end_date"] <= fold.test_date).all()


def test_unfinished_label_is_purged() -> None:
    """A prior month with an unfinished target must not enter fitting."""
    panel = make_panel()

    dates = sorted(panel["as_of_date"].unique())

    blocked_date = pd.Timestamp(dates[6])

    panel.loc[
        panel["as_of_date"].eq(blocked_date),
        "target_end_date",
    ] = pd.Timestamp(dates[9])

    config = WalkForwardConfig(
        min_train_dates=5,
        validation_dates=2,
    )

    folds = build_walk_forward_folds(
        panel,
        config=config,
    )

    relevant = [fold for fold in folds if (blocked_date < fold.test_date < pd.Timestamp(dates[9]))]

    for fold in relevant:
        assert blocked_date not in fold.train_dates

        assert blocked_date not in fold.validation_dates

        assert blocked_date in fold.purged_dates


def test_rolling_mode_limits_training_window() -> None:
    """Rolling mode must retain only the configured train history."""
    panel = make_panel(periods=30)

    config = WalkForwardConfig(
        min_train_dates=5,
        validation_dates=2,
        mode="rolling",
        rolling_train_dates=8,
    )

    folds = build_walk_forward_folds(
        panel,
        config=config,
    )

    assert folds

    assert all(len(fold.train_dates) <= 8 for fold in folds)

    assert any(len(fold.train_dates) == 8 for fold in folds)


def test_invalid_rolling_configuration_fails() -> None:
    """Rolling validation needs an explicit valid train window."""
    with pytest.raises(WalkForwardValidationError):
        WalkForwardConfig(
            min_train_dates=60,
            validation_dates=12,
            mode="rolling",
            rolling_train_dates=36,
        )
