"""Purged walk-forward validation for monthly equity-panel data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

WalkForwardMode = Literal["expanding", "rolling"]


class WalkForwardValidationError(ValueError):
    """Raised when walk-forward validation cannot be constructed."""


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for date-grouped walk-forward validation."""

    min_train_dates: int = 60
    validation_dates: int = 12
    mode: WalkForwardMode = "expanding"
    rolling_train_dates: int | None = None

    def __post_init__(self) -> None:
        """Validate the walk-forward configuration."""
        if self.min_train_dates <= 0:
            raise WalkForwardValidationError("min_train_dates must be positive.")

        if self.validation_dates <= 0:
            raise WalkForwardValidationError("validation_dates must be positive.")

        if self.mode not in (
            "expanding",
            "rolling",
        ):
            raise WalkForwardValidationError("mode must be 'expanding' or 'rolling'.")

        if self.mode == "rolling":
            if self.rolling_train_dates is None:
                raise WalkForwardValidationError(
                    "rolling_train_dates is required when mode='rolling'."
                )

            if self.rolling_train_dates < self.min_train_dates:
                raise WalkForwardValidationError(
                    "rolling_train_dates must be at least min_train_dates."
                )


@dataclass(frozen=True)
class WalkForwardFold:
    """One purged monthly walk-forward fold."""

    fold_id: int
    train_dates: tuple[pd.Timestamp, ...]
    validation_dates: tuple[pd.Timestamp, ...]
    test_date: pd.Timestamp
    purged_dates: tuple[pd.Timestamp, ...]

    @property
    def train_start_date(self) -> pd.Timestamp:
        """Return the first training date."""
        return self.train_dates[0]

    @property
    def train_end_date(self) -> pd.Timestamp:
        """Return the last training date."""
        return self.train_dates[-1]

    @property
    def validation_start_date(self) -> pd.Timestamp:
        """Return the first validation date."""
        return self.validation_dates[0]

    @property
    def validation_end_date(self) -> pd.Timestamp:
        """Return the last validation date."""
        return self.validation_dates[-1]


def _prepare_panel(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize the columns required for walk-forward."""
    required = {
        "as_of_date",
        "ticker",
        "target_end_date",
        "has_target",
    }

    missing = sorted(required.difference(panel.columns))

    if missing:
        raise WalkForwardValidationError(
            "Modeling panel is missing columns: " + ", ".join(missing) + "."
        )

    frame = panel.copy()

    frame["as_of_date"] = pd.to_datetime(
        frame["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    frame["target_end_date"] = pd.to_datetime(
        frame["target_end_date"],
        errors="coerce",
    ).dt.normalize()

    frame["has_target"] = pd.to_numeric(
        frame["has_target"],
        errors="coerce",
    )

    if frame["as_of_date"].isna().any():
        raise WalkForwardValidationError("as_of_date contains invalid or missing dates.")

    duplicated_keys = frame.duplicated(
        subset=[
            "as_of_date",
            "ticker",
        ],
        keep=False,
    )

    if duplicated_keys.any():
        raise WalkForwardValidationError(
            "Modeling panel contains duplicated (as_of_date, ticker) keys."
        )

    modeling = frame.loc[frame["has_target"].eq(1)]

    if modeling.empty:
        raise WalkForwardValidationError("No completed modeling observations were found.")

    if modeling["target_end_date"].isna().any():
        raise WalkForwardValidationError("Modeling rows must have target_end_date.")

    invalid_target_order = modeling["target_end_date"].le(modeling["as_of_date"])

    if invalid_target_order.any():
        raise WalkForwardValidationError("Every modeling target must end after as_of_date.")

    return frame


def _completed_label_dates(
    modeling: pd.DataFrame,
    *,
    test_date: pd.Timestamp,
) -> tuple[pd.Timestamp, ...]:
    """Return prior dates whose full labels were known by test_date."""
    previous = modeling.loc[modeling["as_of_date"].lt(test_date)]

    if previous.empty:
        return ()

    completion = previous.groupby(
        "as_of_date",
        sort=True,
    )["target_end_date"].max()

    eligible = completion.loc[completion.le(test_date)].index

    return tuple(pd.Timestamp(date) for date in eligible)


def build_walk_forward_folds(
    panel: pd.DataFrame,
    *,
    config: WalkForwardConfig | None = None,
) -> tuple[WalkForwardFold, ...]:
    """Build purged walk-forward folds grouped by as_of_date."""
    cfg = WalkForwardConfig() if config is None else config

    frame = _prepare_panel(panel)

    modeling = frame.loc[frame["has_target"].eq(1)].copy()

    modeling_dates = tuple(pd.Timestamp(date) for date in sorted(modeling["as_of_date"].unique()))

    required_history_dates = cfg.min_train_dates + cfg.validation_dates

    folds: list[WalkForwardFold] = []

    for test_date in modeling_dates:
        eligible_dates = _completed_label_dates(
            modeling,
            test_date=test_date,
        )

        if len(eligible_dates) < required_history_dates:
            continue

        validation_dates = eligible_dates[-cfg.validation_dates :]

        train_candidates = eligible_dates[: -cfg.validation_dates]

        if cfg.mode == "rolling":
            assert cfg.rolling_train_dates is not None

            train_dates = train_candidates[-cfg.rolling_train_dates :]
        else:
            train_dates = train_candidates

        if len(train_dates) < cfg.min_train_dates:
            continue

        prior_dates = tuple(date for date in modeling_dates if date < test_date)

        used_history_dates = set(train_dates).union(validation_dates)

        purged_dates = tuple(date for date in prior_dates if date not in used_history_dates)

        folds.append(
            WalkForwardFold(
                fold_id=len(folds) + 1,
                train_dates=tuple(train_dates),
                validation_dates=tuple(validation_dates),
                test_date=test_date,
                purged_dates=purged_dates,
            )
        )

    return tuple(folds)


def split_panel_by_fold(
    panel: pd.DataFrame,
    fold: WalkForwardFold,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Materialize train, validation and test rows for one fold."""
    frame = _prepare_panel(panel)

    train = frame.loc[frame["as_of_date"].isin(fold.train_dates)].copy()

    validation = frame.loc[frame["as_of_date"].isin(fold.validation_dates)].copy()

    test = frame.loc[frame["as_of_date"].eq(fold.test_date)].copy()

    return (
        train,
        validation,
        test,
    )


def walk_forward_folds_to_frame(
    panel: pd.DataFrame,
    folds: tuple[
        WalkForwardFold,
        ...,
    ],
    *,
    mode: WalkForwardMode,
) -> pd.DataFrame:
    """Convert walk-forward folds to a persistent metadata table."""
    frame = _prepare_panel(panel)

    rows: list[dict[str, object]] = []

    for fold in folds:
        train, validation, test = split_panel_by_fold(
            frame,
            fold,
        )

        train_sizes = train.groupby("as_of_date")["ticker"].nunique()

        validation_sizes = validation.groupby("as_of_date")["ticker"].nunique()

        test_sizes = test.groupby("as_of_date")["ticker"].nunique()

        max_train_target_end = train["target_end_date"].max()

        max_validation_target_end = validation["target_end_date"].max()

        rows.append(
            {
                "fold_id": fold.fold_id,
                "mode": mode,
                "train_start_date": (fold.train_start_date),
                "train_end_date": (fold.train_end_date),
                "validation_start_date": (fold.validation_start_date),
                "validation_end_date": (fold.validation_end_date),
                "test_date": fold.test_date,
                "train_date_count": len(fold.train_dates),
                "validation_date_count": len(fold.validation_dates),
                "test_date_count": 1,
                "purged_date_count": len(fold.purged_dates),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "train_cross_section_min": int(train_sizes.min()),
                "train_cross_section_max": int(train_sizes.max()),
                "validation_cross_section_min": int(validation_sizes.min()),
                "validation_cross_section_max": int(validation_sizes.max()),
                "test_cross_section_min": int(test_sizes.min()),
                "test_cross_section_max": int(test_sizes.max()),
                "max_train_target_end": (max_train_target_end),
                "max_validation_target_end": (max_validation_target_end),
            }
        )

    return pd.DataFrame(rows)
