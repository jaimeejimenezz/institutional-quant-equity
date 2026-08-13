"""Point-in-time reconstruction of SEC fundamental information."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd


class SECPointInTimeError(ValueError):
    """Raised when point-in-time fundamentals cannot be built."""


@dataclass(frozen=True)
class SECPointInTimeConfig:
    """Configuration for point-in-time reconstruction."""

    availability_lag_days: int = 1
    require_statement_type_match: bool = True
    exclude_other_duration: bool = True

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> SECPointInTimeConfig:
        """Build configuration from project YAML."""
        config = cls(
            availability_lag_days=int(
                values.get(
                    "availability_lag_days",
                    1,
                )
            ),
            require_statement_type_match=bool(
                values.get(
                    "require_statement_type_match",
                    True,
                )
            ),
            exclude_other_duration=bool(
                values.get(
                    "exclude_other_duration",
                    True,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate configuration."""
        if self.availability_lag_days < 0:
            raise SECPointInTimeError(
                "availability_lag_days cannot be negative."
            )


def _normalize_as_of_dates(
    values: Sequence[Any] | pd.Series,
) -> list[pd.Timestamp]:
    """Normalize and sort snapshot dates."""
    dates = pd.Series(
        values,
        dtype="object",
    )

    dates = pd.to_datetime(
        dates,
        errors="coerce",
    ).dt.normalize()

    if dates.isna().any():
        raise SECPointInTimeError(
            "Point-in-time dates contain invalid values."
        )

    unique_dates = (
        dates.drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not unique_dates:
        raise SECPointInTimeError(
            "At least one point-in-time date is required."
        )

    return [
        pd.Timestamp(date)
        for date in unique_dates
    ]


def _candidate_is_better(
    new: Mapping[str, Any],
    previous: Mapping[str, Any],
) -> bool:
    """Decide which filing best represents one reporting period."""
    new_filed = pd.Timestamp(
        new["filed_date"]
    )

    previous_filed = pd.Timestamp(
        previous["filed_date"]
    )

    if new_filed != previous_filed:
        return (
            new_filed
            > previous_filed
        )

    new_amendment = bool(
        new["is_amendment"]
    )

    previous_amendment = bool(
        previous["is_amendment"]
    )

    if new_amendment != previous_amendment:
        return new_amendment

    new_priority = int(
        new["concept_priority"]
    )

    previous_priority = int(
        previous["concept_priority"]
    )

    if new_priority != previous_priority:
        return (
            new_priority
            < previous_priority
        )

    new_accession = str(
        new.get(
            "accession_number",
            "",
        )
        or ""
    )

    previous_accession = str(
        previous.get(
            "accession_number",
            "",
        )
        or ""
    )

    return (
        new_accession
        > previous_accession
    )


def prepare_point_in_time_events(
    canonical: pd.DataFrame,
    *,
    config: SECPointInTimeConfig,
) -> pd.DataFrame:
    """Prepare canonical facts for point-in-time reconstruction."""
    required_columns = {
        "ticker",
        "cik",
        "entity_name",
        "canonical_metric",
        "statement_type",
        "concept",
        "concept_priority",
        "unit",
        "value",
        "start_date",
        "end_date",
        "filed_date",
        "form",
        "fiscal_year",
        "fiscal_period",
        "accession_number",
        "frame",
        "duration_class",
        "statement_type_match",
        "is_amendment",
    }

    missing = sorted(
        required_columns.difference(
            canonical.columns
        )
    )

    if missing:
        raise SECPointInTimeError(
            "Canonical SEC data are missing columns: "
            + ", ".join(missing)
            + "."
        )

    data = canonical.copy()

    for column in (
        "start_date",
        "end_date",
        "filed_date",
    ):
        data[column] = pd.to_datetime(
            data[column],
            errors="coerce",
        ).dt.normalize()

    if data[
        "end_date"
    ].isna().any():
        raise SECPointInTimeError(
            "Canonical facts contain invalid end dates."
        )

    if data[
        "filed_date"
    ].isna().any():
        raise SECPointInTimeError(
            "Canonical facts contain invalid filed dates."
        )

    if (
        config.require_statement_type_match
    ):
        data = data.loc[
            data[
                "statement_type_match"
            ].astype(bool)
        ].copy()

    if config.exclude_other_duration:
        data = data.loc[
            ~data[
                "duration_class"
            ].eq(
                "other_duration"
            )
        ].copy()

    data[
        "available_date"
    ] = (
        data[
            "filed_date"
        ]
        + pd.to_timedelta(
            config.availability_lag_days,
            unit="D",
        )
    )

    if data.empty:
        raise SECPointInTimeError(
            "No usable SEC facts remain "
            "after point-in-time filtering."
        )

    return (
        data.sort_values(
            [
                "ticker",
                "canonical_metric",
                "duration_class",
                "available_date",
                "end_date",
                "concept_priority",
                "filed_date",
                "accession_number",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


def build_point_in_time_snapshots(
    canonical: pd.DataFrame,
    as_of_dates: Sequence[Any] | pd.Series,
    *,
    config: SECPointInTimeConfig,
) -> pd.DataFrame:
    """Build latest-known SEC fundamentals for each snapshot date."""
    events = prepare_point_in_time_events(
        canonical,
        config=config,
    )

    dates = _normalize_as_of_dates(
        as_of_dates
    )

    snapshot_rows: list[
        dict[str, Any]
    ] = []

    group_columns = [
        "ticker",
        "canonical_metric",
        "duration_class",
    ]

    for (
        ticker,
        metric,
        duration_class,
    ), group in events.groupby(
        group_columns,
        sort=True,
    ):
        group = (
            group.sort_values(
                [
                    "available_date",
                    "end_date",
                    "concept_priority",
                    "filed_date",
                    "accession_number",
                ],
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

        event_records = (
            group.to_dict(
                orient="records"
            )
        )

        current_by_period: dict[
            pd.Timestamp,
            dict[str, Any],
        ] = {}

        event_index = 0

        for as_of_date in dates:
            while (
                event_index
                < len(
                    event_records
                )
                and pd.Timestamp(
                    event_records[
                        event_index
                    ][
                        "available_date"
                    ]
                )
                <= as_of_date
            ):
                event = event_records[
                    event_index
                ]

                period_end = pd.Timestamp(
                    event[
                        "end_date"
                    ]
                )

                previous = (
                    current_by_period.get(
                        period_end
                    )
                )

                if (
                    previous is None
                    or _candidate_is_better(
                        event,
                        previous,
                    )
                ):
                    current_by_period[
                        period_end
                    ] = event

                event_index += 1

            if not current_by_period:
                continue

            latest_period_end = max(
                current_by_period
            )

            selected = (
                current_by_period[
                    latest_period_end
                ]
            )

            snapshot_rows.append(
                {
                    "as_of_date": (
                        as_of_date
                    ),
                    "ticker": ticker,
                    "cik": selected[
                        "cik"
                    ],
                    "entity_name": selected[
                        "entity_name"
                    ],
                    "canonical_metric": (
                        metric
                    ),
                    "statement_type": (
                        selected[
                            "statement_type"
                        ]
                    ),
                    "duration_class": (
                        duration_class
                    ),
                    "value": selected[
                        "value"
                    ],
                    "unit": selected[
                        "unit"
                    ],
                    "start_date": selected[
                        "start_date"
                    ],
                    "end_date": selected[
                        "end_date"
                    ],
                    "filed_date": selected[
                        "filed_date"
                    ],
                    "available_date": (
                        selected[
                            "available_date"
                        ]
                    ),
                    "form": selected[
                        "form"
                    ],
                    "fiscal_year": selected[
                        "fiscal_year"
                    ],
                    "fiscal_period": (
                        selected[
                            "fiscal_period"
                        ]
                    ),
                    "concept": selected[
                        "concept"
                    ],
                    "concept_priority": (
                        selected[
                            "concept_priority"
                        ]
                    ),
                    "accession_number": (
                        selected[
                            "accession_number"
                        ]
                    ),
                    "frame": selected[
                        "frame"
                    ],
                    "is_amendment": (
                        selected[
                            "is_amendment"
                        ]
                    ),
                    "period_age_days": (
                        as_of_date
                        - latest_period_end
                    ).days,
                    "filing_age_days": (
                        as_of_date
                        - pd.Timestamp(
                            selected[
                                "filed_date"
                            ]
                        )
                    ).days,
                }
            )

    snapshots = pd.DataFrame(
        snapshot_rows
    )

    if snapshots.empty:
        raise SECPointInTimeError(
            "Point-in-time reconstruction "
            "produced no snapshots."
        )

    if (
        snapshots[
            "available_date"
        ]
        > snapshots[
            "as_of_date"
        ]
    ).any():
        raise SECPointInTimeError(
            "Point-in-time snapshots contain "
            "future information."
        )

    duplicates = snapshots.duplicated(
        [
            "as_of_date",
            "ticker",
            "canonical_metric",
            "duration_class",
        ]
    )

    if duplicates.any():
        raise SECPointInTimeError(
            "Point-in-time snapshots contain "
            "duplicate metric states."
        )

    return (
        snapshots.sort_values(
            [
                "as_of_date",
                "ticker",
                "canonical_metric",
                "duration_class",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def build_point_in_time_coverage(
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize point-in-time coverage."""
    if snapshots.empty:
        raise SECPointInTimeError(
            "Snapshots cannot be empty."
        )

    return (
        snapshots.groupby(
            [
                "canonical_metric",
                "duration_class",
            ],
            as_index=False,
        )
        .agg(
            snapshot_rows=(
                "value",
                "size",
            ),
            companies=(
                "ticker",
                "nunique",
            ),
            as_of_dates=(
                "as_of_date",
                "nunique",
            ),
            first_as_of_date=(
                "as_of_date",
                "min",
            ),
            last_as_of_date=(
                "as_of_date",
                "max",
            ),
            median_period_age_days=(
                "period_age_days",
                "median",
            ),
        )
        .sort_values(
            [
                "canonical_metric",
                "duration_class",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def get_fundamentals_as_of(
    snapshots: pd.DataFrame,
    *,
    ticker: str,
    as_of_date: Any,
) -> pd.DataFrame:
    """Return the reconstructed fundamental state for one company/date."""
    normalized_ticker = (
        str(ticker)
        .strip()
        .upper()
    )

    normalized_date = (
        pd.Timestamp(
            as_of_date
        ).normalize()
    )

    result = snapshots.loc[
        snapshots[
            "ticker"
        ].eq(
            normalized_ticker
        )
        & snapshots[
            "as_of_date"
        ].eq(
            normalized_date
        )
    ].copy()

    return (
        result.sort_values(
            [
                "canonical_metric",
                "duration_class",
            ]
        )
        .reset_index(
            drop=True
        )
    )