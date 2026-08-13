"""Quarterly and trailing-twelve-month SEC fundamental reconstruction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_equity.data.sec_point_in_time import (
    SECPointInTimeConfig,
    prepare_point_in_time_events,
)


class SECQuarterlyReconstructionError(
    ValueError
):
    """Raised when quarterly fundamentals cannot be reconstructed."""


@dataclass(frozen=True)
class SECQuarterlyReconstructionConfig:
    """Configuration for quarterly and TTM reconstruction."""

    additive_metrics: tuple[str, ...]

    quarter_gap_min_days: int = 60
    quarter_gap_max_days: int = 120

    ttm_span_min_days: int = 240
    ttm_span_max_days: int = 330

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> SECQuarterlyReconstructionConfig:
        """Create configuration from YAML values."""
        metrics = tuple(
            str(value).strip()
            for value in values.get(
                "additive_metrics",
                [],
            )
        )

        config = cls(
            additive_metrics=metrics,
            quarter_gap_min_days=int(
                values.get(
                    "quarter_gap_min_days",
                    60,
                )
            ),
            quarter_gap_max_days=int(
                values.get(
                    "quarter_gap_max_days",
                    120,
                )
            ),
            ttm_span_min_days=int(
                values.get(
                    "ttm_span_min_days",
                    240,
                )
            ),
            ttm_span_max_days=int(
                values.get(
                    "ttm_span_max_days",
                    330,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate reconstruction settings."""
        if not self.additive_metrics:
            raise SECQuarterlyReconstructionError(
                "At least one additive metric is required."
            )

        if (
            self.quarter_gap_min_days < 1
            or self.quarter_gap_max_days
            < self.quarter_gap_min_days
        ):
            raise SECQuarterlyReconstructionError(
                "Invalid quarter gap limits."
            )

        if (
            self.ttm_span_min_days < 1
            or self.ttm_span_max_days
            < self.ttm_span_min_days
        ):
            raise SECQuarterlyReconstructionError(
                "Invalid TTM span limits."
            )


def _fact_is_better(
    new: Mapping[str, Any],
    previous: Mapping[str, Any],
) -> bool:
    """Select the best currently available representation."""
    new_filed = pd.Timestamp(
        new["filed_date"]
    )

    previous_filed = pd.Timestamp(
        previous["filed_date"]
    )

    if new_filed != previous_filed:
        return new_filed > previous_filed

    if bool(
        new["is_amendment"]
    ) != bool(
        previous["is_amendment"]
    ):
        return bool(
            new["is_amendment"]
        )

    new_priority = int(
        new["concept_priority"]
    )

    old_priority = int(
        previous["concept_priority"]
    )

    if new_priority != old_priority:
        return (
            new_priority
            < old_priority
        )

    return str(
        new.get(
            "accession_number",
            "",
        )
        or ""
    ) > str(
        previous.get(
            "accession_number",
            "",
        )
        or ""
    )


def _quarter_event(
    *,
    row: Mapping[str, Any],
    quarter_value: float,
    quarter_start: pd.Timestamp,
    quarter_end: pd.Timestamp,
    source_method: str,
    available_date: pd.Timestamp,
    components: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    """Create one reconstructed quarterly event."""
    accessions = sorted(
        {
            str(
                component.get(
                    "accession_number",
                    "",
                )
                or ""
            )
            for component in components
        }
    )

    concepts = sorted(
        {
            str(
                component.get(
                    "concept",
                    "",
                )
            )
            for component in components
        }
    )

    filed_dates = [
        pd.Timestamp(
            component[
                "filed_date"
            ]
        )
        for component in components
    ]

    return {
        "ticker": row["ticker"],
        "cik": row["cik"],
        "entity_name": (
            row["entity_name"]
        ),
        "canonical_metric": (
            row["canonical_metric"]
        ),
        "unit": row["unit"],
        "quarter_value": float(
            quarter_value
        ),
        "quarter_start": (
            quarter_start
        ),
        "quarter_end": (
            quarter_end
        ),
        "available_date": (
            available_date
        ),
        "source_method": (
            source_method
        ),
        "source_filed_date": max(
            filed_dates
        ),
        "source_accessions": "|".join(
            accessions
        ),
        "source_concepts": "|".join(
            concepts
        ),
        "source_component_count": len(
            components
        ),
    }


def build_quarterly_fundamental_events(
    canonical: pd.DataFrame,
    *,
    pit_config: SECPointInTimeConfig,
    config: SECQuarterlyReconstructionConfig,
) -> pd.DataFrame:
    """Reconstruct individual quarter values point-in-time."""
    events = prepare_point_in_time_events(
        canonical,
        config=pit_config,
    )

    events = events.loc[
        events[
            "canonical_metric"
        ].isin(
            config.additive_metrics
        )
        & events[
            "statement_type"
        ].eq(
            "duration"
        )
        & events[
            "duration_class"
        ].isin(
            [
                "quarter",
                "half_year_ytd",
                "nine_month_ytd",
                "annual",
            ]
        )
    ].copy()

    if events.empty:
        raise SECQuarterlyReconstructionError(
            "No duration fundamentals are available."
        )

    output_rows: list[
        dict[str, Any]
    ] = []

    group_columns = [
        "ticker",
        "canonical_metric",
        "unit",
    ]

    for _, group in events.groupby(
        group_columns,
        sort=True,
    ):
        group = group.sort_values(
            [
                "available_date",
                "filed_date",
                "concept_priority",
                "end_date",
            ]
        )

        state: dict[
            tuple[
                str,
                pd.Timestamp,
                pd.Timestamp,
            ],
            dict[str, Any],
        ] = {}

        emitted: set[
            tuple[
                pd.Timestamp,
                pd.Timestamp,
                str,
                float,
                str,
            ]
        ] = set()

        for (
            available_date,
            batch,
        ) in group.groupby(
            "available_date",
            sort=True,
        ):
            for row in batch.to_dict(
                orient="records"
            ):
                if pd.isna(
                    row[
                        "start_date"
                    ]
                ):
                    continue

                key = (
                    str(
                        row[
                            "duration_class"
                        ]
                    ),
                    pd.Timestamp(
                        row[
                            "start_date"
                        ]
                    ),
                    pd.Timestamp(
                        row[
                            "end_date"
                        ]
                    ),
                )

                previous = state.get(
                    key
                )

                if (
                    previous is None
                    or _fact_is_better(
                        row,
                        previous,
                    )
                ):
                    state[
                        key
                    ] = row

            current_rows = list(
                state.values()
            )

            # Direct quarterly values.
            for row in current_rows:
                if (
                    row[
                        "duration_class"
                    ]
                    != "quarter"
                ):
                    continue

                start = pd.Timestamp(
                    row[
                        "start_date"
                    ]
                )

                end = pd.Timestamp(
                    row[
                        "end_date"
                    ]
                )

                signature = (
                    start,
                    end,
                    "direct",
                    float(
                        row[
                            "value"
                        ]
                    ),
                    str(
                        row.get(
                            "accession_number",
                            "",
                        )
                    ),
                )

                if signature in emitted:
                    continue

                emitted.add(
                    signature
                )

                output_rows.append(
                    _quarter_event(
                        row=row,
                        quarter_value=(
                            float(
                                row[
                                    "value"
                                ]
                            )
                        ),
                        quarter_start=start,
                        quarter_end=end,
                        source_method=(
                            "direct"
                        ),
                        available_date=(
                            pd.Timestamp(
                                available_date
                            )
                        ),
                        components=[
                            row
                        ],
                    )
                )

            by_duration: dict[
                str,
                list[
                    dict[str, Any]
                ],
            ] = {}

            for row in current_rows:
                by_duration.setdefault(
                    str(
                        row[
                            "duration_class"
                        ]
                    ),
                    [],
                ).append(
                    row
                )

            derivations = (
                (
                    "half_year_ytd",
                    "quarter",
                    "derived_q2",
                ),
                (
                    "nine_month_ytd",
                    "half_year_ytd",
                    "derived_q3",
                ),
                (
                    "annual",
                    "nine_month_ytd",
                    "derived_q4",
                ),
            )

            for (
                long_class,
                prior_class,
                method,
            ) in derivations:
                for long_row in (
                    by_duration.get(
                        long_class,
                        [],
                    )
                ):
                    long_start = (
                        pd.Timestamp(
                            long_row[
                                "start_date"
                            ]
                        )
                    )

                    long_end = (
                        pd.Timestamp(
                            long_row[
                                "end_date"
                            ]
                        )
                    )

                    candidates = [
                        row
                        for row in (
                            by_duration.get(
                                prior_class,
                                [],
                            )
                        )
                        if (
                            pd.Timestamp(
                                row[
                                    "start_date"
                                ]
                            )
                            == long_start
                            and pd.Timestamp(
                                row[
                                    "end_date"
                                ]
                            )
                            < long_end
                        )
                    ]

                    if not candidates:
                        continue

                    prior = max(
                        candidates,
                        key=lambda row: (
                            pd.Timestamp(
                                row[
                                    "end_date"
                                ]
                            )
                        ),
                    )

                    prior_end = (
                        pd.Timestamp(
                            prior[
                                "end_date"
                            ]
                        )
                    )

                    quarter_start = (
                        prior_end
                        + pd.Timedelta(
                            days=1
                        )
                    )

                    quarter_value = (
                        float(
                            long_row[
                                "value"
                            ]
                        )
                        - float(
                            prior[
                                "value"
                            ]
                        )
                    )

                    accessions = (
                        str(
                            long_row.get(
                                "accession_number",
                                "",
                            )
                        )
                        + "|"
                        + str(
                            prior.get(
                                "accession_number",
                                "",
                            )
                        )
                    )

                    signature = (
                        quarter_start,
                        long_end,
                        method,
                        quarter_value,
                        accessions,
                    )

                    if signature in emitted:
                        continue

                    emitted.add(
                        signature
                    )

                    output_rows.append(
                        _quarter_event(
                            row=long_row,
                            quarter_value=(
                                quarter_value
                            ),
                            quarter_start=(
                                quarter_start
                            ),
                            quarter_end=(
                                long_end
                            ),
                            source_method=(
                                method
                            ),
                            available_date=(
                                pd.Timestamp(
                                    available_date
                                )
                            ),
                            components=[
                                long_row,
                                prior,
                            ],
                        )
                    )

    result = pd.DataFrame(
        output_rows
    )

    if result.empty:
        raise SECQuarterlyReconstructionError(
            "Quarter reconstruction produced no observations."
        )

    return (
        result.sort_values(
            [
                "ticker",
                "canonical_metric",
                "quarter_end",
                "available_date",
                "source_method",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def _quarter_event_is_better(
    new: Mapping[str, Any],
    previous: Mapping[str, Any],
) -> bool:
    """Select the best quarter representation as knowledge evolves."""
    new_available = pd.Timestamp(
        new[
            "available_date"
        ]
    )

    old_available = pd.Timestamp(
        previous[
            "available_date"
        ]
    )

    if new_available != old_available:
        return (
            new_available
            > old_available
        )

    method_priority = {
        "direct": 0,
        "derived_q2": 1,
        "derived_q3": 1,
        "derived_q4": 1,
    }

    return (
        method_priority.get(
            str(
                new[
                    "source_method"
                ]
            ),
            99,
        )
        <
        method_priority.get(
            str(
                previous[
                    "source_method"
                ]
            ),
            99,
        )
    )


def build_ttm_point_in_time_snapshots(
    quarterly_events: pd.DataFrame,
    as_of_dates: Sequence[Any] | pd.Series,
    *,
    config: SECQuarterlyReconstructionConfig,
) -> pd.DataFrame:
    """Build trailing-twelve-month values from individual quarters."""
    dates = (
        pd.to_datetime(
            pd.Series(
                as_of_dates
            ),
            errors="coerce",
        )
        .dt.normalize()
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not dates:
        raise SECQuarterlyReconstructionError(
            "No valid TTM snapshot dates were supplied."
        )

    snapshot_rows: list[
        dict[str, Any]
    ] = []

    for (
        ticker,
        metric,
        unit,
    ), group in quarterly_events.groupby(
        [
            "ticker",
            "canonical_metric",
            "unit",
        ],
        sort=True,
    ):
        events = (
            group.sort_values(
                [
                    "available_date",
                    "quarter_end",
                ]
            )
            .to_dict(
                orient="records"
            )
        )

        current: dict[
            pd.Timestamp,
            dict[str, Any],
        ] = {}

        index = 0

        for as_of_date in dates:
            as_of = pd.Timestamp(
                as_of_date
            )

            while (
                index < len(
                    events
                )
                and pd.Timestamp(
                    events[
                        index
                    ][
                        "available_date"
                    ]
                )
                <= as_of
            ):
                event = events[
                    index
                ]

                quarter_end = (
                    pd.Timestamp(
                        event[
                            "quarter_end"
                        ]
                    )
                )

                previous = (
                    current.get(
                        quarter_end
                    )
                )

                if (
                    previous is None
                    or _quarter_event_is_better(
                        event,
                        previous,
                    )
                ):
                    current[
                        quarter_end
                    ] = event

                index += 1

            if len(
                current
            ) < 4:
                continue

            quarter_ends = sorted(
                current
            )[-4:]

            gaps = [
                (
                    quarter_ends[
                        position
                    ]
                    - quarter_ends[
                        position - 1
                    ]
                ).days
                for position in range(
                    1,
                    4,
                )
            ]

            if any(
                gap
                < config.quarter_gap_min_days
                or gap
                > config.quarter_gap_max_days
                for gap in gaps
            ):
                continue

            span_days = (
                quarter_ends[-1]
                - quarter_ends[0]
            ).days

            if (
                span_days
                < config.ttm_span_min_days
                or span_days
                > config.ttm_span_max_days
            ):
                continue

            quarters = [
                current[
                    quarter_end
                ]
                for quarter_end in quarter_ends
            ]

            ttm_value = sum(
                float(
                    quarter[
                        "quarter_value"
                    ]
                )
                for quarter in quarters
            )

            maximum_available = max(
                pd.Timestamp(
                    quarter[
                        "available_date"
                    ]
                )
                for quarter in quarters
            )

            if maximum_available > as_of:
                raise SECQuarterlyReconstructionError(
                    "TTM snapshot contains future information."
                )

            snapshot_rows.append(
                {
                    "as_of_date": (
                        as_of
                    ),
                    "ticker": ticker,
                    "canonical_metric": (
                        metric
                    ),
                    "unit": unit,
                    "ttm_value": float(
                        ttm_value
                    ),
                    "latest_quarter_end": (
                        quarter_ends[-1]
                    ),
                    "earliest_quarter_end": (
                        quarter_ends[0]
                    ),
                    "quarter_count": 4,
                    "quarter_span_days": (
                        span_days
                    ),
                    "latest_component_available_date": (
                        maximum_available
                    ),
                    "quarter_methods": "|".join(
                        str(
                            quarter[
                                "source_method"
                            ]
                        )
                        for quarter in quarters
                    ),
                }
            )

    result = pd.DataFrame(
        snapshot_rows
    )

    if result.empty:
        raise SECQuarterlyReconstructionError(
            "TTM reconstruction produced no snapshots."
        )

    duplicates = result.duplicated(
        [
            "as_of_date",
            "ticker",
            "canonical_metric",
        ]
    )

    if duplicates.any():
        raise SECQuarterlyReconstructionError(
            "TTM snapshots contain duplicates."
        )

    return (
        result.sort_values(
            [
                "as_of_date",
                "ticker",
                "canonical_metric",
            ]
        )
        .reset_index(
            drop=True
        )
    )