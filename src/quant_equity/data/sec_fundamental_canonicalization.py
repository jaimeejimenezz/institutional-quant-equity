"""Canonicalize normalized SEC Company Facts into economic metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


class SECCanonicalizationError(ValueError):
    """Raised when SEC facts cannot be canonicalized."""


@dataclass(frozen=True)
class DurationBands:
    """Day ranges used to classify duration facts."""

    quarter_min: int = 70
    quarter_max: int = 110

    half_year_min: int = 150
    half_year_max: int = 210

    nine_month_min: int = 230
    nine_month_max: int = 300

    annual_min: int = 330
    annual_max: int = 400

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> DurationBands:
        """Build duration bands from configuration."""
        bands = cls(
            quarter_min=int(
                values.get(
                    "quarter_min_days",
                    70,
                )
            ),
            quarter_max=int(
                values.get(
                    "quarter_max_days",
                    110,
                )
            ),
            half_year_min=int(
                values.get(
                    "half_year_min_days",
                    150,
                )
            ),
            half_year_max=int(
                values.get(
                    "half_year_max_days",
                    210,
                )
            ),
            nine_month_min=int(
                values.get(
                    "nine_month_min_days",
                    230,
                )
            ),
            nine_month_max=int(
                values.get(
                    "nine_month_max_days",
                    300,
                )
            ),
            annual_min=int(
                values.get(
                    "annual_min_days",
                    330,
                )
            ),
            annual_max=int(
                values.get(
                    "annual_max_days",
                    400,
                )
            ),
        )

        bands.validate()

        return bands

    def validate(self) -> None:
        """Validate duration ranges."""
        pairs = (
            (
                self.quarter_min,
                self.quarter_max,
            ),
            (
                self.half_year_min,
                self.half_year_max,
            ),
            (
                self.nine_month_min,
                self.nine_month_max,
            ),
            (
                self.annual_min,
                self.annual_max,
            ),
        )

        if any(
            lower < 1
            or upper < lower
            for lower, upper in pairs
        ):
            raise SECCanonicalizationError(
                "Invalid duration bands."
            )


@dataclass(frozen=True)
class CanonicalMetricDefinition:
    """Definition of one economic metric."""

    name: str
    statement_type: str
    units: tuple[str, ...]
    concepts: tuple[str, ...]


@dataclass(frozen=True)
class ConceptMapping:
    """Parsed XBRL concept mapping."""

    accepted_taxonomies: tuple[str, ...]
    accepted_forms: tuple[str, ...]
    metrics: tuple[
        CanonicalMetricDefinition,
        ...
    ]


def load_concept_mapping(
    path: Path,
) -> ConceptMapping:
    """Load and validate the canonical XBRL mapping."""
    if not path.exists():
        raise FileNotFoundError(
            f"XBRL concept mapping not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = yaml.safe_load(
            handle
        )

    if not isinstance(
        payload,
        Mapping,
    ):
        raise SECCanonicalizationError(
            "XBRL mapping must be a YAML object."
        )

    raw_metrics = payload.get(
        "metrics"
    )

    if not isinstance(
        raw_metrics,
        Mapping,
    ):
        raise SECCanonicalizationError(
            "XBRL mapping requires metrics."
        )

    definitions: list[
        CanonicalMetricDefinition
    ] = []

    seen_concepts: dict[
        tuple[str, str],
        str,
    ] = {}

    for (
        metric_name,
        metric_values,
    ) in raw_metrics.items():
        if not isinstance(
            metric_values,
            Mapping,
        ):
            raise SECCanonicalizationError(
                f"Metric {metric_name} "
                "must be an object."
            )

        statement_type = str(
            metric_values.get(
                "statement_type",
                "",
            )
        ).strip()

        if statement_type not in {
            "instant",
            "duration",
        }:
            raise SECCanonicalizationError(
                f"Metric {metric_name} "
                "has invalid statement_type."
            )

        units = tuple(
            str(value).strip()
            for value in metric_values.get(
                "units",
                [],
            )
        )

        concepts = tuple(
            str(value).strip()
            for value in metric_values.get(
                "concepts",
                [],
            )
        )

        if not units:
            raise SECCanonicalizationError(
                f"Metric {metric_name} "
                "requires units."
            )

        if not concepts:
            raise SECCanonicalizationError(
                f"Metric {metric_name} "
                "requires concepts."
            )

        for concept in concepts:
            key = (
                statement_type,
                concept,
            )

            previous_metric = (
                seen_concepts.get(
                    key
                )
            )

            if (
                previous_metric is not None
                and previous_metric
                != metric_name
            ):
                raise SECCanonicalizationError(
                    f"Concept {concept} is assigned "
                    "to multiple metrics."
                )

            seen_concepts[
                key
            ] = str(
                metric_name
            )

        definitions.append(
            CanonicalMetricDefinition(
                name=str(
                    metric_name
                ),
                statement_type=(
                    statement_type
                ),
                units=units,
                concepts=concepts,
            )
        )

    accepted_taxonomies = tuple(
        str(value).strip()
        for value in payload.get(
            "accepted_taxonomies",
            [],
        )
    )

    accepted_forms = tuple(
        str(value).strip()
        for value in payload.get(
            "accepted_forms",
            [],
        )
    )

    if not accepted_taxonomies:
        raise SECCanonicalizationError(
            "At least one taxonomy is required."
        )

    if not accepted_forms:
        raise SECCanonicalizationError(
            "At least one filing form is required."
        )

    return ConceptMapping(
        accepted_taxonomies=(
            accepted_taxonomies
        ),
        accepted_forms=(
            accepted_forms
        ),
        metrics=tuple(
            definitions
        ),
    )


def classify_duration(
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp,
    *,
    bands: DurationBands,
) -> tuple[
    int | None,
    str,
]:
    """Classify the duration represented by one fact."""
    if pd.isna(
        start_date
    ):
        return (
            None,
            "instant",
        )

    start = pd.Timestamp(
        start_date
    )

    end = pd.Timestamp(
        end_date
    )

    days = (
        end - start
    ).days + 1

    if days <= 0:
        raise SECCanonicalizationError(
            "A fact contains a non-positive duration."
        )

    if (
        bands.quarter_min
        <= days
        <= bands.quarter_max
    ):
        return (
            days,
            "quarter",
        )

    if (
        bands.half_year_min
        <= days
        <= bands.half_year_max
    ):
        return (
            days,
            "half_year_ytd",
        )

    if (
        bands.nine_month_min
        <= days
        <= bands.nine_month_max
    ):
        return (
            days,
            "nine_month_ytd",
        )

    if (
        bands.annual_min
        <= days
        <= bands.annual_max
    ):
        return (
            days,
            "annual",
        )

    return (
        days,
        "other_duration",
    )


def canonicalize_sec_facts(
    normalized: pd.DataFrame,
    *,
    mapping: ConceptMapping,
    duration_bands: DurationBands,
) -> pd.DataFrame:
    """Map XBRL concepts to canonical economic metrics."""
    required_columns = {
        "ticker",
        "cik",
        "entity_name",
        "taxonomy",
        "concept",
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
    }

    missing = sorted(
        required_columns.difference(
            normalized.columns
        )
    )

    if missing:
        raise SECCanonicalizationError(
            "Normalized SEC facts are "
            "missing columns: "
            + ", ".join(
                missing
            )
            + "."
        )

    candidate_frames: list[
        pd.DataFrame
    ] = []

    for definition in mapping.metrics:
        for (
            priority,
            concept,
        ) in enumerate(
            definition.concepts,
            start=1,
        ):
            mask = (
                normalized[
                    "taxonomy"
                ].isin(
                    mapping.accepted_taxonomies
                )
                & normalized[
                    "form"
                ].isin(
                    mapping.accepted_forms
                )
                & normalized[
                    "concept"
                ].eq(
                    concept
                )
                & normalized[
                    "unit"
                ].isin(
                    definition.units
                )
            )

            subset = (
                normalized.loc[
                    mask
                ]
                .copy()
            )

            if subset.empty:
                continue

            subset[
                "canonical_metric"
            ] = definition.name

            subset[
                "statement_type"
            ] = (
                definition.statement_type
            )

            subset[
                "concept_priority"
            ] = priority

            candidate_frames.append(
                subset
            )

    if not candidate_frames:
        raise SECCanonicalizationError(
            "No SEC facts matched the "
            "canonical concept mapping."
        )

    canonical = pd.concat(
        candidate_frames,
        ignore_index=True,
    )

    period_days: list[
        int | None
    ] = []

    duration_classes: list[
        str
    ] = []

    for row in canonical.itertuples(
        index=False
    ):
        (
            days,
            duration_class,
        ) = classify_duration(
            row.start_date,
            row.end_date,
            bands=duration_bands,
        )

        period_days.append(
            days
        )

        duration_classes.append(
            duration_class
        )

    canonical[
        "period_days"
    ] = pd.array(
        period_days,
        dtype="Int64",
    )

    canonical[
        "duration_class"
    ] = duration_classes

    canonical[
        "is_amendment"
    ] = (
        canonical[
            "form"
        ]
        .astype(
            "string"
        )
        .str.endswith(
            "/A"
        )
        .fillna(
            False
        )
    )

    invalid_instant = (
        canonical[
            "statement_type"
        ].eq(
            "instant"
        )
        & canonical[
            "start_date"
        ].notna()
    )

    invalid_duration = (
        canonical[
            "statement_type"
        ].eq(
            "duration"
        )
        & canonical[
            "start_date"
        ].isna()
    )

    canonical[
        "statement_type_match"
    ] = ~(
        invalid_instant
        | invalid_duration
    )

    canonical = (
        canonical.sort_values(
            [
                "ticker",
                "canonical_metric",
                "end_date",
                "filed_date",
                "concept_priority",
                "accession_number",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    return canonical


def build_canonical_coverage(
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize canonical metric coverage."""
    if canonical.empty:
        raise SECCanonicalizationError(
            "Canonical facts cannot be empty."
        )

    coverage = (
        canonical.groupby(
            "canonical_metric",
            as_index=False,
        )
        .agg(
            observations=(
                "value",
                "size",
            ),
            companies=(
                "ticker",
                "nunique",
            ),
            concepts_used=(
                "concept",
                "nunique",
            ),
            first_filed_date=(
                "filed_date",
                "min",
            ),
            last_filed_date=(
                "filed_date",
                "max",
            ),
            statement_type_match_rate=(
                "statement_type_match",
                "mean",
            ),
        )
        .sort_values(
            [
                "companies",
                "observations",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return coverage