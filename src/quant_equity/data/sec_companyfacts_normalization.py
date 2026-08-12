"""Normalize raw SEC Company Facts JSON into a long tabular dataset."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_equity.data.sec_companyfacts import (
    normalize_cik,
    validate_companyfacts_payload,
)


class SECNormalizationError(ValueError):
    """Raised when SEC Company Facts cannot be normalized."""


NORMALIZED_COLUMNS = (
    "ticker",
    "cik",
    "entity_name",
    "taxonomy",
    "concept",
    "label",
    "description",
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
    "source_file",
)


def _normalize_optional_text(
    value: Any,
) -> str | None:
    """Normalize an optional text field."""
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def _normalize_numeric_value(
    value: Any,
    *,
    ticker: str,
    concept: str,
) -> float:
    """Normalize a SEC XBRL numeric fact."""
    if value is None:
        raise SECNormalizationError(
            f"{ticker} {concept} contains a missing value."
        )

    try:
        numeric_value = float(value)
    except (
        TypeError,
        ValueError,
    ) as error:
        raise SECNormalizationError(
            f"{ticker} {concept} contains "
            f"a non-numeric value: {value!r}."
        ) from error

    if not np.isfinite(
        numeric_value
    ):
        raise SECNormalizationError(
            f"{ticker} {concept} contains "
            "a non-finite value."
        )

    return numeric_value


def normalize_companyfacts_payload(
    payload: Mapping[str, Any],
    *,
    ticker: str,
    source_file: Path | str,
) -> pd.DataFrame:
    """Flatten one SEC Company Facts payload."""
    normalized_ticker = (
        str(ticker)
        .strip()
        .upper()
    )

    if not normalized_ticker:
        raise SECNormalizationError(
            "Ticker cannot be empty."
        )

    expected_cik = normalize_cik(
        payload.get(
            "cik",
            "",
        )
    )

    validate_companyfacts_payload(
        payload,
        expected_cik=expected_cik,
    )

    entity_name = str(
        payload["entityName"]
    ).strip()

    facts = payload["facts"]

    rows: list[
        dict[str, Any]
    ] = []

    for (
        taxonomy,
        concepts,
    ) in facts.items():
        if not isinstance(
            concepts,
            Mapping,
        ):
            raise SECNormalizationError(
                f"{normalized_ticker}: "
                f"taxonomy {taxonomy!r} "
                "is not an object."
            )

        for (
            concept,
            concept_data,
        ) in concepts.items():
            if not isinstance(
                concept_data,
                Mapping,
            ):
                raise SECNormalizationError(
                    f"{normalized_ticker}: "
                    f"concept {concept!r} "
                    "is not an object."
                )

            label = (
                _normalize_optional_text(
                    concept_data.get(
                        "label"
                    )
                )
            )

            description = (
                _normalize_optional_text(
                    concept_data.get(
                        "description"
                    )
                )
            )

            units = concept_data.get(
                "units",
                {},
            )

            if not isinstance(
                units,
                Mapping,
            ):
                raise SECNormalizationError(
                    f"{normalized_ticker}: "
                    f"{taxonomy}:{concept} "
                    "has invalid units."
                )

            for (
                unit,
                observations,
            ) in units.items():
                if not isinstance(
                    observations,
                    list,
                ):
                    raise SECNormalizationError(
                        f"{normalized_ticker}: "
                        f"{taxonomy}:{concept} "
                        f"unit {unit!r} is not a list."
                    )

                for observation in observations:
                    if not isinstance(
                        observation,
                        Mapping,
                    ):
                        raise SECNormalizationError(
                            f"{normalized_ticker}: "
                            f"{taxonomy}:{concept} "
                            "contains an invalid observation."
                        )

                    rows.append(
                        {
                            "ticker": (
                                normalized_ticker
                            ),
                            "cik": expected_cik,
                            "entity_name": (
                                entity_name
                            ),
                            "taxonomy": str(
                                taxonomy
                            ),
                            "concept": str(
                                concept
                            ),
                            "label": label,
                            "description": (
                                description
                            ),
                            "unit": str(
                                unit
                            ),
                            "value": (
                                _normalize_numeric_value(
                                    observation.get(
                                        "val"
                                    ),
                                    ticker=(
                                        normalized_ticker
                                    ),
                                    concept=str(
                                        concept
                                    ),
                                )
                            ),
                            "start_date": (
                                observation.get(
                                    "start"
                                )
                            ),
                            "end_date": (
                                observation.get(
                                    "end"
                                )
                            ),
                            "filed_date": (
                                observation.get(
                                    "filed"
                                )
                            ),
                            "form": (
                                _normalize_optional_text(
                                    observation.get(
                                        "form"
                                    )
                                )
                            ),
                            "fiscal_year": (
                                observation.get(
                                    "fy"
                                )
                            ),
                            "fiscal_period": (
                                _normalize_optional_text(
                                    observation.get(
                                        "fp"
                                    )
                                )
                            ),
                            "accession_number": (
                                _normalize_optional_text(
                                    observation.get(
                                        "accn"
                                    )
                                )
                            ),
                            "frame": (
                                _normalize_optional_text(
                                    observation.get(
                                        "frame"
                                    )
                                )
                            ),
                            "source_file": str(
                                source_file
                            ),
                        }
                    )

    data = pd.DataFrame(
        rows,
        columns=NORMALIZED_COLUMNS,
    )

    if data.empty:
        raise SECNormalizationError(
            f"{normalized_ticker}: "
            "Company Facts produced no observations."
        )

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
        bad_rows = int(
            data[
                "end_date"
            ].isna().sum()
        )

        raise SECNormalizationError(
            f"{normalized_ticker}: "
            f"{bad_rows} observations "
            "have invalid end dates."
        )

    if data[
        "filed_date"
    ].isna().any():
        bad_rows = int(
            data[
                "filed_date"
            ].isna().sum()
        )

        raise SECNormalizationError(
            f"{normalized_ticker}: "
            f"{bad_rows} observations "
            "have invalid filed dates."
        )

    data[
        "fiscal_year"
    ] = pd.to_numeric(
        data[
            "fiscal_year"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    return (
        data.sort_values(
            [
                "ticker",
                "taxonomy",
                "concept",
                "unit",
                "end_date",
                "filed_date",
                "accession_number",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


def validate_normalized_companyfacts(
    data: pd.DataFrame,
) -> None:
    """Validate the normalized SEC Company Facts table."""
    missing_columns = sorted(
        set(
            NORMALIZED_COLUMNS
        ).difference(
            data.columns
        )
    )

    if missing_columns:
        raise SECNormalizationError(
            "Normalized SEC data are missing columns: "
            + ", ".join(
                missing_columns
            )
            + "."
        )

    if data.empty:
        raise SECNormalizationError(
            "Normalized SEC data cannot be empty."
        )

    required_text_columns = (
        "ticker",
        "cik",
        "entity_name",
        "taxonomy",
        "concept",
        "unit",
    )

    for column in required_text_columns:
        values = (
            data[column]
            .astype("string")
            .str.strip()
        )

        if (
            values.isna().any()
            or values.eq("").any()
        ):
            raise SECNormalizationError(
                f"Column {column} "
                "contains missing values."
            )

    numeric_values = pd.to_numeric(
        data["value"],
        errors="coerce",
    )

    if (
        numeric_values.isna().any()
        or not np.isfinite(
            numeric_values.to_numpy(
                dtype=float
            )
        ).all()
    ):
        raise SECNormalizationError(
            "Normalized SEC data contain "
            "invalid numeric values."
        )

    for column in (
        "end_date",
        "filed_date",
    ):
        dates = pd.to_datetime(
            data[column],
            errors="coerce",
        )

        if dates.isna().any():
            raise SECNormalizationError(
                f"Column {column} "
                "contains invalid dates."
            )


def build_companyfacts_quality_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create one normalization summary row per company."""
    validate_normalized_companyfacts(
        data
    )

    working = data.copy()

    working[
        "is_duration"
    ] = working[
        "start_date"
    ].notna()

    working[
        "is_instant"
    ] = working[
        "start_date"
    ].isna()

    summary = (
        working.groupby(
            [
                "ticker",
                "cik",
                "entity_name",
            ],
            as_index=False,
        )
        .agg(
            observation_count=(
                "concept",
                "size",
            ),
            taxonomy_count=(
                "taxonomy",
                "nunique",
            ),
            concept_count=(
                "concept",
                "nunique",
            ),
            unit_count=(
                "unit",
                "nunique",
            ),
            form_count=(
                "form",
                "nunique",
            ),
            first_period_end=(
                "end_date",
                "min",
            ),
            last_period_end=(
                "end_date",
                "max",
            ),
            first_filed_date=(
                "filed_date",
                "min",
            ),
            last_filed_date=(
                "filed_date",
                "max",
            ),
            duration_fact_count=(
                "is_duration",
                "sum",
            ),
            instant_fact_count=(
                "is_instant",
                "sum",
            ),
        )
        .sort_values(
            "ticker"
        )
        .reset_index(
            drop=True
        )
    )

    return summary