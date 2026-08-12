"""SEC EDGAR Company Facts downloading and raw caching."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any

import requests

from quant_equity.config import RAW_DATA_DIR


LOGGER = logging.getLogger(
    "quant_equity.sec_companyfacts"
)

RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


class SECFundamentalsError(RuntimeError):
    """Base error for SEC fundamentals."""


class SECConfigurationError(SECFundamentalsError):
    """Raised when SEC configuration is invalid."""


class SECRequestError(SECFundamentalsError):
    """Raised when a SEC request cannot be completed."""


class SECCompanyFactsValidationError(
    SECFundamentalsError
):
    """Raised when a Company Facts payload is invalid."""


class SECCompanyFactsBatchError(
    SECFundamentalsError
):
    """Raised when one or more companies fail."""


@dataclass(frozen=True)
class SECCompanyFactsConfig:
    """Configuration for SEC Company Facts downloads."""

    companyfacts_base_url: str = (
        "https://data.sec.gov/api/xbrl/companyfacts"
    )

    user_agent_env_var: str = (
        "SEC_USER_AGENT"
    )

    timeout_seconds: float = 30.0

    request_interval_seconds: float = 0.20

    maximum_retries: int = 4

    retry_backoff_seconds: float = 1.0

    overwrite_raw: bool = False

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> SECCompanyFactsConfig:
        """Create SEC configuration from project YAML."""
        config = cls(
            companyfacts_base_url=str(
                values.get(
                    "companyfacts_base_url",
                    (
                        "https://data.sec.gov/"
                        "api/xbrl/companyfacts"
                    ),
                )
            ).rstrip("/"),
            user_agent_env_var=str(
                values.get(
                    "user_agent_env_var",
                    "SEC_USER_AGENT",
                )
            ),
            timeout_seconds=float(
                values.get(
                    "timeout_seconds",
                    30.0,
                )
            ),
            request_interval_seconds=float(
                values.get(
                    "request_interval_seconds",
                    0.20,
                )
            ),
            maximum_retries=int(
                values.get(
                    "maximum_retries",
                    4,
                )
            ),
            retry_backoff_seconds=float(
                values.get(
                    "retry_backoff_seconds",
                    1.0,
                )
            ),
            overwrite_raw=bool(
                values.get(
                    "overwrite_raw",
                    False,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate SEC download settings."""
        if not self.companyfacts_base_url:
            raise SECConfigurationError(
                "companyfacts_base_url cannot be empty."
            )

        if not self.user_agent_env_var:
            raise SECConfigurationError(
                "user_agent_env_var cannot be empty."
            )

        if self.timeout_seconds <= 0.0:
            raise SECConfigurationError(
                "timeout_seconds must be positive."
            )

        if self.request_interval_seconds < 0.0:
            raise SECConfigurationError(
                "request_interval_seconds "
                "cannot be negative."
            )

        if self.maximum_retries < 0:
            raise SECConfigurationError(
                "maximum_retries cannot be negative."
            )

        if self.retry_backoff_seconds < 0.0:
            raise SECConfigurationError(
                "retry_backoff_seconds "
                "cannot be negative."
            )


@dataclass(frozen=True)
class CompanyFactsDownloadRecord:
    """Result for one downloaded or cached company."""

    ticker: str
    cik: str
    entity_name: str
    concept_count: int
    source: str
    path: Path


@dataclass(frozen=True)
class SECCompanyFactsBatchResult:
    """Result of a complete universe download."""

    records: tuple[
        CompanyFactsDownloadRecord,
        ...
    ]

    downloaded_tickers: tuple[str, ...]
    cached_tickers: tuple[str, ...]
    raw_files: tuple[Path, ...]


def normalize_cik(
    value: Any,
) -> str:
    """Normalize a SEC CIK to exactly ten digits."""
    text = str(value).strip()

    if text.upper().startswith("CIK"):
        text = text[3:].strip()

    if text.endswith(".0"):
        candidate = text[:-2]

        if candidate.isdigit():
            text = candidate

    if not text.isdigit():
        raise SECCompanyFactsValidationError(
            f"Invalid CIK: {value!r}."
        )

    if len(text) > 10:
        raise SECCompanyFactsValidationError(
            f"CIK contains more than 10 digits: {value!r}."
        )

    return text.zfill(10)


def resolve_sec_user_agent(
    config: SECCompanyFactsConfig,
) -> str:
    """Read the SEC User-Agent from an environment variable."""
    value = os.getenv(
        config.user_agent_env_var,
        "",
    ).strip()

    if not value:
        raise SECConfigurationError(
            "SEC User-Agent is missing. "
            "Set the PowerShell environment variable "
            f"${{env:{config.user_agent_env_var}}} "
            "before downloading SEC data."
        )

    return value


def build_companyfacts_url(
    cik: Any,
    *,
    base_url: str,
) -> str:
    """Build the SEC Company Facts API URL."""
    normalized_cik = normalize_cik(cik)

    return (
        f"{base_url.rstrip('/')}"
        f"/CIK{normalized_cik}.json"
    )


def get_raw_companyfacts_path(
    ticker: str,
    cik: Any,
    *,
    raw_root: Path = RAW_DATA_DIR,
) -> Path:
    """Return the raw path for one SEC Company Facts file."""
    safe_ticker = re.sub(
        r"[^A-Z0-9.-]",
        "_",
        ticker.strip().upper(),
    )

    if not safe_ticker:
        raise SECCompanyFactsValidationError(
            "Ticker cannot be empty."
        )

    normalized_cik = normalize_cik(cik)

    filename = (
        f"{safe_ticker}"
        f"__CIK{normalized_cik}"
        ".json"
    )

    return (
        raw_root
        / "fundamentals"
        / "sec"
        / "companyfacts"
        / filename
    )


def validate_companyfacts_payload(
    payload: Any,
    *,
    expected_cik: Any,
) -> None:
    """Validate the minimum SEC Company Facts structure."""
    if not isinstance(payload, dict):
        raise SECCompanyFactsValidationError(
            "Company Facts payload must be a JSON object."
        )

    required_fields = {
        "cik",
        "entityName",
        "facts",
    }

    missing = sorted(
        required_fields.difference(payload)
    )

    if missing:
        raise SECCompanyFactsValidationError(
            "Company Facts payload is missing fields: "
            + ", ".join(missing)
            + "."
        )

    actual_cik = normalize_cik(
        payload["cik"]
    )

    normalized_expected_cik = (
        normalize_cik(expected_cik)
    )

    if actual_cik != normalized_expected_cik:
        raise SECCompanyFactsValidationError(
            "Company Facts CIK does not match "
            f"the requested CIK: "
            f"{actual_cik} != "
            f"{normalized_expected_cik}."
        )

    entity_name = payload["entityName"]

    if (
        not isinstance(entity_name, str)
        or not entity_name.strip()
    ):
        raise SECCompanyFactsValidationError(
            "Company Facts entityName is invalid."
        )

    if not isinstance(
        payload["facts"],
        dict,
    ):
        raise SECCompanyFactsValidationError(
            "Company Facts facts field must be an object."
        )


def _count_concepts(
    payload: Mapping[str, Any],
) -> int:
    """Count concepts across SEC taxonomy namespaces."""
    facts = payload.get(
        "facts",
        {},
    )

    if not isinstance(facts, Mapping):
        return 0

    total = 0

    for concepts in facts.values():
        if isinstance(
            concepts,
            Mapping,
        ):
            total += len(concepts)

    return total


def _write_bytes_atomically(
    content: bytes,
    path: Path,
) -> None:
    """Write raw SEC bytes atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        ".tmp.json"
    )

    temporary_path.unlink(
        missing_ok=True
    )

    temporary_path.write_bytes(
        content
    )

    temporary_path.replace(path)


def load_cached_companyfacts(
    path: Path,
    *,
    expected_cik: Any,
) -> dict[str, Any]:
    """Load and validate a cached raw Company Facts file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Company Facts cache not found: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise SECCompanyFactsValidationError(
            f"Invalid cached Company Facts file: {path}"
        ) from error

    validate_companyfacts_payload(
        payload,
        expected_cik=expected_cik,
    )

    return payload


class SECCompanyFactsClient:
    """Small HTTP client for SEC Company Facts."""

    def __init__(
        self,
        config: SECCompanyFactsConfig,
        *,
        user_agent: str,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config

        self.session = (
            session
            if session is not None
            else requests.Session()
        )

        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/json",
                "Accept-Encoding": (
                    "gzip, deflate"
                ),
            }
        )

        self._last_request_time: (
            float | None
        ) = None

    def _throttle(self) -> None:
        """Respect the configured spacing between requests."""
        if self._last_request_time is None:
            return

        elapsed = (
            time.monotonic()
            - self._last_request_time
        )

        remaining = (
            self.config.request_interval_seconds
            - elapsed
        )

        if remaining > 0.0:
            time.sleep(remaining)

    def fetch_companyfacts(
        self,
        cik: Any,
    ) -> tuple[
        dict[str, Any],
        bytes,
    ]:
        """Download and validate one Company Facts response."""
        normalized_cik = normalize_cik(
            cik
        )

        url = build_companyfacts_url(
            normalized_cik,
            base_url=(
                self.config.companyfacts_base_url
            ),
        )

        last_error: Exception | None = None

        for attempt in range(
            self.config.maximum_retries + 1
        ):
            self._throttle()

            try:
                self._last_request_time = (
                    time.monotonic()
                )

                response = self.session.get(
                    url,
                    timeout=(
                        self.config.timeout_seconds
                    ),
                )
            except requests.RequestException as error:
                last_error = error

                if (
                    attempt
                    >= self.config.maximum_retries
                ):
                    break

                wait_seconds = (
                    self.config.retry_backoff_seconds
                    * (2**attempt)
                )

                LOGGER.warning(
                    "SEC request failed for CIK %s. "
                    "Retrying in %.2f seconds.",
                    normalized_cik,
                    wait_seconds,
                )

                time.sleep(
                    wait_seconds
                )

                continue

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as error:
                    raise SECRequestError(
                        "SEC returned invalid JSON "
                        f"for CIK {normalized_cik}."
                    ) from error

                validate_companyfacts_payload(
                    payload,
                    expected_cik=normalized_cik,
                )

                return (
                    payload,
                    response.content,
                )

            if (
                response.status_code
                in RETRYABLE_STATUS_CODES
                and attempt
                < self.config.maximum_retries
            ):
                wait_seconds = (
                    self.config.retry_backoff_seconds
                    * (2**attempt)
                )

                LOGGER.warning(
                    "SEC returned HTTP %s for CIK %s. "
                    "Retrying in %.2f seconds.",
                    response.status_code,
                    normalized_cik,
                    wait_seconds,
                )

                time.sleep(
                    wait_seconds
                )

                continue

            raise SECRequestError(
                "SEC Company Facts request failed "
                f"for CIK {normalized_cik}: "
                f"HTTP {response.status_code}."
            )

        raise SECRequestError(
            "SEC Company Facts request failed "
            f"for CIK {normalized_cik} "
            "after all retries."
        ) from last_error


def download_companyfacts_for_company(
    ticker: str,
    cik: Any,
    *,
    client: SECCompanyFactsClient,
    raw_root: Path = RAW_DATA_DIR,
    overwrite: bool = False,
) -> CompanyFactsDownloadRecord:
    """Download or reuse Company Facts for one company."""
    normalized_ticker = (
        ticker.strip().upper()
    )

    normalized_cik = normalize_cik(
        cik
    )

    path = get_raw_companyfacts_path(
        normalized_ticker,
        normalized_cik,
        raw_root=raw_root,
    )

    if (
        path.exists()
        and not overwrite
    ):
        payload = load_cached_companyfacts(
            path,
            expected_cik=normalized_cik,
        )

        source = "cache"

    else:
        (
            payload,
            raw_bytes,
        ) = client.fetch_companyfacts(
            normalized_cik
        )

        _write_bytes_atomically(
            raw_bytes,
            path,
        )

        source = "download"

    return CompanyFactsDownloadRecord(
        ticker=normalized_ticker,
        cik=normalized_cik,
        entity_name=str(
            payload["entityName"]
        ).strip(),
        concept_count=(
            _count_concepts(payload)
        ),
        source=source,
        path=path,
    )

def download_sec_companyfacts_universe(
    project_config: Mapping[str, Any],
    *,
    force: bool = False,
) -> SECCompanyFactsBatchResult:
    """Download SEC Company Facts for the configured universe."""
    from quant_equity.data.universe import (
        load_universe,
    )

    sec_config = (
        SECCompanyFactsConfig.from_mapping(
            project_config.get(
                "sec_fundamentals",
                {},
            )
        )
    )

    user_agent = resolve_sec_user_agent(
        sec_config
    )

    client = SECCompanyFactsClient(
        sec_config,
        user_agent=user_agent,
    )

    universe_version = str(
        project_config[
            "universe"
        ]["version"]
    )

    universe = load_universe(
        universe_version
    )

    downloaded_tickers: list[str] = []
    cached_tickers: list[str] = []
    raw_files: list[Path] = []
    records: list[
        CompanyFactsDownloadRecord
    ] = []

    errors: list[str] = []

    overwrite = (
        force
        or sec_config.overwrite_raw
    )

    for row in universe.itertuples(
        index=False
    ):
        ticker = str(
            row.ticker
        ).strip().upper()

        cik = getattr(
            row,
            "cik",
            None,
        )

        try:
            record = (
                download_companyfacts_for_company(
                    ticker,
                    cik,
                    client=client,
                    overwrite=overwrite,
                )
            )
        except SECFundamentalsError as error:
            LOGGER.exception(
                "SEC Company Facts failed for %s.",
                ticker,
            )

            errors.append(
                f"{ticker}: {error}"
            )

            continue

        records.append(
            record
        )

        raw_files.append(
            record.path
        )

        if record.source == "download":
            downloaded_tickers.append(
                ticker
            )
        else:
            cached_tickers.append(
                ticker
            )

        LOGGER.info(
            "SEC Company Facts %s: %s "
            "(CIK %s, concepts %s).",
            record.source,
            ticker,
            record.cik,
            record.concept_count,
        )

    if errors:
        raise SECCompanyFactsBatchError(
            "One or more SEC Company Facts "
            "downloads failed:\n- "
            + "\n- ".join(errors)
        )

    return SECCompanyFactsBatchResult(
        records=tuple(records),
        downloaded_tickers=tuple(
            downloaded_tickers
        ),
        cached_tickers=tuple(
            cached_tickers
        ),
        raw_files=tuple(
            raw_files
        ),
    )