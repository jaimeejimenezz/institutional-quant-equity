"""Tests for SEC EDGAR Company Facts downloading."""

from __future__ import annotations

import json

import pytest

from quant_equity.data import (
    SECCompanyFactsClient,
    SECCompanyFactsConfig,
    SECCompanyFactsValidationError,
    SECConfigurationError,
    build_companyfacts_url,
    download_companyfacts_for_company,
    normalize_cik,
    resolve_sec_user_agent,
)


def make_payload(
    cik: int = 320193,
) -> dict:
    """Create a minimal SEC-like Company Facts payload."""
    return {
        "cik": cik,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "label": "Assets",
                    "description": "Assets",
                    "units": {
                        "USD": [],
                    },
                }
            }
        },
    }


class FakeResponse:
    """Minimal requests-like response."""

    def __init__(
        self,
        status_code: int,
        payload: dict,
    ) -> None:
        self.status_code = status_code

        self._payload = payload

        self.content = json.dumps(payload).encode("utf-8")

    def json(self) -> dict:
        """Return response JSON."""
        return self._payload


class FakeSession:
    """Minimal requests-like session."""

    def __init__(
        self,
        responses: list[FakeResponse],
    ) -> None:
        self.responses = list(responses)

        self.calls: list[tuple[str, float]] = []

        self.headers: dict[
            str,
            str,
        ] = {}

    def get(
        self,
        url: str,
        *,
        timeout: float,
    ) -> FakeResponse:
        """Return the next fake response."""
        self.calls.append(
            (
                url,
                timeout,
            )
        )

        return self.responses.pop(0)


def make_config() -> SECCompanyFactsConfig:
    """Create fast SEC test configuration."""
    return SECCompanyFactsConfig(
        companyfacts_base_url=("https://data.sec.gov/api/xbrl/companyfacts"),
        user_agent_env_var=("SEC_USER_AGENT"),
        timeout_seconds=10.0,
        request_interval_seconds=0.0,
        maximum_retries=2,
        retry_backoff_seconds=0.0,
        overwrite_raw=False,
    )


def test_normalize_cik_to_ten_digits() -> None:
    """CIKs should always contain ten digits."""
    assert normalize_cik(320193) == "0000320193"

    assert normalize_cik("0000320193") == "0000320193"

    assert normalize_cik("CIK320193") == "0000320193"


def test_invalid_cik_is_rejected() -> None:
    """Invalid CIK values should fail."""
    with pytest.raises(
        SECCompanyFactsValidationError,
    ):
        normalize_cik("NOT-A-CIK")


def test_companyfacts_url_uses_normalized_cik() -> None:
    """Company Facts URL should use a ten-digit CIK."""
    url = build_companyfacts_url(
        320193,
        base_url=("https://data.sec.gov/api/xbrl/companyfacts"),
    )

    assert url.endswith("/CIK0000320193.json")


def test_user_agent_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automated SEC access requires configured identification."""
    monkeypatch.delenv(
        "SEC_USER_AGENT",
        raising=False,
    )

    with pytest.raises(
        SECConfigurationError,
        match="User-Agent",
    ):
        resolve_sec_user_agent(make_config())


def test_download_is_cached(
    tmp_path,
) -> None:
    """A second request should reuse the raw cache."""
    session = FakeSession(
        [
            FakeResponse(
                200,
                make_payload(),
            )
        ]
    )

    client = SECCompanyFactsClient(
        make_config(),
        user_agent=("TestProject test@example.com"),
        session=session,
    )

    first = download_companyfacts_for_company(
        "AAPL",
        320193,
        client=client,
        raw_root=tmp_path,
    )

    second = download_companyfacts_for_company(
        "AAPL",
        320193,
        client=client,
        raw_root=tmp_path,
    )

    assert first.source == "download"
    assert second.source == "cache"

    assert first.path.exists()

    assert len(session.calls) == 1


def test_mismatched_cik_is_rejected() -> None:
    """Payloads from another company should be rejected."""
    session = FakeSession(
        [
            FakeResponse(
                200,
                make_payload(
                    cik=789019,
                ),
            )
        ]
    )

    client = SECCompanyFactsClient(
        make_config(),
        user_agent=("TestProject test@example.com"),
        session=session,
    )

    with pytest.raises(
        SECCompanyFactsValidationError,
        match="does not match",
    ):
        client.fetch_companyfacts(320193)


def test_retryable_http_error_is_retried() -> None:
    """Temporary SEC errors should be retried."""
    session = FakeSession(
        [
            FakeResponse(
                429,
                {},
            ),
            FakeResponse(
                200,
                make_payload(),
            ),
        ]
    )

    client = SECCompanyFactsClient(
        make_config(),
        user_agent=("TestProject test@example.com"),
        session=session,
    )

    payload, _ = client.fetch_companyfacts(320193)

    assert payload["entityName"] == "Apple Inc."

    assert len(session.calls) == 2
