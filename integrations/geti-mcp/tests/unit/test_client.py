# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the REST transport: TLS, bounds, redirects, and failure classification."""

from __future__ import annotations

import ssl
from dataclasses import replace

import httpx
import pytest

from geti_mcp.client import GetiClient, _build_ssl_context
from geti_mcp.config import Limits
from geti_mcp.errors import ErrorCode, GetiMcpError
from tests.conftest import FakeGeti, make_config


def _client(fake: FakeGeti, **config_kwargs) -> GetiClient:
    return GetiClient(make_config(**config_kwargs), transport=fake.transport())


class TestTls:
    def test_context_requires_certificates_and_hostname(self) -> None:
        context = _build_ssl_context(make_config())
        assert context.verify_mode is ssl.CERT_REQUIRED
        assert context.check_hostname is True

    def test_real_client_uses_a_verifying_context(self) -> None:
        """No configuration path produces an unverified client."""
        client = GetiClient(make_config())
        # Reach into httpx internals deliberately: the point is to prove the context the
        # transport will actually present, not the one the helper returned.
        pool = getattr(client._client._transport, "_pool")
        context = pool._ssl_context
        assert context.verify_mode is ssl.CERT_REQUIRED
        assert context.check_hostname is True

    def test_unloadable_ca_bundle_fails_with_a_clear_error(self, tmp_path) -> None:
        bundle = tmp_path / "bad.pem"
        bundle.write_text("not a certificate")
        config = replace(make_config(), ca_bundle=bundle)
        with pytest.raises(GetiMcpError) as excinfo:
            _build_ssl_context(config)
        assert excinfo.value.code is ErrorCode.INVALID_INPUT
        assert "--ca-bundle" in (excinfo.value.guidance or "")


class TestResponseBounds:
    async def test_oversized_json_is_refused(self, fake_geti: FakeGeti) -> None:
        fake_geti.override(r"/api/projects$", lambda _: httpx.Response(200, json=[{"pad": "x" * 100_000}]))
        async with _client(fake_geti, limits=Limits(max_response_bytes=1024)) as client:
            with pytest.raises(GetiMcpError) as excinfo:
                await client.request_json("GET", "/api/projects")
        assert excinfo.value.code is ErrorCode.LIMIT_EXCEEDED

    async def test_declared_content_length_is_refused_without_downloading(self, fake_geti: FakeGeti) -> None:
        fake_geti.override(
            r"/binary$",
            lambda _: httpx.Response(200, content=b"x" * 5000, headers={"content-type": "image/jpeg"}),
        )
        async with _client(fake_geti) as client:
            with pytest.raises(GetiMcpError) as excinfo:
                await client.request_bytes("GET", "/api/x/binary", max_bytes=100, accept="image/*")
        assert excinfo.value.code is ErrorCode.LIMIT_EXCEEDED

    async def test_response_within_budget_is_returned(self, fake_geti: FakeGeti) -> None:
        async with _client(fake_geti) as client:
            payload = await client.request_json("GET", "/api/projects")
        assert isinstance(payload, list)


class TestRedirects:
    async def test_redirects_are_refused_not_followed(self, fake_geti: FakeGeti) -> None:
        fake_geti.override(
            r"/api/projects$",
            lambda _: httpx.Response(302, headers={"location": "https://evil.test/api/projects"}),
        )
        async with _client(fake_geti) as client:
            with pytest.raises(GetiMcpError) as excinfo:
                await client.request_json("GET", "/api/projects")
        assert excinfo.value.code is ErrorCode.BACKEND_ERROR
        assert "evil.test" not in excinfo.value.message
        assert len(fake_geti.requests) == 1


class TestFailureClassification:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (404, ErrorCode.NOT_FOUND),
            (400, ErrorCode.INVALID_INPUT),
            (409, ErrorCode.INVALID_INPUT),
            (422, ErrorCode.INVALID_INPUT),
            (500, ErrorCode.BACKEND_ERROR),
            (503, ErrorCode.BACKEND_ERROR),
        ],
    )
    async def test_http_status_maps_to_a_stable_code(
        self, fake_geti: FakeGeti, status: int, expected: ErrorCode
    ) -> None:
        fake_geti.override(r"/api/projects$", lambda _: httpx.Response(status, json={"detail": "nope"}))
        async with _client(fake_geti) as client:
            with pytest.raises(GetiMcpError) as excinfo:
                await client.request_json("GET", "/api/projects")
        assert excinfo.value.code is expected

    async def test_unreachable_backend_is_distinguished_from_a_backend_error(self, fake_geti: FakeGeti) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused", request=request)

        fake_geti.override(r"/api/projects$", refuse)
        async with _client(fake_geti) as client:
            with pytest.raises(GetiMcpError) as excinfo:
                await client.request_json("GET", "/api/projects")
        assert excinfo.value.code is ErrorCode.BACKEND_UNAVAILABLE

    async def test_interrupted_submission_is_reported_as_ambiguous(self, fake_geti: FakeGeti) -> None:
        """A POST that fails in transport may or may not have been applied."""

        def timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        fake_geti.override(r"/api/jobs$", timeout)
        async with _client(fake_geti) as client:
            with pytest.raises(GetiMcpError) as excinfo:
                await client.request_json("POST", "/api/jobs", json_body={"job_type": "train"})
        assert excinfo.value.code is ErrorCode.SUBMISSION_OUTCOME_UNKNOWN
        assert "resubmit" in (excinfo.value.guidance or "").lower()

    async def test_interrupted_read_is_not_ambiguous(self, fake_geti: FakeGeti) -> None:
        def timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        fake_geti.override(r"/api/projects$", timeout)
        async with _client(fake_geti) as client:
            with pytest.raises(GetiMcpError) as excinfo:
                await client.request_json("GET", "/api/projects")
        assert excinfo.value.code is ErrorCode.BACKEND_UNAVAILABLE

    async def test_backend_detail_is_redacted(self, fake_geti: FakeGeti) -> None:
        fake_geti.override(
            r"/api/projects$",
            lambda _: httpx.Response(500, json={"detail": "failed opening /Users/alice/geti/data/geti.db"}),
        )
        async with _client(fake_geti) as client:
            with pytest.raises(GetiMcpError) as excinfo:
                await client.request_json("GET", "/api/projects")
        assert "alice" not in str(excinfo.value.details)


class TestContract:
    async def test_reports_the_backend_api_version(self, fake_geti: FakeGeti) -> None:
        async with _client(fake_geti) as client:
            contract = await client.get_contract()
        assert contract.api_version == "3.2.0"
        assert contract.compatible is True
        assert contract.fingerprint.startswith("sha256:")

    async def test_untested_version_is_flagged_not_blocked(self, fake_geti: FakeGeti) -> None:
        fake_geti.api_version = "4.0.0"
        async with _client(fake_geti) as client:
            contract = await client.get_contract()
        assert contract.compatible is False

    async def test_contract_is_fetched_once(self, fake_geti: FakeGeti) -> None:
        async with _client(fake_geti) as client:
            await client.get_contract()
            await client.get_contract()
        assert sum(1 for r in fake_geti.requests if r.url.path.endswith("openapi.json")) == 1

    def test_base_url_is_sanitized_for_display(self, fake_geti: FakeGeti) -> None:
        assert _client(fake_geti).sanitized_base_url == "https://geti.test:7860"
