# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Thin async REST client for the Geti application API.

This module owns transport concerns only: TLS trust, timeouts, redirect refusal,
response-size bounds, and translation of HTTP failures into :class:`GetiMcpError`.
It performs no authorization; that is :mod:`geti_mcp.authz`.
"""

from __future__ import annotations

import json
import logging
import ssl
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import httpx

from geti_mcp.config import SUPPORTED_API_VERSIONS, ServerConfig
from geti_mcp.errors import ErrorCode, GetiMcpError, redact

logger = logging.getLogger(__name__)

API_PREFIX = "/api"

#: Endpoints whose failure mid-flight leaves the outcome genuinely ambiguous.
_NON_IDEMPOTENT_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Statuses that mean "the request was wrong", as opposed to "the backend is broken".
_CLIENT_INPUT_STATUSES = frozenset({400, 409, 422})


@dataclass(frozen=True, slots=True)
class BackendContract:
    """Identity of the Geti contract observed at connection time.

    Attributes:
        api_version: ``info.version`` from the served OpenAPI document.
        fingerprint: Short digest over the sorted set of ``METHOD path`` pairs. Two
            instances with the same fingerprint expose the same operation surface.
        operation_count: Number of operations in the served document.
        compatible: Whether ``api_version`` falls in the tested compatibility range.
    """

    api_version: str
    fingerprint: str
    operation_count: int
    compatible: bool


def _build_ssl_context(config: ServerConfig) -> ssl.SSLContext:
    """Create an SSL context using system trust, optionally extended with one CA file.

    Hostname verification and certificate verification always stay enabled. There is no
    option to disable them, and certificates are never fetched from the network.

    Args:
        config: The server configuration.

    Returns:
        ssl.SSLContext: Context used for every outbound HTTPS request.

    Raises:
        GetiMcpError: If the configured CA bundle cannot be loaded.
    """
    context = ssl.create_default_context()
    if config.ca_bundle is not None:
        try:
            context.load_verify_locations(cafile=str(config.ca_bundle))
        except (ssl.SSLError, OSError) as exc:
            raise GetiMcpError(
                ErrorCode.INVALID_INPUT,
                f"The configured trusted certificate file could not be loaded: {type(exc).__name__}.",
                guidance="Ask the operator to point --ca-bundle at a valid PEM certificate file.",
            ) from exc
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    return context


class GetiClient:
    """Bounded async HTTP client for one Geti instance."""

    def __init__(self, config: ServerConfig, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._config = config
        self._limits = config.limits
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.limits.request_timeout, connect=config.limits.connect_timeout),
            # Geti is a fixed operator-chosen origin: a redirect would move the request to an
            # unvetted destination, so refuse rather than follow.
            follow_redirects=False,
            verify=_build_ssl_context(config),
            transport=transport,
            headers={"Accept": "application/json"},
        )
        self._contract: BackendContract | None = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()

    @property
    def sanitized_base_url(self) -> str:
        """The configured base URL, which is validated at startup to carry no credentials."""
        return self._config.base_url

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        """Perform a request and decode a bounded JSON response.

        Args:
            method: HTTP method.
            path: Path below the base URL, e.g. ``/api/projects``.
            params: Query parameters; ``None`` values are dropped.
            json_body: Request body to serialize as JSON.

        Returns:
            Any: The decoded JSON payload, or ``None`` for ``204 No Content``.

        Raises:
            GetiMcpError: For transport failures, refused redirects, oversized payloads,
                non-2xx responses, or undecodable bodies.
        """
        body, status_code = await self._send(method, path, params=params, json_body=json_body)
        if status_code == httpx.codes.NO_CONTENT or not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise GetiMcpError(
                ErrorCode.BACKEND_ERROR,
                f"Geti returned a non-JSON body for {method} {path}.",
                guidance="Confirm the base URL points at the Geti API and not at a proxy or the web UI.",
            ) from exc

    async def request_bytes(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        max_bytes: int,
        accept: str,
    ) -> tuple[bytes, str]:
        """Perform a request and return a bounded binary payload with its media type.

        Args:
            method: HTTP method.
            path: Path below the base URL.
            params: Query parameters.
            max_bytes: Hard cap on the number of bytes read.
            accept: Value for the ``Accept`` header.

        Returns:
            tuple[bytes, str]: The payload and its ``Content-Type`` (parameters stripped).

        Raises:
            GetiMcpError: As for :meth:`request_json`, plus ``LIMIT_EXCEEDED`` when the
                payload is larger than ``max_bytes``.
        """
        body, _, content_type = await self._send_raw(
            method, path, params=params, json_body=None, max_bytes=max_bytes, accept=accept
        )
        return body, content_type

    async def _send(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
    ) -> tuple[bytes, int]:
        body, status_code, _ = await self._send_raw(
            method,
            path,
            params=params,
            json_body=json_body,
            max_bytes=self._limits.max_response_bytes,
            accept="application/json",
        )
        return body, status_code

    async def _send_raw(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        max_bytes: int,
        accept: str,
    ) -> tuple[bytes, int, str]:
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        request = self._client.build_request(
            method, path, params=clean_params, json=json_body, headers={"Accept": accept}
        )
        try:
            response = await self._client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise self._transport_error(method, path, "timed out", exc) from exc
        except httpx.TransportError as exc:
            raise self._transport_error(method, path, "failed", exc) from exc

        try:
            body = await self._read_bounded(response, method=method, path=path, max_bytes=max_bytes)
        except httpx.TimeoutException as exc:
            raise self._transport_error(method, path, "timed out while streaming the response", exc) from exc
        except httpx.TransportError as exc:
            raise self._transport_error(method, path, "failed while streaming the response", exc) from exc
        finally:
            await response.aclose()

        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if response.is_redirect:
            raise GetiMcpError(
                ErrorCode.BACKEND_ERROR,
                f"Geti redirected {method} {path}; redirects are not followed.",
                guidance="Point --base-url directly at the Geti API origin rather than at a redirecting front end.",
                details={"status_code": response.status_code},
            )
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise self._http_error(response.status_code, body, method=method, path=path)
        return body, response.status_code, content_type

    async def _read_bounded(self, response: httpx.Response, *, method: str, path: str, max_bytes: int) -> bytes:
        declared = response.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > max_bytes:
            raise GetiMcpError(
                ErrorCode.LIMIT_EXCEEDED,
                f"Geti response for {method} {path} is {int(declared)} bytes, over the {max_bytes} byte limit.",
                guidance="Narrow the request (smaller page size or a more specific filter), "
                "or raise --max-response-bytes if the operator considers it safe.",
            )
        buffer = bytearray()
        async for chunk in response.aiter_bytes():
            buffer.extend(chunk)
            if len(buffer) > max_bytes:
                raise GetiMcpError(
                    ErrorCode.LIMIT_EXCEEDED,
                    f"Geti response for {method} {path} exceeded the {max_bytes} byte limit.",
                    guidance="Narrow the request, or raise --max-response-bytes if the operator considers it safe.",
                )
        return bytes(buffer)

    def _transport_error(self, method: str, path: str, what: str, exc: Exception) -> GetiMcpError:
        # A failed non-idempotent request may still have been applied by Geti. Say so rather
        # than implying nothing happened, and never retry it automatically.
        if method.upper() in _NON_IDEMPOTENT_METHODS:
            return GetiMcpError(
                ErrorCode.SUBMISSION_OUTCOME_UNKNOWN,
                f"The connection for {method} {path} {what}, after the request may already have reached Geti.",
                guidance=(
                    "Do not resubmit. List the relevant jobs and check whether one was created just now; "
                    "resubmit only after confirming none exists."
                ),
                details={"transport_error": redact(str(exc))},
            )
        return GetiMcpError(
            ErrorCode.BACKEND_UNAVAILABLE,
            f"Request {method} {path} {what}.",
            guidance="Confirm Geti is running and reachable at the configured base URL, and that its "
            "TLS certificate is trusted (see --ca-bundle).",
            details={"transport_error": redact(str(exc))},
        )

    def _http_error(self, status_code: int, body: bytes, *, method: str, path: str) -> GetiMcpError:
        detail = _extract_detail(body)
        if status_code == httpx.codes.NOT_FOUND:
            return GetiMcpError(
                ErrorCode.NOT_FOUND,
                f"Geti has no entity for {method} {path}.",
                guidance="Verify the identifier. Jobs are not guaranteed to survive a Geti restart.",
                details={"status_code": status_code, "backend_detail": detail},
            )
        if status_code in _CLIENT_INPUT_STATUSES:
            return GetiMcpError(
                ErrorCode.INVALID_INPUT,
                f"Geti rejected {method} {path} with status {status_code}.",
                guidance="Backend validation is authoritative; correct the inputs it named and try again.",
                details={"status_code": status_code, "backend_detail": detail},
            )
        return GetiMcpError(
            ErrorCode.BACKEND_ERROR,
            f"Geti returned status {status_code} for {method} {path}.",
            guidance="Check the Geti server logs for the corresponding failure.",
            details={"status_code": status_code, "backend_detail": detail},
        )

    async def get_contract(self) -> BackendContract:
        """Fetch and cache the served OpenAPI contract identity.

        Geti exposes no dedicated version endpoint, so the served OpenAPI document is the
        only authoritative source of both its version and its operation surface.

        Returns:
            BackendContract: Version, fingerprint, operation count, and compatibility.

        Raises:
            GetiMcpError: If the document cannot be fetched or lacks a version.
        """
        if self._contract is not None:
            return self._contract

        document = await self.request_json("GET", f"{API_PREFIX}/openapi.json")
        if not isinstance(document, dict):
            raise GetiMcpError(
                ErrorCode.INCOMPATIBLE_BACKEND,
                "The OpenAPI document served by Geti is not a JSON object.",
                guidance="Confirm --base-url points at a Geti instance.",
            )
        version = str(document.get("info", {}).get("version", "")).strip()
        if not version:
            raise GetiMcpError(
                ErrorCode.INCOMPATIBLE_BACKEND,
                "The OpenAPI document served by Geti declares no version.",
                guidance="Confirm --base-url points at a supported Geti instance.",
            )
        operations = sorted(
            f"{method.upper()} {path}"
            for path, methods in (document.get("paths") or {}).items()
            if isinstance(methods, dict)
            for method in methods
            if method.lower() in {"get", "post", "put", "patch", "delete"}
        )
        self._contract = BackendContract(
            api_version=version,
            fingerprint=_fingerprint(operations),
            operation_count=len(operations),
            compatible=_major_minor(version) in SUPPORTED_API_VERSIONS,
        )
        return self._contract


def _major_minor(version: str) -> str:
    parts = version.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else version


def _fingerprint(operations: list[str]) -> str:
    import hashlib

    digest = hashlib.sha256("\n".join(operations).encode("utf-8")).hexdigest()
    return f"sha256:{digest[:16]}"


def _extract_detail(body: bytes) -> str | None:
    """Pull a redacted, bounded ``detail`` out of a FastAPI error body."""
    if not body:
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return redact(body.decode("utf-8", errors="replace"), max_chars=200)
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if detail is None:
        return None
    return redact(detail if isinstance(detail, str) else json.dumps(detail), max_chars=400)
