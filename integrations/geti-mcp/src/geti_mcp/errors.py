# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Stable error codes and the exception type surfaced to MCP clients."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError

# Absolute filesystem paths that may appear in backend error messages.
_POSIX_PATH = re.compile(r"(?<![\w.])/(?:[\w.\-+@]+/){1,}[\w.\-+@]*")
_WINDOWS_PATH = re.compile(r"[A-Za-z]:\\(?:[\w.\-+@ ]+\\)*[\w.\-+@ ]*")
_URL_CREDENTIALS = re.compile(r"(?P<scheme>[a-zA-Z][\w+.-]*://)[^/@\s]+@")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(authorization|api[_-]?key|token|secret|password|passwd|cookie)\b\s*[:=]\s*"
    r"(?:(?:bearer|basic|token|digest)\s+)?\S+"
)

MAX_ERROR_MESSAGE_CHARS = 600


class ErrorCode(StrEnum):
    """Stable, machine-readable failure codes.

    These strings are part of the public tool contract: clients and prompts may branch on
    them, so they must not be renamed without a version bump.
    """

    PERMISSION_DENIED = "permission_denied"
    """The operation is outside the operator-configured authorization scope."""

    INCOMPATIBLE_BACKEND = "incompatible_backend"
    """The Geti instance does not advertise a tested-compatible API contract."""

    BACKEND_UNAVAILABLE = "backend_unavailable"
    """Geti could not be reached, or the transport failed before a response arrived."""

    BACKEND_ERROR = "backend_error"
    """Geti responded, but with an error or an unusable payload."""

    NOT_FOUND = "not_found"
    """The requested entity does not exist (or is no longer retained)."""

    INVALID_INPUT = "invalid_input"
    """The tool arguments were rejected locally or by Geti."""

    OWNERSHIP_UNKNOWN = "ownership_unknown"
    """The owning project of a job could not be established, so access fails closed."""

    LIMIT_EXCEEDED = "limit_exceeded"
    """A configured response-size, page-size, or dimension bound was exceeded."""

    UNSUPPORTED_MEDIA = "unsupported_media"
    """The media item cannot be previewed in a format MCP can carry."""

    SUBMISSION_OUTCOME_UNKNOWN = "submission_outcome_unknown"
    """A non-idempotent submission may or may not have reached Geti. Never retried."""


class GetiMcpError(Exception):
    """A tool failure carrying a stable code and actionable guidance.

    Args:
        code: The stable error code.
        message: Human-readable explanation. Must already be free of secrets.
        guidance: Concrete next step for the caller, when one exists.
        details: Additional non-sensitive structured context.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        guidance: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.guidance = guidance
        self.details = details or {}
        super().__init__(message)

    def to_tool_error(self) -> ToolError:
        """Render this failure as the exception the MCP SDK reports to the client.

        Returns:
            ToolError: Carries a compact JSON payload so clients can branch on ``code``.
        """
        payload: dict[str, Any] = {"error_code": str(self.code), "message": self.message}
        if self.guidance:
            payload["guidance"] = self.guidance
        if self.details:
            payload["details"] = self.details
        return ToolError(json.dumps(payload, sort_keys=True))


def redact(text: str, *, max_chars: int = MAX_ERROR_MESSAGE_CHARS) -> str:
    """Strip credentials and filesystem paths from text that came from Geti.

    Backend messages are untrusted input that may embed absolute paths or, in principle,
    credential-bearing URLs. They must never be relayed verbatim into model context.

    Args:
        text: The raw message.
        max_chars: Maximum length of the redacted result.

    Returns:
        str: The redacted, length-bounded message.
    """
    redacted = _URL_CREDENTIALS.sub(r"\g<scheme>[redacted]@", text)
    redacted = _SECRET_ASSIGNMENT.sub(lambda m: f"{m.group(1)}=[redacted]", redacted)
    redacted = _WINDOWS_PATH.sub("[path]", redacted)
    redacted = _POSIX_PATH.sub("[path]", redacted)
    redacted = " ".join(redacted.split())
    if len(redacted) > max_chars:
        redacted = redacted[: max_chars - 1].rstrip() + "…"
    return redacted
