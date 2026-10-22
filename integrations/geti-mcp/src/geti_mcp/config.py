# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Operator-supplied configuration.

Every value here comes from process startup (CLI arguments or environment) and is
immutable for the process lifetime. Nothing in this module may be influenced by tool
arguments, which are model-controlled and therefore untrusted.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

ENV_PREFIX = "GETI_MCP_"

#: Geti API contract versions (``info.version`` major.minor) this adapter was tested against.
SUPPORTED_API_VERSIONS: tuple[str, ...] = ("3.2",)

DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_PAGE_SIZE = 50
DEFAULT_MAX_ITEMS = 200
DEFAULT_MAX_WAIT_SECONDS = 20.0
DEFAULT_PREVIEW_MAX_DIMENSION = 768
DEFAULT_PREVIEW_MAX_BYTES = 1024 * 1024

#: Hard ceiling from the Geti media endpoint; a larger page size is rejected by the backend.
BACKEND_MAX_PAGE_SIZE = 100
#: Upper bound the operator may configure for ``wait_for_job``, to keep MCP calls short.
MAX_CONFIGURABLE_WAIT_SECONDS = 120.0


class ConfigError(ValueError):
    """Raised when operator-supplied configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Permissions:
    """Opt-in capability switches. Everything defaults to off (read-only)."""

    allow_training: bool = False
    allow_job_cancellation: bool = False
    allow_image_access: bool = False


@dataclass(frozen=True, slots=True)
class Limits:
    """Bounds applied to every outbound request and every returned payload."""

    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    max_page_size: int = DEFAULT_MAX_PAGE_SIZE
    max_items: int = DEFAULT_MAX_ITEMS
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS
    preview_max_dimension: int = DEFAULT_PREVIEW_MAX_DIMENSION
    preview_max_bytes: int = DEFAULT_PREVIEW_MAX_BYTES


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """The complete, immutable configuration of one MCP server process."""

    base_url: str
    allowed_projects: frozenset[str] | None
    permissions: Permissions
    limits: Limits
    ca_bundle: Path | None
    log_level: str

    @property
    def all_projects_allowed(self) -> bool:
        """Whether the operator opted into every project on the instance."""
        return self.allowed_projects is None


def _env(name: str) -> str | None:
    value = os.environ.get(ENV_PREFIX + name)
    return value.strip() if value and value.strip() else None


def _env_flag(name: str) -> bool:
    value = _env(name)
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def validate_base_url(raw: str) -> str:
    """Validate and normalise the Geti base URL.

    Args:
        raw: The operator-supplied URL.

    Returns:
        str: Scheme, host, port and path prefix, with no trailing slash.

    Raises:
        ConfigError: If the scheme is unsupported, the host is missing, or the URL
            embeds credentials, a query, or a fragment.
    """
    parts = urlsplit(raw.strip())
    if parts.scheme not in {"http", "https"}:
        raise ConfigError(f"Base URL must use http or https, got '{parts.scheme or raw}'.")
    if not parts.hostname:
        raise ConfigError("Base URL must include a host.")
    if parts.username or parts.password:
        raise ConfigError("Base URL must not embed credentials; Geti does not use HTTP authentication.")
    if parts.query or parts.fragment:
        raise ConfigError("Base URL must not include a query string or fragment.")
    return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"


def _parse_projects(raw: str) -> frozenset[str]:
    ids: set[str] = set()
    for chunk in raw.replace("\n", ",").split(","):
        candidate = chunk.strip()
        if not candidate:
            continue
        try:
            ids.add(str(UUID(candidate)))
        except ValueError as exc:
            raise ConfigError(f"Project allowlist entry '{candidate}' is not a valid UUID.") from exc
    if not ids:
        raise ConfigError("Project allowlist is empty; pass --all-projects to authorize every project.")
    return frozenset(ids)


def _validate_ca_bundle(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_file():
        raise ConfigError(f"Trusted certificate file '{path}' does not exist or is not a file.")
    return path


def _bounded(name: str, value: float, *, minimum: float, maximum: float) -> float:
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}, got {value}.")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the startup argument parser.

    Returns:
        argparse.ArgumentParser: Parser whose defaults are taken from the environment.
    """
    parser = argparse.ArgumentParser(
        prog="geti-mcp",
        description=(
            "MCP server exposing a read-only-by-default view of a running Geti instance over stdio. "
            "Geti must already be running; this server never starts, migrates, or resets it."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=_env("BASE_URL"),
        help=f"Base URL of the running Geti instance, e.g. https://localhost:7860 (env {ENV_PREFIX}BASE_URL).",
    )

    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--projects",
        default=_env("PROJECTS"),
        help=f"Comma-separated allowlist of project UUIDs the assistant may access (env {ENV_PREFIX}PROJECTS).",
    )
    scope.add_argument(
        "--all-projects",
        action="store_true",
        default=_env_flag("ALL_PROJECTS"),
        help=f"Authorize every project on the instance (env {ENV_PREFIX}ALL_PROJECTS).",
    )

    parser.add_argument(
        "--allow-training",
        action="store_true",
        default=_env_flag("ALLOW_TRAINING"),
        help=f"Permit submitting training jobs (env {ENV_PREFIX}ALLOW_TRAINING).",
    )
    parser.add_argument(
        "--allow-job-cancellation",
        action="store_true",
        default=_env_flag("ALLOW_JOB_CANCELLATION"),
        help=f"Permit requesting job cancellation (env {ENV_PREFIX}ALLOW_JOB_CANCELLATION).",
    )
    parser.add_argument(
        "--allow-image-access",
        action="store_true",
        default=_env_flag("ALLOW_IMAGE_ACCESS"),
        help=(
            "Permit returning image previews. Previews leave this machine through the assistant "
            f"provider (env {ENV_PREFIX}ALLOW_IMAGE_ACCESS)."
        ),
    )

    parser.add_argument(
        "--ca-bundle",
        default=_env("CA_BUNDLE"),
        help=(
            "Path to a PEM certificate or CA bundle to trust in addition to the system store. "
            f"Hostname verification always stays enabled (env {ENV_PREFIX}CA_BUNDLE)."
        ),
    )

    parser.add_argument(
        "--connect-timeout", type=float, default=float(_env("CONNECT_TIMEOUT") or DEFAULT_CONNECT_TIMEOUT)
    )
    parser.add_argument(
        "--request-timeout", type=float, default=float(_env("REQUEST_TIMEOUT") or DEFAULT_REQUEST_TIMEOUT)
    )
    parser.add_argument(
        "--max-response-bytes", type=int, default=int(_env("MAX_RESPONSE_BYTES") or DEFAULT_MAX_RESPONSE_BYTES)
    )
    parser.add_argument("--max-page-size", type=int, default=int(_env("MAX_PAGE_SIZE") or DEFAULT_MAX_PAGE_SIZE))
    parser.add_argument("--max-items", type=int, default=int(_env("MAX_ITEMS") or DEFAULT_MAX_ITEMS))
    parser.add_argument(
        "--max-wait-seconds", type=float, default=float(_env("MAX_WAIT_SECONDS") or DEFAULT_MAX_WAIT_SECONDS)
    )
    parser.add_argument(
        "--preview-max-dimension",
        type=int,
        default=int(_env("PREVIEW_MAX_DIMENSION") or DEFAULT_PREVIEW_MAX_DIMENSION),
    )
    parser.add_argument(
        "--preview-max-bytes", type=int, default=int(_env("PREVIEW_MAX_BYTES") or DEFAULT_PREVIEW_MAX_BYTES)
    )
    parser.add_argument(
        "--log-level",
        default=_env("LOG_LEVEL") or "INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log verbosity. Logs always go to stderr; stdout carries only MCP protocol messages.",
    )
    return parser


def config_from_args(argv: list[str] | None = None) -> ServerConfig:
    """Parse startup arguments into a validated configuration.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        ServerConfig: The validated configuration.

    Raises:
        ConfigError: If required values are missing or any value is out of range.
    """
    args = build_parser().parse_args(argv)

    if not args.base_url:
        raise ConfigError(f"--base-url is required (or set {ENV_PREFIX}BASE_URL).")
    if not args.projects and not args.all_projects:
        raise ConfigError(
            "A project access policy is required: pass --projects with an allowlist of project UUIDs, "
            "or --all-projects to authorize every project on the instance."
        )

    limits = Limits(
        connect_timeout=_bounded("--connect-timeout", args.connect_timeout, minimum=0.1, maximum=120.0),
        request_timeout=_bounded("--request-timeout", args.request_timeout, minimum=0.5, maximum=600.0),
        max_response_bytes=int(
            _bounded("--max-response-bytes", args.max_response_bytes, minimum=1024, maximum=64 * 1024 * 1024)
        ),
        max_page_size=int(_bounded("--max-page-size", args.max_page_size, minimum=1, maximum=BACKEND_MAX_PAGE_SIZE)),
        max_items=int(_bounded("--max-items", args.max_items, minimum=1, maximum=1000)),
        max_wait_seconds=_bounded(
            "--max-wait-seconds", args.max_wait_seconds, minimum=1.0, maximum=MAX_CONFIGURABLE_WAIT_SECONDS
        ),
        preview_max_dimension=int(
            _bounded("--preview-max-dimension", args.preview_max_dimension, minimum=64, maximum=4096)
        ),
        preview_max_bytes=int(
            _bounded("--preview-max-bytes", args.preview_max_bytes, minimum=4096, maximum=16 * 1024 * 1024)
        ),
    )

    return ServerConfig(
        base_url=validate_base_url(args.base_url),
        allowed_projects=None if args.all_projects else _parse_projects(args.projects),
        permissions=Permissions(
            allow_training=bool(args.allow_training),
            allow_job_cancellation=bool(args.allow_job_cancellation),
            allow_image_access=bool(args.allow_image_access),
        ),
        limits=limits,
        ca_bundle=_validate_ca_bundle(args.ca_bundle) if args.ca_bundle else None,
        log_level=args.log_level,
    )
