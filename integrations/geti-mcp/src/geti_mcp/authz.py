# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Authorization and bounds enforcement.

Every project-scoped operation passes through :class:`Authorizer`. Enforcement lives here
rather than in the tool definitions so that a client invoking a tool directly — including
one the server never advertised — still cannot escape the operator's policy.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from geti_mcp.config import ServerConfig
from geti_mcp.errors import ErrorCode, GetiMcpError


class Authorizer:
    """Enforces the operator's project allowlist, permission opt-ins, and output bounds."""

    def __init__(self, config: ServerConfig) -> None:
        self._config = config

    @property
    def config(self) -> ServerConfig:
        """The configuration being enforced."""
        return self._config

    def parse_uuid(self, value: str, *, field: str) -> str:
        """Validate that a model-supplied identifier is a UUID.

        Args:
            value: The raw identifier.
            field: Field name used in the error message.

        Returns:
            str: The canonical lowercase UUID string.

        Raises:
            GetiMcpError: ``INVALID_INPUT`` if the value is not a UUID.
        """
        try:
            return str(UUID(str(value).strip()))
        except (ValueError, AttributeError) as exc:
            raise GetiMcpError(
                ErrorCode.INVALID_INPUT,
                f"'{field}' must be a UUID.",
                guidance="Use an identifier returned by list_projects, list_media, list_jobs, or list_models.",
            ) from exc

    def require_project(self, project_id: str) -> str:
        """Validate a project identifier and check it against the allowlist.

        Args:
            project_id: The project the caller wants to act on.

        Returns:
            str: The canonical project UUID.

        Raises:
            GetiMcpError: ``PERMISSION_DENIED`` if the project is outside the configured scope.
        """
        canonical = self.parse_uuid(project_id, field="project_id")
        allowed = self._config.allowed_projects
        # Fail closed: an absent allowlist only grants access when the operator explicitly
        # opted into all projects. Never rely on `assert`, which `python -O` strips.
        if not self._config.all_projects_allowed and (allowed is None or canonical not in allowed):
            raise GetiMcpError(
                ErrorCode.PERMISSION_DENIED,
                "This Geti MCP server is not authorized to access that project.",
                guidance=(
                    "Call list_projects to see the authorized projects. Widening the scope requires the "
                    "operator to restart the server with an updated --projects allowlist."
                ),
            )
        return canonical

    def is_project_allowed(self, project_id: str | None) -> bool:
        """Whether a project falls inside the configured scope, without raising.

        Args:
            project_id: Candidate project identifier, possibly ``None`` or malformed.

        Returns:
            bool: ``True`` only for a well-formed, authorized project.
        """
        if project_id is None:
            return False
        try:
            canonical = str(UUID(str(project_id)))
        except (ValueError, AttributeError):
            return False
        return self._config.all_projects_allowed or canonical in (self._config.allowed_projects or frozenset())

    def require_training(self) -> None:
        """Ensure training submission was opted into.

        Raises:
            GetiMcpError: ``PERMISSION_DENIED`` when training is not enabled.
        """
        if not self._config.permissions.allow_training:
            raise GetiMcpError(
                ErrorCode.PERMISSION_DENIED,
                "Training submission is disabled on this Geti MCP server.",
                guidance="The operator must restart the server with --allow-training to permit it.",
            )

    def require_job_cancellation(self) -> None:
        """Ensure job cancellation was opted into.

        Raises:
            GetiMcpError: ``PERMISSION_DENIED`` when cancellation is not enabled.
        """
        if not self._config.permissions.allow_job_cancellation:
            raise GetiMcpError(
                ErrorCode.PERMISSION_DENIED,
                "Job cancellation is disabled on this Geti MCP server.",
                guidance="The operator must restart the server with --allow-job-cancellation to permit it.",
            )

    def require_image_access(self) -> None:
        """Ensure image disclosure was opted into.

        Raises:
            GetiMcpError: ``PERMISSION_DENIED`` when image access is not enabled.
        """
        if not self._config.permissions.allow_image_access:
            raise GetiMcpError(
                ErrorCode.PERMISSION_DENIED,
                "Image previews are disabled on this Geti MCP server.",
                guidance=(
                    "Image content would leave this machine through the assistant provider. The operator "
                    "must restart the server with --allow-image-access to permit it."
                ),
            )

    def require_authorized_job(self, job: dict[str, Any]) -> str:
        """Resolve a job's owning project and authorize it.

        Args:
            job: A ``JobView`` payload from Geti.

        Returns:
            str: The owning project UUID.

        Raises:
            GetiMcpError: ``OWNERSHIP_UNKNOWN`` when the owning project cannot be
                established, or ``PERMISSION_DENIED`` when it is out of scope.
        """
        owner = owning_project_id(job)
        if owner is None:
            raise GetiMcpError(
                ErrorCode.OWNERSHIP_UNKNOWN,
                "The owning project of that job could not be determined, so access is refused.",
                guidance=(
                    "Jobs that are not bound to a single project — such as dataset staging and "
                    "import-as-new-project — are outside this server's scope."
                ),
            )
        return self.require_project(owner)

    def clamp_limit(self, requested: int | None) -> int:
        """Clamp a requested page size to the configured maximum.

        Args:
            requested: Model-supplied page size, or ``None`` for the default.

        Returns:
            int: A page size within ``[1, limits.max_page_size]``.
        """
        maximum = self._config.limits.max_page_size
        if requested is None:
            return min(20, maximum)
        return max(1, min(int(requested), maximum))

    def clamp_wait(self, requested: float | None) -> float:
        """Clamp a requested wait duration to the configured maximum.

        Args:
            requested: Model-supplied wait in seconds, or ``None`` for the maximum.

        Returns:
            float: A duration within ``[0.5, limits.max_wait_seconds]``.
        """
        maximum = self._config.limits.max_wait_seconds
        if requested is None:
            return maximum
        return max(0.5, min(float(requested), maximum))


def owning_project_id(job: dict[str, Any]) -> str | None:
    """Extract the project that owns a job, if the payload states one unambiguously.

    Training and quantization jobs carry ``metadata.project.id``. Dataset import jobs carry
    a nullable ``metadata.project_id``, and export/staging jobs may carry neither. Anything
    other than a single well-formed project identifier is reported as unknown so callers
    can fail closed.

    Args:
        job: A ``JobView`` payload from Geti.

    Returns:
        str | None: The canonical owning project UUID, or ``None`` if indeterminate.
    """
    metadata = job.get("metadata")
    if not isinstance(metadata, dict):
        return None
    candidate: Any = None
    project = metadata.get("project")
    if isinstance(project, dict):
        candidate = project.get("id")
    if candidate is None:
        candidate = metadata.get("project_id")
    if candidate is None:
        return None
    try:
        return str(UUID(str(candidate)))
    except (ValueError, AttributeError):
        return None
