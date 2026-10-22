# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Geti operations exposed by the MCP tools.

Each method authorizes first, then calls the REST API, then parses the response into a
bounded public schema. The MCP tool definitions in :mod:`geti_mcp.server` stay thin
wrappers so that authorization cannot be bypassed by calling a tool directly.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from collections.abc import Collection
from datetime import UTC, datetime
from typing import Any

from PIL import Image, UnidentifiedImageError

from geti_mcp.authz import Authorizer, owning_project_id
from geti_mcp.client import API_PREFIX, GetiClient
from geti_mcp.config import SUPPORTED_API_VERSIONS
from geti_mcp.errors import ErrorCode, GetiMcpError, redact
from geti_mcp.lifecycle import NormalizedJobState, is_terminal, normalize_status
from geti_mcp.schemas import (
    ArchitectureInfo,
    ArchitectureList,
    BenchmarkMetrics,
    ConfigurationEntry,
    ConnectionInfo,
    DatasetStatistics,
    DeviceInfo,
    EvaluationResult,
    JobCancellation,
    JobDetail,
    JobList,
    JobSubmission,
    JobSummary,
    JobWait,
    LabelInfo,
    LabelInstanceCount,
    MediaItem,
    MediaList,
    MediaPreview,
    MetricValue,
    ModelList,
    ModelResults,
    ModelSummary,
    ModelVariantSummary,
    PermissionInfo,
    ProjectDetail,
    ProjectList,
    ProjectScopeInfo,
    ProjectSummary,
    ResizeTransform,
    TrainingConfiguration,
)

logger = logging.getLogger(__name__)

#: Refuse to decode images beyond this many pixels, independent of byte size.
MAX_DECODED_PIXELS = 64_000_000

_PREVIEWABLE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/bmp", "image/webp", "image/tiff"})

#: Job types Geti 3.2 defines. Used to reject typos instead of silently returning nothing.
JOB_TYPES = (
    "train",
    "quantize",
    "prepare_dataset_for_import",
    "import_dataset_as_new_project",
    "import_dataset_to_project",
    "export_dataset",
    "stage_dataset",
)

_POLL_INITIAL_INTERVAL = 0.5
_POLL_MAX_INTERVAL = 3.0
_POLL_BACKOFF = 1.5

_COMPARISON_NOTE = (
    "Metrics computed on different dataset_revision_id or subset values are not comparable. "
    "Check that context before ranking models against each other."
)


class GetiService:
    """Implements every MCP tool against the Geti REST API."""

    def __init__(self, client: GetiClient, authorizer: Authorizer) -> None:
        self._client = client
        self._authz = authorizer
        self._limits = authorizer.config.limits

    # ------------------------------------------------------------------ diagnostics

    async def connection_info(self) -> ConnectionInfo:
        """Probe the instance and report what this server is permitted to do.

        Returns:
            ConnectionInfo: Reachability, contract identity, permissions, project scope
            and training devices. Sanitized; never includes certificates or headers.
        """
        config = self._authz.config
        permissions = PermissionInfo(
            allow_training=config.permissions.allow_training,
            allow_job_cancellation=config.permissions.allow_job_cancellation,
            allow_image_access=config.permissions.allow_image_access,
        )
        scope = ProjectScopeInfo(
            policy="all_projects" if config.all_projects_allowed else "allowlist",
            authorized_project_ids=None if config.all_projects_allowed else sorted(config.allowed_projects or ()),
        )
        warnings: list[str] = []

        try:
            contract = await self._client.get_contract()
        except GetiMcpError as exc:
            # This tool exists to diagnose connectivity, so an unreachable backend is a
            # reportable result rather than a tool failure.
            return ConnectionInfo(
                summary=f"Geti at {self._client.sanitized_base_url} is not reachable: {exc.message}",
                reachable=False,
                base_url=self._client.sanitized_base_url,
                compatible=False,
                tested_api_versions=list(SUPPORTED_API_VERSIONS),
                permissions=permissions,
                project_scope=scope,
                warnings=[exc.message] + ([exc.guidance] if exc.guidance else []),
            )

        if not contract.compatible:
            warnings.append(
                f"Geti reports API version {contract.api_version}; this adapter was tested against "
                f"{', '.join(SUPPORTED_API_VERSIONS)}. Read-only tools may still work, but training "
                "submission is refused."
            )

        devices: list[DeviceInfo] = []
        license_accepted: bool | None = None
        platform: str | None = None
        try:
            devices = await self._training_devices()
        except GetiMcpError as exc:
            warnings.append(f"Training devices could not be listed: {exc.message}")
        try:
            info = await self._client.request_json("GET", f"{API_PREFIX}/system/info")
            if isinstance(info, dict):
                license_accepted = info.get("license_accepted")
                platform = info.get("platform")
        except GetiMcpError as exc:
            warnings.append(f"System info could not be read: {exc.message}")

        if license_accepted is False:
            warnings.append("The Geti license has not been accepted; some operations will be refused by the backend.")

        enabled = [
            name
            for name, on in (
                ("training", permissions.allow_training),
                ("job_cancellation", permissions.allow_job_cancellation),
                ("image_access", permissions.allow_image_access),
            )
            if on
        ]
        return ConnectionInfo(
            summary=(
                f"Connected to Geti {contract.api_version} at {self._client.sanitized_base_url} "
                f"({'compatible' if contract.compatible else 'untested version'}). "
                f"Scope: {scope.policy}. Enabled beyond read-only: {', '.join(enabled) or 'nothing'}. "
                f"{len(devices)} training device(s)."
            ),
            reachable=True,
            base_url=self._client.sanitized_base_url,
            api_version=contract.api_version,
            contract_fingerprint=contract.fingerprint,
            compatible=contract.compatible,
            tested_api_versions=list(SUPPORTED_API_VERSIONS),
            permissions=permissions,
            project_scope=scope,
            training_devices=devices,
            license_accepted=license_accepted,
            platform=platform,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ projects

    async def list_projects(self) -> ProjectList:
        """List the projects inside the configured authorization scope.

        Returns:
            ProjectList: Authorized projects only. Projects outside the scope are omitted
            entirely rather than reported as denied.
        """
        payload = await self._client.request_json("GET", f"{API_PREFIX}/projects")
        raw = payload if isinstance(payload, list) else []
        authorized = [p for p in raw if isinstance(p, dict) and self._authz.is_project_allowed(p.get("id"))]
        limited = authorized[: self._limits.max_items]
        projects = [
            ProjectSummary(
                id=str(p.get("id")),
                name=str(p.get("name", "")),
                task_type=str((p.get("task") or {}).get("task_type", "unknown")),
                created_at=_iso(p.get("created_at")),
            )
            for p in limited
        ]
        return ProjectList(
            summary=f"{len(projects)} authorized project(s): " + (", ".join(p.name for p in projects) or "none"),
            projects=projects,
            total_authorized=len(authorized),
            truncated=len(authorized) > len(limited),
        )

    async def get_project(self, project_id: str) -> ProjectDetail:
        """Return one authorized project's task configuration and labels.

        Args:
            project_id: Project to describe.

        Returns:
            ProjectDetail: Task type, label set, and pipeline state.
        """
        canonical = self._authz.require_project(project_id)
        project = await self._get_project_payload(canonical)
        task = project.get("task") or {}
        labels = [
            LabelInfo(id=str(label.get("id")), name=str(label.get("name", "")), color=label.get("color"))
            for label in (task.get("labels") or [])
            if isinstance(label, dict)
        ][: self._limits.max_items]
        task_type = str(task.get("task_type", "unknown"))
        return ProjectDetail(
            summary=(
                f"Project '{project.get('name')}' is a {task_type} task with {len(labels)} label(s): "
                f"{', '.join(label.name for label in labels) or 'none'}."
            ),
            id=canonical,
            name=str(project.get("name", "")),
            task_type=task_type,
            exclusive_labels=bool(task.get("exclusive_labels", False)),
            labels=labels,
            active_pipeline=bool(project.get("active_pipeline", False)),
            created_at=_iso(project.get("created_at")),
        )

    # ------------------------------------------------------------------ dataset

    async def dataset_statistics(self, project_id: str) -> DatasetStatistics:
        """Return Geti's dataset statistics plus observations derived only from them.

        Args:
            project_id: Project whose dataset to summarize.

        Returns:
            DatasetStatistics: Counts as reported, and readiness warnings that are
            explicitly not a trainability verdict.
        """
        canonical = self._authz.require_project(project_id)
        project = await self._get_project_payload(canonical)
        label_names = {
            str(label.get("id")): str(label.get("name", ""))
            for label in ((project.get("task") or {}).get("labels") or [])
            if isinstance(label, dict)
        }

        payload = await self._client.request_json("GET", f"{API_PREFIX}/projects/{canonical}/dataset/statistics")
        stats = payload if isinstance(payload, dict) else {}
        media = stats.get("media_counts") or {}
        annotations = stats.get("annotations_counts") or {}

        per_label_raw = annotations.get("instances_per_label") or []
        per_label = [
            LabelInstanceCount(
                label_id=str(entry.get("label_id")) if entry.get("label_id") is not None else None,
                label_name=label_names.get(str(entry.get("label_id"))),
                instances=int(entry.get("instances", 0)),
            )
            for entry in per_label_raw
            if isinstance(entry, dict)
        ][: self._limits.max_items]

        images = int(media.get("images", 0))
        videos = int(media.get("videos", 0))
        video_frames = int(media.get("video_frames", 0))
        annotated_images = int(annotations.get("annotated_images", 0))
        annotated_videos = int(annotations.get("annotated_videos", 0))
        annotated_video_frames = int(annotations.get("annotated_video_frames", 0))
        total_instances = int(annotations.get("instances", 0))

        counted = {entry.label_id for entry in per_label if entry.instances > 0}
        unused = [name for label_id, name in label_names.items() if label_id not in counted]

        warnings: list[str] = []
        if images + videos == 0:
            warnings.append("The dataset contains no media.")
        if annotated_images + annotated_video_frames == 0:
            warnings.append("No annotated media were reported, so there is nothing to learn from yet.")
        if images > annotated_images:
            warnings.append(f"{images - annotated_images} of {images} image(s) have no annotations.")
        if unused:
            warnings.append(f"Label(s) with no annotated instances: {', '.join(sorted(unused))}.")
        warnings.append(
            "Geti does not publish a minimum dataset size; it validates trainability when a training "
            "job is submitted. Treat the above as observations, not as a decision."
        )

        return DatasetStatistics(
            summary=(
                f"{images} image(s), {videos} video(s) with {video_frames} frame(s); "
                f"{annotated_images} annotated image(s), {total_instances} annotated instance(s) "
                f"across {len(per_label)} label(s)."
            ),
            project_id=canonical,
            images=images,
            videos=videos,
            video_frames=video_frames,
            annotated_images=annotated_images,
            annotated_videos=annotated_videos,
            annotated_video_frames=annotated_video_frames,
            total_instances=total_instances,
            instances_per_label=per_label,
            readiness_warnings=warnings,
        )

    async def list_media(
        self,
        project_id: str,
        *,
        offset: int = 0,
        limit: int | None = None,
        annotation_status: str | None = None,
        subsets: list[str] | None = None,
        label_ids: list[str] | None = None,
    ) -> MediaList:
        """Return a bounded page of dataset media metadata. Never returns image bytes.

        Args:
            project_id: Project to list media from.
            offset: Zero-based offset of the first item.
            limit: Requested page size, clamped to the configured maximum.
            annotation_status: Optional ``with_annotations`` / ``missing_annotations`` filter.
            subsets: Optional subset filter.
            label_ids: Optional label filter.

        Returns:
            MediaList: One page of metadata plus pagination state.
        """
        canonical = self._authz.require_project(project_id)
        effective_limit = self._authz.clamp_limit(limit)
        if offset < 0:
            raise GetiMcpError(ErrorCode.INVALID_INPUT, "'offset' must be zero or greater.")

        params: dict[str, Any] = {"limit": effective_limit, "offset": offset}
        if annotation_status is not None:
            params["annotation_status"] = _require_choice(
                annotation_status, {"with_annotations", "missing_annotations"}, field="annotation_status"
            )
        if subsets:
            params["subsets"] = [
                _require_choice(s, {"unassigned", "training", "validation", "testing"}, field="subsets")
                for s in subsets
            ]
        if label_ids:
            params["labels"] = [self._authz.parse_uuid(label, field="label_ids") for label in label_ids]

        payload = await self._client.request_json(
            "GET", f"{API_PREFIX}/projects/{canonical}/dataset/media", params=params
        )
        data = payload if isinstance(payload, dict) else {}
        pagination = data.get("pagination") or {}
        items = [_media_item(entry) for entry in (data.get("items") or []) if isinstance(entry, dict)]
        total = int(pagination.get("total", len(items)))
        used_offset = int(pagination.get("offset", offset))
        next_offset = used_offset + len(items) if used_offset + len(items) < total else None

        return MediaList(
            summary=(
                f"{len(items)} media item(s) at offset {used_offset} of {total} total"
                + (f"; next_offset={next_offset}." if next_offset is not None else "; end of results.")
            ),
            project_id=canonical,
            items=items,
            offset=used_offset,
            limit=int(pagination.get("limit", effective_limit)),
            count=len(items),
            total=total,
            next_offset=next_offset,
        )

    async def view_media(self, project_id: str, media_id: str, frame_index: int | None = None) -> MediaPreview:
        """Return a downscaled preview of one image or video frame.

        Args:
            project_id: Project the media belongs to.
            media_id: Media item to preview.
            frame_index: Frame to extract, required for videos.

        Returns:
            MediaPreview: Base64 preview bytes plus the resize transform that produced them.

        Raises:
            GetiMcpError: ``PERMISSION_DENIED`` when image access is disabled,
                ``UNSUPPORTED_MEDIA`` for undecodable or non-image payloads, and
                ``LIMIT_EXCEEDED`` when the source exceeds the configured bounds.
        """
        self._authz.require_image_access()
        canonical = self._authz.require_project(project_id)
        media_uuid = self._authz.parse_uuid(media_id, field="media_id")
        if frame_index is not None and frame_index < 0:
            raise GetiMcpError(ErrorCode.INVALID_INPUT, "'frame_index' must be zero or greater.")

        raw, content_type = await self._client.request_bytes(
            "GET",
            f"{API_PREFIX}/projects/{canonical}/dataset/media/{media_uuid}/binary",
            params={"frame_index": frame_index},
            max_bytes=self._limits.max_response_bytes,
            accept="image/*",
        )
        if content_type and not content_type.startswith("image/"):
            raise GetiMcpError(
                ErrorCode.UNSUPPORTED_MEDIA,
                f"Media {media_uuid} returned '{content_type}', which cannot be previewed as an image.",
                guidance="For a video, pass frame_index to preview a single frame.",
            )
        if content_type and content_type not in _PREVIEWABLE_MIME_TYPES:
            logger.debug("Attempting to decode uncommon image type %s", content_type)

        preview_bytes, mime_type, transform = self._render_preview(raw, media_uuid)
        return MediaPreview(
            summary=(
                f"Preview of media {media_uuid}"
                + (f" frame {frame_index}" if frame_index is not None else "")
                + f": {transform.source_width}x{transform.source_height} source rendered at "
                f"{transform.preview_width}x{transform.preview_height} "
                f"({'scaled ' + format(transform.scale, '.3f') if transform.applied else 'unscaled'})."
            ),
            project_id=canonical,
            media_id=media_uuid,
            frame_index=frame_index,
            mime_type=mime_type,
            image_base64=base64.b64encode(preview_bytes).decode("ascii"),
            preview_bytes=len(preview_bytes),
            transform=transform,
            disclosure_notice=(
                "This image was disclosed because --allow-image-access is enabled. It leaves this machine "
                "through the assistant provider."
            ),
        )

    def _render_preview(self, raw: bytes, media_uuid: str) -> tuple[bytes, str, ResizeTransform]:
        """Decode, bound, and downscale source bytes into a preview."""
        previous_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = MAX_DECODED_PIXELS
        try:
            with Image.open(io.BytesIO(raw)) as decoded:
                source_width, source_height = decoded.size
                if source_width * source_height > MAX_DECODED_PIXELS:
                    raise GetiMcpError(
                        ErrorCode.LIMIT_EXCEEDED,
                        f"Media {media_uuid} decodes to {source_width}x{source_height} pixels, over the "
                        f"{MAX_DECODED_PIXELS} pixel preview limit.",
                    )
                frame = decoded.convert("RGB")
        except GetiMcpError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise GetiMcpError(
                ErrorCode.UNSUPPORTED_MEDIA,
                f"Media {media_uuid} could not be decoded as an image.",
                guidance="Confirm the item is an image, or pass frame_index for a video.",
                details={"decode_error": redact(str(exc), max_chars=200)},
            ) from exc
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit

        max_dimension = self._limits.preview_max_dimension
        scale = min(1.0, max_dimension / max(source_width, source_height))
        if scale < 1.0:
            frame = frame.resize(
                (max(1, round(source_width * scale)), max(1, round(source_height * scale))),
                Image.Resampling.LANCZOS,
            )
        preview_width, preview_height = frame.size

        encoded = _encode_within_budget(frame, self._limits.preview_max_bytes)
        if encoded is None:
            raise GetiMcpError(
                ErrorCode.LIMIT_EXCEEDED,
                f"Media {media_uuid} could not be encoded within the {self._limits.preview_max_bytes} byte "
                "preview budget.",
                guidance="Lower --preview-max-dimension or raise --preview-max-bytes.",
            )
        return (
            encoded,
            "image/jpeg",
            ResizeTransform(
                applied=scale < 1.0,
                source_width=source_width,
                source_height=source_height,
                preview_width=preview_width,
                preview_height=preview_height,
                scale=round(preview_width / source_width, 6) if source_width else 1.0,
            ),
        )

    # ------------------------------------------------------------------ architectures & configuration

    async def list_architectures(self, project_id: str) -> ArchitectureList:
        """List architectures compatible with a project's task.

        Args:
            project_id: Project whose task determines compatibility.

        Returns:
            ArchitectureList: Architecture IDs accepted by ``start_training``.
        """
        canonical = self._authz.require_project(project_id)
        project = await self._get_project_payload(canonical)
        task_type = str((project.get("task") or {}).get("task_type", ""))
        if not task_type:
            raise GetiMcpError(
                ErrorCode.BACKEND_ERROR,
                f"Project {canonical} does not declare a task type.",
            )

        payload = await self._client.request_json(
            "GET", f"{API_PREFIX}/model_architectures", params={"task": task_type}
        )
        data = payload if isinstance(payload, dict) else {}
        raw = [entry for entry in (data.get("model_architectures") or []) if isinstance(entry, dict)]
        recommendations = _top_pick_index(data.get("top_picks"))
        limited = raw[: self._limits.max_items]

        architectures = [
            ArchitectureInfo(
                id=str(entry.get("id")),
                name=str(entry.get("name", "")),
                task=str(entry.get("task", task_type)),
                support_status=entry.get("support_status"),
                license=entry.get("license"),
                gigaflops=_as_float((entry.get("stats") or {}).get("gigaflops")),
                trainable_parameters=_as_float((entry.get("stats") or {}).get("trainable_parameters")),
                benchmark_metrics=_benchmarks((entry.get("stats") or {}).get("benchmark_metrics")),
                recommended_for=sorted(recommendations.get(str(entry.get("id")), set())),
            )
            for entry in limited
        ]
        return ArchitectureList(
            summary=(
                f"{len(architectures)} architecture(s) compatible with the {task_type} task. "
                f"Pass one of these ids to start_training: {', '.join(a.id for a in architectures[:5])}"
                + ("…" if len(architectures) > 5 else "")
            ),
            project_id=canonical,
            task_type=task_type,
            architectures=architectures,
            total_available=len(raw),
            truncated=len(raw) > len(limited),
        )

    async def training_configuration(self, project_id: str, model_architecture_id: str) -> TrainingConfiguration:
        """Return the effective training configuration for a project and architecture.

        This is read-only; the adapter never mutates configuration.

        Args:
            project_id: Project to read configuration for.
            model_architecture_id: Architecture to read configuration for.

        Returns:
            TrainingConfiguration: Flattened effective values, bounded in count.
        """
        canonical = self._authz.require_project(project_id)
        architecture = _require_architecture_id(model_architecture_id)
        payload = await self._client.request_json(
            "GET",
            f"{API_PREFIX}/projects/{canonical}/training_configuration",
            params={"model_architecture_id": architecture},
        )
        entries = _configuration_entries(payload)
        limited = entries[: self._limits.max_items]
        return TrainingConfiguration(
            summary=(
                f"{len(limited)} effective configuration value(s) for {architecture} on project {canonical}"
                + (" (truncated)." if len(entries) > len(limited) else ".")
            ),
            project_id=canonical,
            model_architecture_id=architecture,
            entries=limited,
            truncated=len(entries) > len(limited),
            observed_at=datetime.now(tz=UTC).isoformat(),
            note=(
                "Values reflect a single read and may change before a job starts; Geti does not expose an "
                "atomic configuration snapshot."
            ),
        )

    # ------------------------------------------------------------------ jobs

    async def start_training(
        self,
        project_id: str,
        model_architecture_id: str,
        device: str,
        *,
        parent_model_revision_id: str | None = None,
        dataset_revision_id: str | None = None,
    ) -> JobSubmission:
        """Submit exactly one training job and return as soon as Geti accepts it.

        Inputs are checked against the project, the compatible architectures, and the
        available devices before submission, but Geti's own validation is authoritative.
        Nothing is silently substituted: no fallback architecture, no fallback device, no
        configuration edit, no licence acceptance, and no cancelling of other work.

        Args:
            project_id: Project to train in.
            model_architecture_id: Architecture identifier from ``list_model_architectures``.
            device: Device identifier from ``get_connection_info``.
            parent_model_revision_id: Model to fine-tune from, or ``None`` to train fresh.
            dataset_revision_id: Existing dataset revision, or ``None`` for the latest data.

        Returns:
            JobSubmission: The accepted job's identifier and initial state.

        Raises:
            GetiMcpError: ``PERMISSION_DENIED`` when training is not enabled,
                ``INCOMPATIBLE_BACKEND`` on an untested API version, ``INVALID_INPUT``
                when preflight or Geti rejects the request, and
                ``SUBMISSION_OUTCOME_UNKNOWN`` when the connection fails mid-submission.
        """
        self._authz.require_training()
        canonical = self._authz.require_project(project_id)
        architecture = _require_architecture_id(model_architecture_id)

        contract = await self._client.get_contract()
        if not contract.compatible:
            raise GetiMcpError(
                ErrorCode.INCOMPATIBLE_BACKEND,
                f"Geti reports API version {contract.api_version}, outside this adapter's tested range "
                f"({', '.join(SUPPORTED_API_VERSIONS)}). Training submission is refused.",
                guidance="Use the Geti UI or REST API directly, or upgrade this adapter.",
            )

        preflight: list[str] = [f"Geti API version {contract.api_version} ({contract.fingerprint})."]

        project = await self._get_project_payload(canonical)
        task_type = str((project.get("task") or {}).get("task_type", ""))
        label_count = len((project.get("task") or {}).get("labels") or [])
        preflight.append(f"Project '{project.get('name')}' is a {task_type} task with {label_count} label(s).")

        architectures = await self.list_architectures(canonical)
        known = {entry.id for entry in architectures.architectures}
        if architecture not in known:
            raise GetiMcpError(
                ErrorCode.INVALID_INPUT,
                f"'{architecture}' is not among the architectures compatible with the {task_type} task.",
                guidance="Call list_model_architectures for this project and pass one of the ids it returns.",
                details={"truncated_architecture_list": architectures.truncated},
            )
        preflight.append(f"Architecture '{architecture}' is listed as compatible with the {task_type} task.")

        devices = await self._training_devices()
        available = sorted(entry.device for entry in devices)
        if device not in available:
            raise GetiMcpError(
                ErrorCode.INVALID_INPUT,
                f"Device '{device}' is not among the training devices Geti reports.",
                guidance="Pass one of the available devices exactly as listed; do not substitute another.",
                details={"available_devices": available},
            )
        preflight.append(f"Device '{device}' is reported as available for training.")

        try:
            stats = await self.dataset_statistics(canonical)
            preflight.append(
                f"Dataset at preflight: {stats.images} image(s), {stats.annotated_images} annotated, "
                f"{stats.total_instances} instance(s)."
            )
        except GetiMcpError as exc:
            preflight.append(f"Dataset statistics unavailable at preflight: {exc.message}")

        parameters: dict[str, Any] = {"device": device, "model_architecture_id": architecture}
        if parent_model_revision_id is not None:
            parameters["parent_model_revision_id"] = self._authz.parse_uuid(
                parent_model_revision_id, field="parent_model_revision_id"
            )
        if dataset_revision_id is not None:
            parameters["dataset_revision_id"] = self._authz.parse_uuid(dataset_revision_id, field="dataset_revision_id")

        preflight.append("These observations were read separately before submission and are not an atomic snapshot.")

        payload = await self._client.request_json(
            "POST",
            f"{API_PREFIX}/jobs",
            json_body={"job_type": "train", "project_id": canonical, "parameters": parameters},
        )
        job = payload if isinstance(payload, dict) else {}
        backend_status = str(job.get("status", ""))
        state = normalize_status(backend_status)
        job_id = str(job.get("job_id", ""))
        return JobSubmission(
            summary=(
                f"Training job {job_id} accepted for project {canonical} using {architecture} on {device} "
                f"(status {backend_status}). Poll it with get_job or wait_for_job."
            ),
            job_id=job_id,
            project_id=canonical,
            model_architecture_id=architecture,
            device=device,
            backend_status=backend_status,
            state=state.value,
            preflight=preflight,
        )

    async def list_jobs(self, *, project_id: str | None = None, job_type: str | None = None) -> JobList:
        """List jobs whose owning project is inside the authorization scope.

        Geti's job endpoint is global and unpaginated, so filtering and bounding happen
        here. Jobs whose owning project cannot be resolved are excluded, not guessed at.

        Args:
            project_id: Restrict to one authorized project.
            job_type: Restrict to one Geti job type, e.g. ``train``.

        Returns:
            JobList: Authorized jobs, most recently active first.
        """
        wanted_project = self._authz.require_project(project_id) if project_id is not None else None
        wanted_type = _require_choice(job_type, JOB_TYPES, field="job_type") if job_type is not None else None

        payload = await self._client.request_json("GET", f"{API_PREFIX}/jobs")
        raw = payload if isinstance(payload, list) else []

        authorized: list[tuple[dict[str, Any], str]] = []
        excluded = 0
        for entry in raw:
            if not isinstance(entry, dict):
                excluded += 1
                continue
            owner = owning_project_id(entry)
            if owner is None or not self._authz.is_project_allowed(owner):
                excluded += 1
                continue
            if wanted_project is not None and owner != wanted_project:
                continue
            if job_type is not None and str(entry.get("job_type")) != wanted_type:
                continue
            authorized.append((entry, owner))

        authorized.sort(key=lambda pair: _sort_key(pair[0]), reverse=True)
        limited = authorized[: self._limits.max_items]
        jobs = [_job_summary(entry, owner) for entry, owner in limited]
        return JobList(
            summary=(
                f"{len(jobs)} authorized job(s)"
                + (f" for project {wanted_project}" if wanted_project else "")
                + (f"; {excluded} out-of-scope or unattributable job(s) excluded." if excluded else ".")
            ),
            jobs=jobs,
            returned=len(jobs),
            excluded_unauthorized=excluded,
            truncated=len(authorized) > len(limited),
        )

    async def get_job(self, job_id: str) -> JobDetail:
        """Return the full state of one authorized job.

        Args:
            job_id: Job to inspect.

        Returns:
            JobDetail: Backend status, normalized state, progress, and result references.

        Raises:
            GetiMcpError: ``NOT_FOUND`` when Geti has no such job, which after a restart
                means the record may have been lost rather than that training failed.
        """
        job = await self._fetch_authorized_job(job_id)
        return _job_detail(job, owning_project_id(job) or "")

    async def wait_for_job(self, job_id: str, max_wait_seconds: float | None = None) -> JobWait:
        """Wait a bounded time for a job to reach a terminal state.

        A timeout means the job is still running. Abandoning this call, or the client
        disconnecting, never cancels the underlying job.

        Args:
            job_id: Job to wait on.
            max_wait_seconds: Requested bound, clamped to the configured maximum.

        Returns:
            JobWait: The job's state when the wait ended, and why it ended.
        """
        deadline_budget = self._authz.clamp_wait(max_wait_seconds)
        started = time.monotonic()
        interval = _POLL_INITIAL_INTERVAL
        polls = 0

        try:
            while True:
                job = await self._fetch_authorized_job(job_id)
                polls += 1
                detail = _job_detail(job, owning_project_id(job) or "")
                elapsed = time.monotonic() - started
                if detail.terminal:
                    return JobWait(
                        summary=f"Job {detail.job_id} reached terminal state '{detail.state}' after {elapsed:.1f}s.",
                        job=detail,
                        reached_terminal=True,
                        waited_seconds=round(elapsed, 3),
                        timed_out=False,
                        polls=polls,
                    )
                remaining = deadline_budget - elapsed
                if remaining <= 0:
                    return JobWait(
                        summary=(
                            f"Job {detail.job_id} is still '{detail.state}' after waiting {elapsed:.1f}s. "
                            "This is not a failure; the job continues running. Call wait_for_job again "
                            "to keep watching."
                        ),
                        job=detail,
                        reached_terminal=False,
                        waited_seconds=round(elapsed, 3),
                        timed_out=True,
                        polls=polls,
                    )
                await asyncio.sleep(min(interval, remaining))
                interval = min(interval * _POLL_BACKOFF, _POLL_MAX_INTERVAL)
        except asyncio.CancelledError:
            # The caller went away. Deliberately do nothing to the job: a wait is an
            # observation, and abandoning it must never stop training.
            logger.info("wait_for_job was cancelled; job %s is left running untouched", job_id)
            raise

    async def cancel_job(self, job_id: str) -> JobCancellation:
        """Request cancellation of one authorized job.

        Args:
            job_id: Job to cancel.

        Returns:
            JobCancellation: Distinguishes a cancellation that was merely requested from
            one that has actually terminated.

        Raises:
            GetiMcpError: ``PERMISSION_DENIED`` when cancellation is not enabled.
        """
        self._authz.require_job_cancellation()
        # Resolve ownership from the job record before mutating anything.
        existing = await self._fetch_authorized_job(job_id)
        owner = owning_project_id(existing) or ""
        canonical_job = self._authz.parse_uuid(job_id, field="job_id")

        existing_state = normalize_status(str(existing.get("status", "")))
        if is_terminal(existing_state):
            # Already finished: sending a cancellation would be a pointless mutation.
            return JobCancellation(
                summary=(f"Job {canonical_job} already reached '{existing_state.value}', so no cancellation was sent."),
                job_id=canonical_job,
                project_id=owner,
                cancellation_requested=False,
                terminated=True,
                backend_status=str(existing.get("status", "")),
                state=existing_state.value,
            )

        payload = await self._client.request_json("POST", f"{API_PREFIX}/jobs/{canonical_job}:cancel")
        job = payload if isinstance(payload, dict) else existing
        backend_status = str(job.get("status", ""))
        state = normalize_status(backend_status)
        terminated = is_terminal(state)
        return JobCancellation(
            summary=(
                f"Cancellation requested for job {canonical_job}; Geti reports '{backend_status}'. "
                + (
                    "The job has terminated."
                    if terminated
                    else "The job has not terminated yet; poll get_job to confirm it stops."
                )
            ),
            job_id=canonical_job,
            project_id=owner,
            cancellation_requested=True,
            terminated=terminated,
            backend_status=backend_status,
            state=state.value,
        )

    # ------------------------------------------------------------------ models

    async def list_models(self, project_id: str, dataset_revision_id: str | None = None) -> ModelList:
        """List models belonging to one authorized project.

        Args:
            project_id: Project whose models to list.
            dataset_revision_id: Optional filter for models trained on one revision.

        Returns:
            ModelList: Model identifiers, variants, and available summary metrics.
        """
        canonical = self._authz.require_project(project_id)
        params: dict[str, Any] = {}
        if dataset_revision_id is not None:
            params["dataset_revision_id"] = self._authz.parse_uuid(dataset_revision_id, field="dataset_revision_id")

        payload = await self._client.request_json("GET", f"{API_PREFIX}/projects/{canonical}/models", params=params)
        raw = [entry for entry in (payload if isinstance(payload, list) else []) if isinstance(entry, dict)]
        limited = raw[: self._limits.max_items]
        models = [_model_summary(entry) for entry in limited]
        return ModelList(
            summary=(
                f"{len(models)} model(s) in project {canonical}: "
                + (", ".join(f"{m.name} [{m.architecture}]" for m in models[:5]) or "none")
                + ("…" if len(models) > 5 else "")
            ),
            project_id=canonical,
            models=models,
            total_available=len(raw),
            truncated=len(raw) > len(limited),
            comparison_note=_COMPARISON_NOTE,
        )

    async def model_results(self, project_id: str, model_id: str) -> ModelResults:
        """Return one model's evaluation results together with their context.

        Args:
            project_id: Project the model belongs to.
            model_id: Model to describe.

        Returns:
            ModelResults: Metrics grouped by variant, the distinct dataset revisions and
            subsets they came from, and explicit statements of what is missing.
        """
        canonical = self._authz.require_project(project_id)
        model_uuid = self._authz.parse_uuid(model_id, field="model_id")
        payload = await self._client.request_json("GET", f"{API_PREFIX}/projects/{canonical}/models/{model_uuid}")
        model = payload if isinstance(payload, dict) else {}
        training = model.get("training_info") or {}

        evaluations, revisions, subsets, truncated = _parse_evaluations(
            model.get("variants"), budget=self._limits.max_items
        )

        missing: list[str] = []
        if not evaluations:
            missing.append("Geti reported no evaluations for any variant of this model.")
        if not training.get("dataset_revision_id"):
            missing.append("Geti did not report the dataset revision this model was trained on.")
        if model.get("files_deleted"):
            missing.append("Model files have been deleted; only recorded metadata remains.")
        if len(revisions) > 1:
            missing.append(
                f"Evaluations span {len(revisions)} different dataset revisions; metrics across them are "
                "not directly comparable."
            )
        if len(subsets) > 1:
            missing.append(
                f"Evaluations span {len(subsets)} different subsets ({', '.join(sorted(subsets))}); compare "
                "only within the same subset."
            )

        metric_total = sum(len(e.metrics) for group in evaluations.values() for e in group)
        return ModelResults(
            summary=(
                f"Model '{model.get('name')}' ({model.get('architecture')}): {metric_total} metric value(s) "
                f"across {len(evaluations)} variant(s), {len(revisions)} dataset revision(s), "
                f"{len(subsets)} subset(s)." + (f" Missing: {len(missing)} item(s)." if missing else "")
            ),
            project_id=canonical,
            model_id=model_uuid,
            name=str(model.get("name", "")),
            architecture=str(model.get("architecture", "")),
            training_status=training.get("status"),
            training_started_at=_iso(training.get("start_time")),
            training_ended_at=_iso(training.get("end_time")),
            trained_on_dataset_revision_id=_opt_str(training.get("dataset_revision_id")),
            evaluations_by_variant=evaluations,
            distinct_dataset_revisions=sorted(revisions),
            distinct_subsets=sorted(subsets),
            missing_data=missing,
            truncated=truncated,
        )

    # ------------------------------------------------------------------ internals

    async def _get_project_payload(self, canonical_project_id: str) -> dict[str, Any]:
        payload = await self._client.request_json("GET", f"{API_PREFIX}/projects/{canonical_project_id}")
        if not isinstance(payload, dict):
            raise GetiMcpError(ErrorCode.BACKEND_ERROR, f"Unexpected project payload for {canonical_project_id}.")
        return payload

    async def _fetch_authorized_job(self, job_id: str) -> dict[str, Any]:
        canonical = self._authz.parse_uuid(job_id, field="job_id")
        payload = await self._client.request_json("GET", f"{API_PREFIX}/jobs/{canonical}")
        if not isinstance(payload, dict):
            raise GetiMcpError(ErrorCode.BACKEND_ERROR, f"Unexpected job payload for {canonical}.")
        # Ownership is resolved from the job record and authorized before any data is
        # returned or any state is changed.
        self._authz.require_authorized_job(payload)
        return payload

    async def _training_devices(self) -> list[DeviceInfo]:
        payload = await self._client.request_json("GET", f"{API_PREFIX}/system/devices/training")
        devices: list[DeviceInfo] = []
        for entry in payload if isinstance(payload, list) else []:
            if not isinstance(entry, dict):
                continue
            device_type = str(entry.get("type", ""))
            index = entry.get("index")
            identifier = device_type if device_type in {"cpu", "auto"} else f"{device_type}-{index or 0}"
            devices.append(
                DeviceInfo(
                    device=identifier,
                    type=device_type,
                    name=str(entry.get("name", "")),
                    memory_bytes=entry.get("memory"),
                )
            )
        return devices


# ---------------------------------------------------------------------- parsing helpers


def _media_item(entry: dict[str, Any]) -> MediaItem:
    return MediaItem(
        id=str(entry.get("id")),
        name=str(entry.get("name", "")),
        type=str(entry.get("type", "")),
        width=entry.get("width"),
        height=entry.get("height"),
        size_bytes=entry.get("size"),
        format=_opt_str(entry.get("format")),
        fps=_as_float(entry.get("fps")),
        frame_count=entry.get("frame_count"),
        annotated_frame_count=entry.get("annotated_frame_count"),
    )


def _job_summary(entry: dict[str, Any], owner: str) -> JobSummary:
    backend_status = str(entry.get("status", ""))
    state = normalize_status(backend_status)
    return JobSummary(
        job_id=str(entry.get("job_id")),
        job_type=str(entry.get("job_type", "")),
        project_id=owner,
        backend_status=backend_status,
        state=state.value,
        terminal=is_terminal(state),
        progress=_as_float(entry.get("progress")),
        started_at=_iso(entry.get("started_at")),
        finished_at=_iso(entry.get("finished_at")),
    )


def _job_detail(entry: dict[str, Any], owner: str) -> JobDetail:
    backend_status = str(entry.get("status", ""))
    state = normalize_status(backend_status)
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    model = metadata.get("model") if isinstance(metadata.get("model"), dict) else {}
    device = metadata.get("device") if isinstance(metadata.get("device"), dict) else {}
    device_type = str(device.get("type", "")) if device else ""
    device_id = (
        (device_type if device_type in {"cpu", "auto"} else f"{device_type}-{device.get('index') or 0}")
        if device_type
        else None
    )

    state_note = {
        NormalizedJobState.CANCELLING: " Cancellation was requested but the job has not stopped yet.",
        NormalizedJobState.UNKNOWN: " This adapter does not recognize that status; treat it as indeterminate.",
    }.get(state, "")

    return JobDetail(
        summary=(
            f"Job {entry.get('job_id')} ({entry.get('job_type')}) is '{backend_status}' "
            f"-> normalized '{state.value}', progress {_as_float(entry.get('progress')) or 0:.0f}%." + state_note
        ),
        job_id=str(entry.get("job_id")),
        job_type=str(entry.get("job_type", "")),
        project_id=owner,
        backend_status=backend_status,
        state=state.value,
        terminal=is_terminal(state),
        progress=_as_float(entry.get("progress")),
        message=redact(str(entry["message"])) if entry.get("message") else None,
        error=redact(str(entry["error"])) if entry.get("error") else None,
        started_at=_iso(entry.get("started_at")),
        finished_at=_iso(entry.get("finished_at")),
        model_id=_opt_str(model.get("id")),
        model_architecture_id=_opt_str(model.get("architecture")),
        dataset_revision_id=_opt_str(model.get("dataset_revision_id")),
        device=device_id,
    )


def _model_summary(entry: dict[str, Any]) -> ModelSummary:
    training = entry.get("training_info") or {}
    variants: list[ModelVariantSummary] = []
    primary_metrics: list[MetricValue] = []
    contexts: set[tuple[str, str]] = set()

    for variant in entry.get("variants") or []:
        if not isinstance(variant, dict):
            continue
        evaluations = [e for e in (variant.get("evaluations") or []) if isinstance(e, dict)]
        variants.append(
            ModelVariantSummary(
                id=str(variant.get("id")),
                format=str(variant.get("format", "")),
                precision=str(variant.get("precision", "")),
                weights_size_bytes=variant.get("weights_size"),
                files_deleted=bool(variant.get("files_deleted", False)),
                quantized=variant.get("quantization_info") is not None,
                evaluation_count=len(evaluations),
            )
        )
        for evaluation in evaluations:
            contexts.add((str(evaluation.get("dataset_revision_id", "")), str(evaluation.get("subset", ""))))
            for metric in evaluation.get("metrics") or []:
                if isinstance(metric, dict) and metric.get("primary") and metric.get("value") is not None:
                    primary_metrics.append(
                        MetricValue(
                            name=str(metric.get("name", "")),
                            value=float(metric["value"]),
                            primary=True,
                        )
                    )

    # Only surface a headline metric when it is unambiguous: one evaluation context and one
    # primary metric. Anything else must be read from get_model_results with its context.
    headline = primary_metrics[0] if len(primary_metrics) == 1 and len(contexts) == 1 else None

    return ModelSummary(
        id=str(entry.get("id")),
        name=str(entry.get("name", "")),
        architecture=str(entry.get("architecture", "")),
        parent_revision_id=_opt_str(entry.get("parent_revision")),
        training_status=_opt_str(training.get("status")),
        dataset_revision_id=_opt_str(training.get("dataset_revision_id")),
        size_bytes=entry.get("size"),
        files_deleted=bool(entry.get("files_deleted", False)),
        variants=variants,
        headline_metric=headline,
    )


def _top_pick_index(top_picks: Any) -> dict[str, set[str]]:
    """Map architecture id -> the top-pick categories it appears in."""
    index: dict[str, set[str]] = {}
    if not isinstance(top_picks, dict):
        return index
    for category, value in top_picks.items():
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            identifier = candidate.get("id") if isinstance(candidate, dict) else candidate
            if isinstance(identifier, str):
                index.setdefault(identifier, set()).add(str(category))
    return index


def _benchmarks(raw: Any) -> BenchmarkMetrics | None:
    if not isinstance(raw, dict):
        return None
    return BenchmarkMetrics(
        imagenet_top1_accuracy=_as_float(raw.get("imagenet_top1_accuracy")),
        imagenet_top5_accuracy=_as_float(raw.get("imagenet_top5_accuracy")),
        coco_map_50=_as_float(raw.get("coco_map_50")),
        coco_map_50_95=_as_float(raw.get("coco_map_50_95")),
    )


def _parse_evaluations(
    variants: Any, *, budget: int
) -> tuple[dict[str, list[EvaluationResult]], set[str], set[str], bool]:
    """Group a model's evaluations by variant, bounded by ``budget`` evaluations in total.

    Args:
        variants: Raw ``variants`` list from a Geti model payload.
        budget: Maximum number of evaluations to parse across all variants.

    Returns:
        tuple: Evaluations per variant id, the distinct dataset revisions seen, the distinct
        subsets seen, and whether the budget cut the result short.
    """
    evaluations: dict[str, list[EvaluationResult]] = {}
    revisions: set[str] = set()
    subsets: set[str] = set()
    truncated = False

    for variant in variants or []:
        if not isinstance(variant, dict):
            continue
        parsed: list[EvaluationResult] = []
        for evaluation in variant.get("evaluations") or []:
            if not isinstance(evaluation, dict):
                continue
            if budget <= 0:
                truncated = True
                break
            budget -= 1
            revision = str(evaluation.get("dataset_revision_id", ""))
            subset = str(evaluation.get("subset", ""))
            revisions.add(revision)
            subsets.add(subset)
            parsed.append(
                EvaluationResult(
                    dataset_revision_id=revision,
                    subset=subset,
                    metrics=[
                        MetricValue(
                            name=str(metric.get("name", "")),
                            value=float(metric.get("value", 0.0)),
                            primary=bool(metric.get("primary", False)),
                        )
                        for metric in (evaluation.get("metrics") or [])
                        if isinstance(metric, dict) and metric.get("value") is not None
                    ],
                )
            )
        if parsed:
            evaluations[str(variant.get("id"))] = parsed

    return evaluations, revisions, subsets, truncated


def _configuration_entries(payload: Any) -> list[ConfigurationEntry]:
    """Walk Geti's recursive parameter tree into a flat, ordered list of entries.

    Geti returns ``{"parameters": [...]}`` where each item is either a leaf parameter or a
    ``parameter_group`` holding more items. Group keys are joined with dots so the resulting
    key identifies the parameter unambiguously.
    """
    entries: list[ConfigurationEntry] = []
    _collect_parameters(payload.get("parameters") if isinstance(payload, dict) else None, "", entries)
    return entries


def _collect_parameters(nodes: Any, prefix: str, entries: list[ConfigurationEntry]) -> None:
    if not isinstance(nodes, list):
        return
    for node in nodes:
        if not isinstance(node, dict):
            continue
        key = str(node.get("key", ""))
        path = f"{prefix}.{key}" if prefix else key
        if node.get("type") == "parameter_group" or isinstance(node.get("parameters"), list):
            _collect_parameters(node.get("parameters"), path, entries)
            continue
        entries.append(
            ConfigurationEntry(
                key=path,
                name=str(node.get("name", key)),
                value=_render_value(node.get("value")),
                default_value=(None if node.get("default_value") is None else _render_value(node.get("default_value"))),
                value_type=_opt_str(node.get("value_type")),
                min_value=_as_float(node.get("min_value")),
                max_value=_as_float(node.get("max_value")),
                allowed_values=(
                    [_render_value(item) for item in node["allowed_values"]]
                    if isinstance(node.get("allowed_values"), list)
                    else None
                ),
            )
        )


def _render_value(value: Any) -> str:
    """Render a configuration value as bounded text."""
    if isinstance(value, list):
        return redact(", ".join(str(item) for item in value), max_chars=200)
    return redact(str(value), max_chars=200)


def _encode_within_budget(frame: Image.Image, budget: int) -> bytes | None:
    """Encode to JPEG, stepping quality down until the payload fits the budget."""
    for quality in (85, 70, 55, 40, 25):
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded = buffer.getvalue()
        if len(encoded) <= budget:
            return encoded
    return None


def _require_choice(value: str, allowed: Collection[str], *, field: str) -> str:
    if value not in allowed:
        raise GetiMcpError(
            ErrorCode.INVALID_INPUT,
            f"'{field}' must be one of {', '.join(sorted(allowed))}, got '{value}'.",
        )
    return value


def _require_architecture_id(value: str) -> str:
    candidate = str(value).strip()
    # Architecture ids are opaque slugs; reject anything that could alter the request path.
    if not candidate or len(candidate) > 200 or not all(c.isalnum() or c in "-_." for c in candidate):
        raise GetiMcpError(
            ErrorCode.INVALID_INPUT,
            "'model_architecture_id' must be an architecture slug such as 'object-detection-atss-mobilenet-v2'.",
            guidance="Call list_model_architectures and pass one of the ids it returns.",
        )
    return candidate


def _sort_key(job: dict[str, Any]) -> str:
    return str(job.get("finished_at") or job.get("started_at") or "")


def _iso(value: Any) -> str | None:
    return str(value) if value is not None else None


def _opt_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
