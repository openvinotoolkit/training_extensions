# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Public tool input and output schemas.

Every tool declares an explicit Pydantic return model so the MCP ``outputSchema`` is
generated rather than hand-written. Each model carries a ``summary`` field holding the
concise text rendering of the same data, which is what a host shows when it does not
render structured content.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TaskTypeName = Literal["classification", "detection", "instance_segmentation"]
MediaTypeName = Literal["image", "video", "video_frame"]
SubsetName = Literal["unassigned", "training", "validation", "testing"]
AnnotationStatusName = Literal["with_annotations", "missing_annotations"]
NormalizedStateName = Literal["queued", "running", "cancelling", "succeeded", "failed", "cancelled", "unknown"]


class _Result(BaseModel):
    """Base for every tool result."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="Concise human-readable rendering of this result.")


class PermissionInfo(BaseModel):
    """Which mutating or disclosing capabilities the operator enabled."""

    model_config = ConfigDict(extra="forbid")

    allow_training: bool = Field(description="Whether start_training may submit jobs.")
    allow_job_cancellation: bool = Field(description="Whether cancel_job may request cancellation.")
    allow_image_access: bool = Field(description="Whether view_media may return image content.")


class ProjectScopeInfo(BaseModel):
    """The operator-configured project authorization policy."""

    model_config = ConfigDict(extra="forbid")

    policy: Literal["allowlist", "all_projects"] = Field(description="How project access was configured.")
    authorized_project_ids: list[str] | None = Field(
        default=None, description="Allowlisted project IDs, or null when every project is authorized."
    )


class DeviceInfo(BaseModel):
    """A training device Geti reports as available."""

    model_config = ConfigDict(extra="forbid")

    device: str = Field(description="Identifier to pass to start_training, e.g. 'cpu', 'xpu-0', 'cuda-1'.")
    type: str = Field(description="Device type reported by Geti: cpu, xpu, or cuda.")
    name: str = Field(description="Human-readable device name.")
    memory_bytes: int | None = Field(default=None, description="Total device memory, null for CPU.")


class ConnectionInfo(_Result):
    """Diagnostics for the configured Geti instance and this server's own policy."""

    reachable: bool = Field(description="Whether Geti responded to the contract probe.")
    base_url: str = Field(description="Configured Geti base URL. Validated at startup to carry no credentials.")
    api_version: str | None = Field(default=None, description="Geti API contract version, when reachable.")
    contract_fingerprint: str | None = Field(
        default=None, description="Digest over the served operation surface; identical values mean identical APIs."
    )
    compatible: bool = Field(description="Whether the API version is in this adapter's tested range.")
    tested_api_versions: list[str] = Field(description="API versions this adapter was tested against.")
    permissions: PermissionInfo = Field(description="Capabilities the operator enabled.")
    project_scope: ProjectScopeInfo = Field(description="Project authorization policy.")
    training_devices: list[DeviceInfo] = Field(
        default_factory=list, description="Devices available for training, when reachable."
    )
    license_accepted: bool | None = Field(
        default=None, description="Whether the Geti license has been accepted for the running version."
    )
    platform: str | None = Field(default=None, description="Operating system Geti runs on.")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal problems observed while probing.")


class LabelInfo(BaseModel):
    """A label defined on a project's task."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Label identifier.")
    name: str = Field(description="Label name. Untrusted user-supplied text.")
    color: str | None = Field(default=None, description="Hex display colour.")


class ProjectSummary(BaseModel):
    """Identity and task type of one authorized project."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Project identifier.")
    name: str = Field(description="Project name. Untrusted user-supplied text.")
    task_type: TaskTypeName | str = Field(description="Task type the project was created with.")
    created_at: str | None = Field(default=None, description="ISO-8601 creation timestamp.")


class ProjectList(_Result):
    """Authorized projects."""

    projects: list[ProjectSummary] = Field(description="Projects inside the configured authorization scope.")
    total_authorized: int = Field(description="Number of authorized projects found on the instance.")
    truncated: bool = Field(description="Whether the list was cut short by the configured item limit.")


class ProjectDetail(_Result):
    """One project's task configuration and label set."""

    id: str = Field(description="Project identifier.")
    name: str = Field(description="Project name. Untrusted user-supplied text.")
    task_type: TaskTypeName | str = Field(description="Task type the project was created with.")
    exclusive_labels: bool = Field(description="True for multi-class classification, false for multi-label.")
    labels: list[LabelInfo] = Field(description="Labels defined on the task.")
    active_pipeline: bool = Field(description="Whether an inference pipeline is currently active.")
    created_at: str | None = Field(default=None, description="ISO-8601 creation timestamp.")


class LabelInstanceCount(BaseModel):
    """Annotated instance count for one label."""

    model_config = ConfigDict(extra="forbid")

    label_id: str | None = Field(default=None, description="Label identifier, null for unlabelled instances.")
    label_name: str | None = Field(default=None, description="Label name resolved from the project, when known.")
    instances: int = Field(description="Number of annotated instances carrying this label.")


class DatasetStatistics(_Result):
    """Media and annotation counts Geti reports for a project's dataset."""

    project_id: str = Field(description="Project the statistics belong to.")
    images: int = Field(description="Number of images.")
    videos: int = Field(description="Number of videos.")
    video_frames: int = Field(description="Number of video frames.")
    annotated_images: int = Field(description="Number of annotated images.")
    annotated_videos: int = Field(description="Number of videos with at least one annotated frame.")
    annotated_video_frames: int = Field(description="Number of annotated video frames.")
    total_instances: int = Field(description="Total annotated instances across the dataset.")
    instances_per_label: list[LabelInstanceCount] = Field(description="Annotated instances broken down by label.")
    readiness_warnings: list[str] = Field(
        description=(
            "Observations derived only from the counts above, such as labels with no instances. "
            "These are not a trainability verdict: Geti validates that when a training job is submitted."
        )
    )


class MediaItem(BaseModel):
    """Metadata for one media item. No pixel data."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Media identifier.")
    name: str = Field(description="File name. Untrusted user-supplied text.")
    type: MediaTypeName | str = Field(description="Media type.")
    width: int | None = Field(default=None, description="Pixel width.")
    height: int | None = Field(default=None, description="Pixel height.")
    size_bytes: int | None = Field(default=None, description="Stored size in bytes.")
    format: str | None = Field(default=None, description="Container or image format.")
    fps: float | None = Field(default=None, description="Frames per second, videos only.")
    frame_count: int | None = Field(default=None, description="Total frames, videos only.")
    annotated_frame_count: int | None = Field(default=None, description="Annotated frames, videos only.")


class MediaList(_Result):
    """A bounded page of dataset media metadata."""

    project_id: str = Field(description="Project the media belong to.")
    items: list[MediaItem] = Field(description="Media metadata for this page.")
    offset: int = Field(description="Offset this page starts at.")
    limit: int = Field(description="Page size actually used, after clamping to the configured maximum.")
    count: int = Field(description="Number of items in this page.")
    total: int = Field(description="Total items matching the filters across all pages.")
    next_offset: int | None = Field(default=None, description="Offset for the next page, or null at the end.")


class ResizeTransform(BaseModel):
    """How a preview was scaled down relative to the source."""

    model_config = ConfigDict(extra="forbid")

    applied: bool = Field(description="Whether the image was resized.")
    source_width: int = Field(description="Width of the decoded source image.")
    source_height: int = Field(description="Height of the decoded source image.")
    preview_width: int = Field(description="Width of the returned preview.")
    preview_height: int = Field(description="Height of the returned preview.")
    scale: float = Field(description="preview_width / source_width. 1.0 when no resize was applied.")


class MediaPreview(_Result):
    """A gated single-image preview, returned as base64 alongside its transform."""

    project_id: str = Field(description="Project the media belongs to.")
    media_id: str = Field(description="Media identifier.")
    frame_index: int | None = Field(default=None, description="Video frame index, null for images.")
    mime_type: str = Field(description="Media type of the returned preview bytes.")
    image_base64: str = Field(description="Base64-encoded preview image bytes.")
    preview_bytes: int = Field(description="Size of the decoded preview in bytes.")
    transform: ResizeTransform = Field(description="Resize applied to the source before encoding.")
    disclosure_notice: str = Field(description="Reminder that preview content leaves this machine.")


class BenchmarkMetrics(BaseModel):
    """Published benchmark scores for an architecture.

    These come from public benchmarks, not from this Geti project's data. They are a rough
    guide to relative quality and say nothing about how the architecture will score here.
    """

    model_config = ConfigDict(extra="forbid")

    imagenet_top1_accuracy: float | None = Field(default=None, description="ImageNet top-1 accuracy, percent.")
    imagenet_top5_accuracy: float | None = Field(default=None, description="ImageNet top-5 accuracy, percent.")
    coco_map_50: float | None = Field(default=None, description="COCO mAP@0.5, percent.")
    coco_map_50_95: float | None = Field(default=None, description="COCO mAP@0.5:0.95, percent.")


class ArchitectureInfo(BaseModel):
    """A model architecture compatible with a project's task."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Architecture identifier. This is the value start_training requires.")
    name: str = Field(description="Display name.")
    task: str = Field(description="Task type the architecture supports.")
    support_status: str | None = Field(default=None, description="Lifecycle status Geti reports, e.g. 'active'.")
    license: str | None = Field(default=None, description="License the architecture is distributed under.")
    gigaflops: float | None = Field(default=None, description="Reported GFLOPs.")
    trainable_parameters: float | None = Field(
        default=None, description="Reported trainable parameter count, in millions."
    )
    benchmark_metrics: BenchmarkMetrics | None = Field(
        default=None, description="Published benchmark scores, when Geti reports them. Not this project's scores."
    )
    recommended_for: list[str] = Field(
        default_factory=list, description="Top-pick categories Geti places this architecture in."
    )


class ArchitectureList(_Result):
    """Architectures compatible with one project's task."""

    project_id: str = Field(description="Project whose task was used for the lookup.")
    task_type: str = Field(description="Task type the architectures were filtered by.")
    architectures: list[ArchitectureInfo] = Field(description="Compatible architectures.")
    total_available: int = Field(description="Number Geti reported before this server's item limit was applied.")
    truncated: bool = Field(description="Whether the list was cut short by the configured item limit.")


class ConfigurationEntry(BaseModel):
    """One effective training configuration value."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(description="Dotted path of the parameter within the configuration tree.")
    name: str = Field(description="Human-readable parameter name as Geti presents it.")
    value: str = Field(description="Effective value, rendered as text.")
    default_value: str | None = Field(default=None, description="Geti's default, rendered as text.")
    value_type: str | None = Field(default=None, description="Declared type, e.g. 'int', 'bool', 'float_range'.")
    min_value: float | None = Field(default=None, description="Lower bound, when the parameter is bounded.")
    max_value: float | None = Field(default=None, description="Upper bound, when the parameter is bounded.")
    allowed_values: list[str] | None = Field(
        default=None, description="Permitted values, when the parameter is an enumeration."
    )


class TrainingConfiguration(_Result):
    """The effective training configuration for a project and architecture."""

    project_id: str = Field(description="Project the configuration applies to.")
    model_architecture_id: str = Field(description="Architecture the configuration applies to.")
    entries: list[ConfigurationEntry] = Field(description="Flattened effective configuration values.")
    truncated: bool = Field(description="Whether entries were cut short by the configured item limit.")
    observed_at: str = Field(description="ISO-8601 time this configuration was read.")
    note: str = Field(
        description=(
            "Values reflect a single read and may change before a job starts; Geti does not expose an "
            "atomic configuration snapshot."
        )
    )


class JobSummary(BaseModel):
    """Compact state of one authorized job."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(description="Job identifier.")
    job_type: str = Field(description="Geti job type, e.g. 'train'.")
    project_id: str = Field(description="Owning project, resolved and authorized before returning.")
    backend_status: str = Field(description="Status string exactly as Geti reported it.")
    state: NormalizedStateName = Field(description="Normalized lifecycle state.")
    terminal: bool = Field(description="Whether the normalized state is terminal.")
    progress: float | None = Field(default=None, description="Percentage complete, 0-100, when Geti reports it.")
    started_at: str | None = Field(default=None, description="ISO-8601 start time.")
    finished_at: str | None = Field(default=None, description="ISO-8601 finish time.")


class JobList(_Result):
    """Jobs inside the configured authorization scope."""

    jobs: list[JobSummary] = Field(description="Authorized jobs, newest activity first.")
    returned: int = Field(description="Number of jobs returned.")
    excluded_unauthorized: int = Field(
        description="Jobs Geti listed that were skipped because their owning project is out of scope or unknown."
    )
    truncated: bool = Field(description="Whether the list was cut short by the configured item limit.")


class JobDetail(_Result):
    """Full state of one authorized job."""

    job_id: str = Field(description="Job identifier.")
    job_type: str = Field(description="Geti job type.")
    project_id: str = Field(description="Owning project, resolved and authorized before returning.")
    backend_status: str = Field(description="Status string exactly as Geti reported it.")
    state: NormalizedStateName = Field(description="Normalized lifecycle state.")
    terminal: bool = Field(description="Whether the normalized state is terminal.")
    progress: float | None = Field(default=None, description="Percentage complete, 0-100, when Geti reports it.")
    message: str | None = Field(default=None, description="Status message from Geti. Untrusted text, redacted.")
    error: str | None = Field(default=None, description="Error from Geti when the job failed. Redacted.")
    started_at: str | None = Field(default=None, description="ISO-8601 start time.")
    finished_at: str | None = Field(default=None, description="ISO-8601 finish time.")
    model_id: str | None = Field(default=None, description="Model this job produced or operated on, when applicable.")
    model_architecture_id: str | None = Field(default=None, description="Architecture used, for training jobs.")
    dataset_revision_id: str | None = Field(default=None, description="Dataset revision used, when Geti reports one.")
    device: str | None = Field(default=None, description="Device the job runs on, for training jobs.")


class JobSubmission(_Result):
    """Acknowledgement that a training job was accepted by Geti."""

    job_id: str = Field(description="Identifier of the submitted job. Use get_job or wait_for_job to follow it.")
    project_id: str = Field(description="Project the job was submitted for.")
    model_architecture_id: str = Field(description="Architecture submitted.")
    device: str = Field(description="Device submitted.")
    backend_status: str = Field(description="Status Geti reported on acceptance.")
    state: NormalizedStateName = Field(description="Normalized lifecycle state on acceptance.")
    preflight: list[str] = Field(
        description=(
            "Facts observed while validating inputs before submission. These were read separately and are "
            "not an atomic snapshot of the state Geti trained against."
        )
    )


class JobWait(_Result):
    """Outcome of a bounded wait for a job to reach a terminal state."""

    job: JobDetail = Field(description="The job's state when the wait ended.")
    reached_terminal: bool = Field(description="Whether a terminal state was observed before the deadline.")
    waited_seconds: float = Field(description="How long this call actually waited.")
    timed_out: bool = Field(
        description="True when the bound elapsed first. The job is still running; this is not a failure."
    )
    polls: int = Field(description="Number of times the job was polled.")


class JobCancellation(_Result):
    """Outcome of a cancellation request."""

    job_id: str = Field(description="Job a cancellation was requested for.")
    project_id: str = Field(description="Owning project.")
    cancellation_requested: bool = Field(description="Whether Geti accepted the cancellation request.")
    terminated: bool = Field(
        description="Whether the job has actually reached a terminal state. False means cancellation is in progress."
    )
    backend_status: str = Field(description="Status Geti reported after the request.")
    state: NormalizedStateName = Field(description="Normalized lifecycle state after the request.")


class MetricValue(BaseModel):
    """One evaluation metric."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Metric name as defined by Geti.")
    value: float = Field(description="Metric value.")
    primary: bool = Field(description="Whether Geti marks this as the headline metric for the task.")


class EvaluationResult(BaseModel):
    """Metrics for one dataset revision and subset."""

    model_config = ConfigDict(extra="forbid")

    dataset_revision_id: str = Field(description="Dataset revision the metrics were computed on.")
    subset: str = Field(description="Subset the metrics were computed on, e.g. 'testing'.")
    metrics: list[MetricValue] = Field(description="Metric values for this evaluation context.")


class ModelVariantSummary(BaseModel):
    """One deployable or trained artifact belonging to a model."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Variant identifier.")
    format: str = Field(description="Artifact format, e.g. 'openvino', 'onnx', 'pytorch'.")
    precision: str = Field(description="Numeric precision, e.g. 'fp32', 'fp16', 'int8'.")
    weights_size_bytes: int | None = Field(default=None, description="Size of the weights file.")
    files_deleted: bool = Field(description="Whether the artifact files were removed from disk.")
    quantized: bool = Field(description="Whether Geti reports quantization metadata for this variant.")
    evaluation_count: int = Field(description="Number of evaluation contexts recorded for this variant.")


class ModelSummary(BaseModel):
    """Compact description of one trained model."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Model identifier.")
    name: str = Field(description="Model name. Untrusted user-supplied text.")
    architecture: str = Field(description="Architecture the model was trained with.")
    parent_revision_id: str | None = Field(default=None, description="Model this one was fine-tuned from.")
    training_status: str | None = Field(default=None, description="Training status Geti reports for the model.")
    dataset_revision_id: str | None = Field(
        default=None, description="Dataset revision trained on. Required context for any metric comparison."
    )
    size_bytes: int | None = Field(default=None, description="Total on-disk size.")
    files_deleted: bool = Field(description="Whether model files were removed from disk.")
    variants: list[ModelVariantSummary] = Field(description="Artifacts belonging to this model.")
    headline_metric: MetricValue | None = Field(
        default=None, description="Primary metric if exactly one unambiguous evaluation exists, else null."
    )


class ModelList(_Result):
    """Models belonging to one authorized project."""

    project_id: str = Field(description="Project the models belong to.")
    models: list[ModelSummary] = Field(description="Models found.")
    total_available: int = Field(description="Number Geti reported before this server's item limit was applied.")
    truncated: bool = Field(description="Whether the list was cut short by the configured item limit.")
    comparison_note: str = Field(
        description=(
            "Metrics from different dataset_revision_id or subset values are not comparable. "
            "Check that context before ranking models against each other."
        )
    )


class ModelResults(_Result):
    """Evaluation results and context for one model."""

    project_id: str = Field(description="Project the model belongs to.")
    model_id: str = Field(description="Model identifier.")
    name: str = Field(description="Model name. Untrusted user-supplied text.")
    architecture: str = Field(description="Architecture the model was trained with.")
    training_status: str | None = Field(default=None, description="Training status Geti reports.")
    training_started_at: str | None = Field(default=None, description="ISO-8601 training start time.")
    training_ended_at: str | None = Field(default=None, description="ISO-8601 training end time.")
    trained_on_dataset_revision_id: str | None = Field(
        default=None, description="Dataset revision the model was trained on."
    )
    evaluations_by_variant: dict[str, list[EvaluationResult]] = Field(
        description="Evaluations keyed by variant identifier, each with its own dataset revision and subset."
    )
    distinct_dataset_revisions: list[str] = Field(
        description="Every dataset revision appearing in these evaluations. More than one means results are mixed."
    )
    distinct_subsets: list[str] = Field(description="Every subset appearing in these evaluations.")
    missing_data: list[str] = Field(
        description="Explicit statements of what Geti did not report, so gaps are not mistaken for zeros."
    )
    truncated: bool = Field(description="Whether evaluations were cut short by the configured item limit.")
