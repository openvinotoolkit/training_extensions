# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures: an in-process fake Geti built on ``httpx.MockTransport``.

Payload shapes here are derived from the generated Geti OpenAPI document, and
``tests/integration/test_rest_contract.py`` validates them against the committed snapshot
of that document, so drift in the real backend surfaces as a contract test failure.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from geti_mcp.authz import Authorizer
from geti_mcp.client import GetiClient
from geti_mcp.config import Limits, Permissions, ServerConfig
from geti_mcp.service import GetiService

ASSETS = Path(__file__).parent / "assets"

PROJECT_A = "7b073838-99d3-42ff-9018-4e901eb047fc"
PROJECT_B = "11111111-2222-3333-4444-555555555555"
LABEL_CAT = "a22d82ba-afa9-4d6e-bbc1-8c8e4002ec29"
LABEL_DOG = "8aa85368-11ba-4507-88f2-6a6704d78ef5"
MEDIA_A = "c1feaabc-da2b-442e-9b3e-55c11c2c2ff3"
MODEL_A = "76e07d18-196e-4e33-bf98-ac1d35dca4cb"
VARIANT_A = "06091f82-5506-41b9-b97f-c761380df870"
REVISION_A = "3c6c6d38-1cd8-4458-b759-b9880c048b78"
REVISION_B = "9d9d9d9d-1111-2222-3333-444444444444"
JOB_A = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
JOB_OTHER_PROJECT = "4fa85f64-5717-4562-b3fc-2c963f66afa7"
JOB_NO_OWNER = "5fa85f64-5717-4562-b3fc-2c963f66afa8"
ARCHITECTURE = "object-detection-atss-mobilenet-v2"

API_VERSION = "3.2.0"


def _project(project_id: str, name: str, task_type: str = "detection") -> dict[str, Any]:
    return {
        "id": project_id,
        "name": name,
        "active_pipeline": False,
        "created_at": "2026-01-01T00:00:00Z",
        "task": {
            "task_type": task_type,
            "exclusive_labels": True,
            "labels": [
                {"id": LABEL_CAT, "name": "cat", "color": "#FF5733", "hotkey": "S"},
                {"id": LABEL_DOG, "name": "dog", "color": "#33FF57", "hotkey": "D"},
            ],
        },
    }


def _training_job(job_id: str, project_id: str, status: str, progress: float = 0.0, **extra: Any) -> dict[str, Any]:
    job: dict[str, Any] = {
        "job_id": job_id,
        "job_type": "train",
        "status": status,
        "progress": progress,
        "metadata": {
            "project": {"id": project_id},
            "model": {
                "id": MODEL_A,
                "name": "ATSS MobileNetV2",
                "architecture": ARCHITECTURE,
                "parent_revision_id": None,
                "dataset_revision_id": REVISION_A,
            },
            "device": {"type": "cpu", "name": "CPU", "memory": None, "index": None},
        },
    }
    job.update(extra)
    return job


def make_image_bytes(width: int, height: int, fmt: str = "JPEG") -> bytes:
    """Build an in-memory test image.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        fmt: Pillow format name.

    Returns:
        bytes: Encoded image data.
    """
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (120, 30, 200)).save(buffer, format=fmt)
    return buffer.getvalue()


class FakeGeti:
    """A scriptable stand-in for the Geti REST API."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.api_version = API_VERSION
        self.projects: dict[str, dict[str, Any]] = {
            PROJECT_A: _project(PROJECT_A, "animals"),
            PROJECT_B: _project(PROJECT_B, "secret-project", "classification"),
        }
        self.jobs: dict[str, dict[str, Any]] = {
            JOB_A: _training_job(JOB_A, PROJECT_A, "RUNNING", 42.0, started_at="2026-01-02T10:00:00Z"),
            JOB_OTHER_PROJECT: _training_job(JOB_OTHER_PROJECT, PROJECT_B, "DONE", 100.0),
            JOB_NO_OWNER: {
                "job_id": JOB_NO_OWNER,
                "job_type": "stage_dataset",
                "status": "RUNNING",
                "progress": 5.0,
                "metadata": {"dataset_id": None, "project_id": None},
            },
        }
        self.media_total = 3
        self.image_bytes = make_image_bytes(1600, 900)
        self.image_content_type = "image/jpeg"
        self.submitted_training: list[dict[str, Any]] = []
        #: Path regex -> callable(request) -> httpx.Response, consulted before normal routing.
        self.overrides: list[tuple[re.Pattern[str], Any]] = []

    def override(self, pattern: str, responder: Any) -> None:
        """Force a response for paths matching ``pattern``."""
        self.overrides.insert(0, (re.compile(pattern), responder))

    def transport(self) -> httpx.MockTransport:
        """Return a transport that routes requests into this fake."""
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        for pattern, responder in self.overrides:
            if pattern.search(path):
                return responder(request)
        return self._route(request, path)

    def _route(self, request: httpx.Request, path: str) -> httpx.Response:
        method = request.method
        params = request.url.params

        if path == "/api/openapi.json":
            return _json(
                {
                    "openapi": "3.1.0",
                    "info": {"title": "Geti", "version": self.api_version},
                    "paths": {"/api/projects": {"get": {}}, "/api/jobs": {"get": {}, "post": {}}},
                }
            )
        if path == "/api/system/info":
            return _json({"license_accepted": True, "platform": "linux"})
        if path == "/api/system/devices/training":
            return _json(
                [
                    {"type": "cpu", "name": "CPU", "memory": None, "index": None},
                    {"type": "xpu", "name": "Intel Arc B580", "memory": 12884901888, "index": 0},
                ]
            )
        if path == "/api/projects" and method == "GET":
            return _json(list(self.projects.values()))
        if path == "/api/model_architectures":
            return _json(self._architectures(params.get("task", "detection")))
        if path == "/api/jobs" and method == "GET":
            return _json(list(self.jobs.values()))
        if path == "/api/jobs" and method == "POST":
            return self._submit_job(request)

        job_match = re.fullmatch(r"/api/jobs/([0-9a-f-]+)(:cancel)?", path)
        if job_match:
            return self._job_route(job_match.group(1), cancel=bool(job_match.group(2)), method=method)

        project_match = re.match(r"/api/projects/([0-9a-f-]+)(/.*)?$", path)
        if project_match:
            return self._project_route(project_match.group(1), project_match.group(2) or "", params)

        return _json({"detail": f"No fake route for {method} {path}"}, status_code=404)

    def _project_route(self, project_id: str, rest: str, params: httpx.QueryParams) -> httpx.Response:
        project = self.projects.get(project_id)
        if project is None:
            return _json({"detail": "Project not found"}, status_code=404)
        if rest == "":
            return _json(project)
        if rest == "/dataset/statistics":
            return _json(
                {
                    "media_counts": {"images": 10, "videos": 1, "video_frames": 312},
                    "annotations_counts": {
                        "annotated_images": 8,
                        "annotated_videos": 1,
                        "annotated_video_frames": 29,
                        "instances": 56,
                        "instances_per_label": [{"label_id": LABEL_CAT, "instances": 56}],
                    },
                }
            )
        if rest == "/dataset/media":
            return _json(self._media_page(int(params.get("limit", 10)), int(params.get("offset", 0))))
        if rest == "/training_configuration":
            return _json(
                {
                    "parameters": [
                        {
                            "key": "training",
                            "name": "Training",
                            "type": "parameter_group",
                            "parameters": [
                                {
                                    "key": "max_epochs",
                                    "name": "Maximum epochs",
                                    "value": 100,
                                    "default_value": 200,
                                    "value_type": "int",
                                    "min_value": 1,
                                    "max_value": 1000,
                                },
                                {
                                    "key": "learning_rate",
                                    "name": "Learning rate",
                                    "value": 0.001,
                                    "default_value": 0.001,
                                    "value_type": "float",
                                },
                                {
                                    "key": "early_stopping",
                                    "name": "Early stopping",
                                    "type": "parameter_group",
                                    "parameters": [
                                        {
                                            "key": "patience",
                                            "name": "Patience",
                                            "value": 10,
                                            "default_value": 10,
                                            "value_type": "int",
                                        }
                                    ],
                                },
                            ],
                        },
                        {
                            "key": "augmentation",
                            "name": "Augmentation",
                            "type": "parameter_group",
                            "parameters": [
                                {
                                    "key": "random_resize_crop",
                                    "name": "Random resize crop",
                                    "type": "parameter_group",
                                    "parameters": [
                                        {
                                            "key": "enable",
                                            "name": "Enable",
                                            "value": True,
                                            "default_value": False,
                                            "value_type": "bool",
                                        }
                                    ],
                                }
                            ],
                        },
                    ]
                }
            )
        if rest == "/models":
            revision = params.get("dataset_revision_id")
            models = [self._model()]
            if revision is not None:
                models = [m for m in models if m["training_info"]["dataset_revision_id"] == revision]
            return _json(models)
        if rest == f"/models/{MODEL_A}":
            return _json(self._model())
        media_binary = re.fullmatch(r"/dataset/media/([0-9a-f-]+)/binary", rest)
        if media_binary:
            return httpx.Response(200, content=self.image_bytes, headers={"content-type": self.image_content_type})
        return _json({"detail": f"No fake route for project sub-path {rest}"}, status_code=404)

    def _job_route(self, job_id: str, *, cancel: bool, method: str) -> httpx.Response:
        job = self.jobs.get(job_id)
        if job is None:
            return _json({"detail": "Job not found"}, status_code=404)
        if cancel and method == "POST":
            job = dict(job)
            job["status"] = "CANCELLING"
            self.jobs[job_id] = job
            return _json(job, status_code=202)
        return _json(job)

    def _submit_job(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.submitted_training.append(body)
        return _json(
            _training_job(JOB_A, body["project_id"], "PENDING", 0.0),
            status_code=202,
        )

    def _media_page(self, limit: int, offset: int) -> dict[str, Any]:
        items = [
            {
                "id": MEDIA_A if index == 0 else f"{index:08d}-0000-0000-0000-000000000000",
                "name": f"img-{index:04d}",
                "type": "image",
                "format": "jpg",
                "width": 1280,
                "height": 720,
                "size": 2211840,
                "source_id": None,
            }
            for index in range(offset, min(offset + limit, self.media_total))
        ]
        return {
            "items": items,
            "pagination": {"offset": offset, "limit": limit, "count": len(items), "total": self.media_total},
        }

    def _architectures(self, task: str) -> dict[str, Any]:
        return {
            "model_architectures": [
                {
                    "id": ARCHITECTURE,
                    "name": "ATSS MobileNetV2",
                    "description": "Balanced detector for general use.",
                    "task": task,
                    "license": "Apache 2.0",
                    "support_status": "active",
                    "capabilities": {"xai": True, "tiling": False},
                    "stats": {
                        "gigaflops": 20.6,
                        "trainable_parameters": 4.2,
                        "benchmark_metrics": {"coco_map_50": 61.3, "coco_map_50_95": 42.1},
                    },
                },
                {
                    "id": "object-detection-dfine-m",
                    "name": "D-FINE-M",
                    "description": "Higher accuracy detector.",
                    "task": task,
                    "license": "Apache 2.0",
                    "support_status": "active",
                    "capabilities": {"xai": False, "tiling": False},
                    "stats": {
                        "gigaflops": 57,
                        "trainable_parameters": 19,
                        "benchmark_metrics": {"coco_map_50": 70.4, "coco_map_50_95": 52.0},
                    },
                },
            ],
            "top_picks": {
                "balance": ARCHITECTURE,
                "speed": "object-detection-dfine-m",
                "accuracy": "object-detection-dfine-m",
            },
        }

    def _model(self) -> dict[str, Any]:
        return {
            "id": MODEL_A,
            "name": "ATSS MobileNetV2 (76e07d18)",
            "architecture": ARCHITECTURE,
            "parent_revision": None,
            "files_deleted": False,
            "size": 18_000_000,
            "training_info": {
                "status": "successful",
                "label_schema_revision": {},
                "start_time": "2026-01-02T10:00:00Z",
                "end_time": "2026-01-02T10:40:00Z",
                "dataset_revision_id": REVISION_A,
            },
            "variants": [
                {
                    "id": VARIANT_A,
                    "format": "openvino",
                    "precision": "fp16",
                    "weights_size": 9_000_000,
                    "files_deleted": False,
                    "quantization_info": None,
                    "optimal_confidence_threshold": 0.35,
                    "evaluations": [
                        {
                            "dataset_revision_id": REVISION_A,
                            "subset": "testing",
                            "metrics": [
                                {"name": "mAP", "value": 0.81, "primary": True},
                                {"name": "Precision", "value": 0.9, "primary": False},
                            ],
                        }
                    ],
                }
            ],
        }


def _json(payload: Any, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


def make_config(
    *,
    allowed_projects: frozenset[str] | None = frozenset({PROJECT_A}),
    permissions: Permissions | None = None,
    limits: Limits | None = None,
) -> ServerConfig:
    """Build a configuration for tests.

    Args:
        allowed_projects: Allowlist, or ``None`` to authorize every project.
        permissions: Capability opt-ins, defaulting to fully read-only.
        limits: Bounds, defaulting to the production defaults.

    Returns:
        ServerConfig: A validated configuration.
    """
    return ServerConfig(
        base_url="https://geti.test:7860",
        allowed_projects=allowed_projects,
        permissions=permissions or Permissions(),
        limits=limits or Limits(),
        ca_bundle=None,
        log_level="DEBUG",
    )


@pytest.fixture
def fake_geti() -> FakeGeti:
    """An in-process fake Geti instance."""
    return FakeGeti()


@pytest.fixture
def make_service(fake_geti: FakeGeti):
    """Factory building a :class:`GetiService` wired to the fake backend."""

    def factory(config: ServerConfig | None = None) -> GetiService:
        resolved = config or make_config()
        client = GetiClient(resolved, transport=fake_geti.transport())
        return GetiService(client, Authorizer(resolved))

    return factory


@pytest.fixture
def service(make_service) -> GetiService:
    """A read-only service authorized for ``PROJECT_A`` only."""
    return make_service()


@pytest.fixture(scope="session")
def openapi_snapshot() -> dict[str, Any]:
    """The committed trimmed snapshot of the Geti OpenAPI contract."""
    return json.loads((ASSETS / "geti-openapi-snapshot.json").read_text())
