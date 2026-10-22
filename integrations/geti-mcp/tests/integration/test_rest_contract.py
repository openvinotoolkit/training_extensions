# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Contract tests against the Geti OpenAPI document.

These validate the assumptions this adapter hard-codes — endpoint paths, required query
parameters, enum values, and response shapes — and validate the fake payloads used by the
unit tests against the real schemas, so a fake that drifts from Geti fails here.

By default the committed snapshot in ``tests/assets/`` is used, so no running Geti is
needed. Point ``GETI_MCP_OPENAPI_SPEC`` at a freshly generated spec to check a newer
backend:

    cd application/backend && just gen-api-spec --output-path /tmp/geti-openapi.json
    GETI_MCP_OPENAPI_SPEC=/tmp/geti-openapi.json pytest tests/integration/test_rest_contract.py
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from jsonschema import Draft202012Validator
from referencing import Registry
from referencing.jsonschema import DRAFT202012

from geti_mcp.client import API_PREFIX
from geti_mcp.config import BACKEND_MAX_PAGE_SIZE, SUPPORTED_API_VERSIONS
from geti_mcp.lifecycle import _STATUS_MAP
from geti_mcp.service import JOB_TYPES
from tests.conftest import ASSETS, PROJECT_A, FakeGeti, _project, _training_job

#: Every endpoint this adapter calls, as (method, path template).
REQUIRED_OPERATIONS = [
    ("get", "/api/system/info"),
    ("get", "/api/system/devices/training"),
    ("get", "/api/projects"),
    ("get", "/api/projects/{project_id}"),
    ("get", "/api/projects/{project_id}/dataset/statistics"),
    ("get", "/api/projects/{project_id}/dataset/media"),
    ("get", "/api/projects/{project_id}/dataset/media/{media_id}/binary"),
    ("get", "/api/model_architectures"),
    ("get", "/api/projects/{project_id}/training_configuration"),
    ("get", "/api/projects/{project_id}/models"),
    ("get", "/api/jobs"),
    ("post", "/api/jobs"),
    ("get", "/api/jobs/{job_id}"),
    ("post", "/api/jobs/{job_id}:cancel"),
]

#: Base URI the specification is registered under while resolving schema references.
_SPEC_URI = "urn:geti-openapi"


@pytest.fixture(scope="session")
def spec() -> dict[str, Any]:
    """Load the OpenAPI document under test."""
    override = os.environ.get("GETI_MCP_OPENAPI_SPEC")
    path = Path(override) if override else ASSETS / "geti-openapi-snapshot.json"
    if not path.is_file():
        pytest.fail(f"OpenAPI document not found at {path}.")
    return json.loads(path.read_text())


def _validator(spec: dict[str, Any], schema: dict[str, Any]) -> Draft202012Validator:
    """Build a validator whose ``#/components/schemas/...`` references resolve against the spec.

    The schema under test is a fragment of the document, so it is wrapped in a reference to a
    registered copy of the whole specification; otherwise local references would be resolved
    relative to the fragment.
    """
    registry = Registry().with_resource(_SPEC_URI, DRAFT202012.create_resource(spec))
    rebased = json.loads(json.dumps(schema).replace('"#/components/', f'"{_SPEC_URI}#/components/'))
    return Draft202012Validator(rebased, registry=registry)


def _response_schema(spec: dict[str, Any], method: str, path: str) -> dict[str, Any]:
    operation = spec["paths"][path][method]
    for status in ("200", "202"):
        content = operation.get("responses", {}).get(status, {}).get("content", {})
        if "application/json" in content:
            return content["application/json"]["schema"]
    pytest.fail(f"No JSON success response declared for {method.upper()} {path}.")


def _request_schema(spec: dict[str, Any], method: str, path: str) -> dict[str, Any]:
    body = spec["paths"][path][method]["requestBody"]
    return body["content"]["application/json"]["schema"]


def _enum(spec: dict[str, Any], name: str) -> set[str]:
    schema = spec["components"]["schemas"][name]
    return set(schema.get("enum") or [])


class TestSurface:
    def test_the_adapter_targets_the_declared_api_version(self, spec: dict[str, Any]) -> None:
        major_minor = ".".join(spec["info"]["version"].split(".")[:2])
        assert major_minor in SUPPORTED_API_VERSIONS, (
            f"Geti now serves {spec['info']['version']}; update SUPPORTED_API_VERSIONS and re-verify."
        )

    @pytest.mark.parametrize(("method", "path"), REQUIRED_OPERATIONS)
    def test_every_endpoint_the_adapter_calls_exists(self, spec: dict[str, Any], method: str, path: str) -> None:
        assert path in spec["paths"], f"Geti no longer exposes {path}."
        assert method in spec["paths"][path], f"Geti no longer exposes {method.upper()} {path}."

    def test_every_path_is_under_the_api_prefix_the_client_uses(self) -> None:
        assert all(path.startswith(API_PREFIX) for _, path in REQUIRED_OPERATIONS)

    def test_model_architectures_still_requires_the_task_parameter(self, spec: dict[str, Any]) -> None:
        params = spec["paths"]["/api/model_architectures"]["get"]["parameters"]
        task = next(param for param in params if param["name"] == "task")
        assert task["required"] is True

    def test_training_configuration_still_requires_the_architecture_parameter(self, spec: dict[str, Any]) -> None:
        params = spec["paths"]["/api/projects/{project_id}/training_configuration"]["get"]["parameters"]
        architecture = next(param for param in params if param["name"] == "model_architecture_id")
        assert architecture["required"] is True

    def test_job_listing_is_still_unfiltered_so_client_side_filtering_stays_necessary(
        self, spec: dict[str, Any]
    ) -> None:
        params = spec["paths"]["/api/jobs"]["get"].get("parameters", [])
        assert params == [], (
            "GET /api/jobs now accepts parameters; server-side filtering may replace the client-side "
            "authorization filter in list_jobs."
        )

    def test_media_page_size_ceiling_is_still_what_the_config_assumes(self, spec: dict[str, Any]) -> None:
        params = spec["paths"]["/api/projects/{project_id}/dataset/media"]["get"]["parameters"]
        limit = next(param for param in params if param["name"] == "limit")
        assert limit["schema"]["maximum"] == BACKEND_MAX_PAGE_SIZE


class TestEnums:
    def test_job_status_values_are_all_mapped(self) -> None:
        """Verify the job lifecycle mapping against the backend source.

        ``JobStatus`` is an ``IntEnum`` serialised by name, so the specification only types it as
        a bare string. The backend source is the only available contract for its values.
        """
        source = Path(__file__).parents[4] / "application/backend/app/core/jobs/models/job.py"
        if not source.is_file():
            pytest.skip("Backend source is not available in this checkout.")
        tree = ast.parse(source.read_text())
        enum = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "JobStatus")
        names = {
            target.id
            for statement in enum.body
            if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Name)
        }
        assert names == set(_STATUS_MAP), "Geti changed JobStatus; update lifecycle._STATUS_MAP."

    def test_job_type_values_are_all_known(self, spec: dict[str, Any]) -> None:
        assert _enum(spec, "JobType") == set(JOB_TYPES)

    def test_task_types_match_the_documented_set(self, spec: dict[str, Any]) -> None:
        assert _enum(spec, "TaskType") == {"classification", "detection", "instance_segmentation"}

    def test_subset_names_match(self, spec: dict[str, Any]) -> None:
        assert _enum(spec, "DatasetItemSubset") == {"unassigned", "training", "validation", "testing"}

    def test_annotation_status_names_match(self, spec: dict[str, Any]) -> None:
        assert _enum(spec, "DatasetItemAnnotationStatus") == {"with_annotations", "missing_annotations"}


class TestFakePayloadsMatchTheContract:
    """The unit-test fake must produce payloads Geti could actually produce."""

    def test_project_payload(self, spec: dict[str, Any]) -> None:
        schema = _response_schema(spec, "get", "/api/projects/{project_id}")
        _validator(spec, schema).validate(_project(PROJECT_A, "animals"))

    def test_project_list_payload(self, spec: dict[str, Any], fake_geti: FakeGeti) -> None:
        schema = _response_schema(spec, "get", "/api/projects")
        _validator(spec, schema).validate(list(fake_geti.projects.values()))

    def test_dataset_statistics_payload(self, spec: dict[str, Any], fake_geti: FakeGeti) -> None:
        payload = _fake_json(fake_geti, f"/api/projects/{PROJECT_A}/dataset/statistics")
        schema = _response_schema(spec, "get", "/api/projects/{project_id}/dataset/statistics")
        _validator(spec, schema).validate(payload)

    def test_training_configuration_payload(self, spec: dict[str, Any], fake_geti: FakeGeti) -> None:
        payload = _fake_json(
            fake_geti,
            f"/api/projects/{PROJECT_A}/training_configuration?model_architecture_id=x",
        )
        schema = _response_schema(spec, "get", "/api/projects/{project_id}/training_configuration")
        _validator(spec, schema).validate(payload)

    def test_media_page_payload(self, spec: dict[str, Any], fake_geti: FakeGeti) -> None:
        schema = _response_schema(spec, "get", "/api/projects/{project_id}/dataset/media")
        _validator(spec, schema).validate(fake_geti._media_page(10, 0))

    def test_architecture_payload(self, spec: dict[str, Any], fake_geti: FakeGeti) -> None:
        schema = _response_schema(spec, "get", "/api/model_architectures")
        _validator(spec, schema).validate(fake_geti._architectures("detection"))

    def test_model_payload(self, spec: dict[str, Any], fake_geti: FakeGeti) -> None:
        schema = _response_schema(spec, "get", "/api/projects/{project_id}/models")
        _validator(spec, schema).validate([fake_geti._model()])

    def test_training_job_payload(self, spec: dict[str, Any]) -> None:
        schema = _response_schema(spec, "get", "/api/jobs/{job_id}")
        job = _training_job("3fa85f64-5717-4562-b3fc-2c963f66afa6", PROJECT_A, "RUNNING", 42.0)
        _validator(spec, schema).validate(job)

    def test_training_submission_body(self, spec: dict[str, Any]) -> None:
        """The exact body start_training sends must satisfy the request schema."""
        schema = _request_schema(spec, "post", "/api/jobs")
        body = {
            "job_type": "train",
            "project_id": PROJECT_A,
            "parameters": {"device": "cpu", "model_architecture_id": "object-detection-atss-mobilenet-v2"},
        }
        _validator(spec, schema).validate(body)

    def test_training_submission_body_with_optional_parameters(self, spec: dict[str, Any]) -> None:
        schema = _request_schema(spec, "post", "/api/jobs")
        body = {
            "job_type": "train",
            "project_id": PROJECT_A,
            "parameters": {
                "device": "xpu-0",
                "model_architecture_id": "object-detection-atss-mobilenet-v2",
                "parent_model_revision_id": "76e07d18-196e-4e33-bf98-ac1d35dca4cb",
                "dataset_revision_id": "3c6c6d38-1cd8-4458-b759-b9880c048b78",
            },
        }
        _validator(spec, schema).validate(body)


def _fake_json(fake: FakeGeti, url: str) -> Any:
    """Replay a GET through the fake backend and return its decoded payload."""
    request = httpx.Request("GET", f"https://geti.test:7860{url}")
    response = fake._handle(request)
    assert response.status_code == httpx.codes.OK, response.text
    return json.loads(response.content)
