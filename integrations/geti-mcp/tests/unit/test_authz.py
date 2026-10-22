# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the authorization layer, including bypass attempts and fail-closed ownership."""

from __future__ import annotations

import pytest

from geti_mcp.authz import Authorizer, owning_project_id
from geti_mcp.config import Limits, Permissions
from geti_mcp.errors import ErrorCode, GetiMcpError
from tests.conftest import PROJECT_A, PROJECT_B, make_config


@pytest.fixture
def authorizer() -> Authorizer:
    return Authorizer(make_config())


class TestProjectScope:
    def test_allows_listed_project(self, authorizer: Authorizer) -> None:
        assert authorizer.require_project(PROJECT_A) == PROJECT_A

    def test_denies_unlisted_project(self, authorizer: Authorizer) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            authorizer.require_project(PROJECT_B)
        assert excinfo.value.code is ErrorCode.PERMISSION_DENIED

    def test_denial_does_not_disclose_whether_the_project_exists(self, authorizer: Authorizer) -> None:
        unknown = "00000000-0000-0000-0000-000000000000"
        with pytest.raises(GetiMcpError) as unknown_error:
            authorizer.require_project(unknown)
        with pytest.raises(GetiMcpError) as existing_error:
            authorizer.require_project(PROJECT_B)
        assert unknown_error.value.code is existing_error.value.code

    def test_case_and_brace_variants_resolve_to_the_same_project(self, authorizer: Authorizer) -> None:
        assert authorizer.require_project(PROJECT_A.upper()) == PROJECT_A
        assert authorizer.require_project("{" + PROJECT_A + "}") == PROJECT_A

    @pytest.mark.parametrize(
        "attempt",
        [
            "../../../etc/passwd",
            f"{PROJECT_A}/../{PROJECT_B}",
            f"{PROJECT_B}%2F..%2F{PROJECT_A}",
            f"{PROJECT_A}\n{PROJECT_B}",
            f"{PROJECT_A} or 1=1",
            "",
            "*",
        ],
    )
    def test_traversal_and_injection_attempts_are_rejected_as_invalid(
        self, authorizer: Authorizer, attempt: str
    ) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            authorizer.require_project(attempt)
        assert excinfo.value.code is ErrorCode.INVALID_INPUT

    def test_all_projects_mode_still_requires_a_valid_uuid(self) -> None:
        authorizer = Authorizer(make_config(allowed_projects=None))
        assert authorizer.require_project(PROJECT_B) == PROJECT_B
        with pytest.raises(GetiMcpError):
            authorizer.require_project("not-a-uuid")


class TestCapabilities:
    def test_all_capabilities_denied_by_default(self, authorizer: Authorizer) -> None:
        for require in (
            authorizer.require_training,
            authorizer.require_job_cancellation,
            authorizer.require_image_access,
        ):
            with pytest.raises(GetiMcpError) as excinfo:
                require()
            assert excinfo.value.code is ErrorCode.PERMISSION_DENIED

    def test_enabling_one_capability_does_not_enable_the_others(self) -> None:
        authorizer = Authorizer(make_config(permissions=Permissions(allow_training=True)))
        assert authorizer.require_training() is None
        with pytest.raises(GetiMcpError):
            authorizer.require_job_cancellation()
        with pytest.raises(GetiMcpError):
            authorizer.require_image_access()

    def test_denial_tells_the_model_the_operator_must_act(self, authorizer: Authorizer) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            authorizer.require_training()
        assert "operator" in (excinfo.value.guidance or "").lower()


class TestJobOwnership:
    def test_resolves_training_job_owner(self) -> None:
        assert owning_project_id({"metadata": {"project": {"id": PROJECT_A}}}) == PROJECT_A

    def test_resolves_import_job_owner(self) -> None:
        assert owning_project_id({"metadata": {"project_id": PROJECT_A}}) == PROJECT_A

    @pytest.mark.parametrize(
        "job",
        [
            {},
            {"metadata": None},
            {"metadata": {}},
            {"metadata": {"project_id": None}},
            {"metadata": {"project": None}},
            {"metadata": {"project": {"id": None}}},
            {"metadata": {"project": "not-an-object"}},
            {"metadata": ["unexpected"]},
        ],
    )
    def test_indeterminate_ownership_returns_none(self, job: dict) -> None:
        assert owning_project_id(job) is None

    def test_indeterminate_ownership_fails_closed(self, authorizer: Authorizer) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            authorizer.require_authorized_job({"metadata": {}})
        assert excinfo.value.code is ErrorCode.OWNERSHIP_UNKNOWN

    def test_indeterminate_ownership_fails_closed_even_with_all_projects(self) -> None:
        authorizer = Authorizer(make_config(allowed_projects=None))
        with pytest.raises(GetiMcpError) as excinfo:
            authorizer.require_authorized_job({"metadata": {}})
        assert excinfo.value.code is ErrorCode.OWNERSHIP_UNKNOWN

    def test_job_in_another_project_is_denied(self, authorizer: Authorizer) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            authorizer.require_authorized_job({"metadata": {"project": {"id": PROJECT_B}}})
        assert excinfo.value.code is ErrorCode.PERMISSION_DENIED


class TestBounds:
    def test_default_page_size_is_conservative(self, authorizer: Authorizer) -> None:
        assert authorizer.clamp_limit(None) <= authorizer.config.limits.max_page_size

    @pytest.mark.parametrize("requested", [-100, 0, 1, 10_000])
    def test_page_size_is_always_within_bounds(self, authorizer: Authorizer, requested: int) -> None:
        clamped = authorizer.clamp_limit(requested)
        assert 1 <= clamped <= authorizer.config.limits.max_page_size

    @pytest.mark.parametrize("requested", [-5.0, 0.0, 0.1, 10_000.0])
    def test_wait_is_always_within_bounds(self, authorizer: Authorizer, requested: float) -> None:
        clamped = authorizer.clamp_wait(requested)
        assert 0 < clamped <= authorizer.config.limits.max_wait_seconds

    def test_clamping_respects_a_stricter_operator_limit(self) -> None:
        authorizer = Authorizer(make_config(limits=Limits(max_page_size=3, max_wait_seconds=2.0)))
        assert authorizer.clamp_limit(100) == 3
        assert authorizer.clamp_wait(100.0) == 2.0
