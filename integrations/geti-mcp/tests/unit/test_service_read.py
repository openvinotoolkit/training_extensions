# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the read-only inspection tools."""

from __future__ import annotations

import httpx
import pytest

from geti_mcp.config import Limits, Permissions
from geti_mcp.errors import ErrorCode, GetiMcpError
from tests.conftest import (
    ARCHITECTURE,
    LABEL_CAT,
    MEDIA_A,
    MODEL_A,
    PROJECT_A,
    PROJECT_B,
    REVISION_A,
    REVISION_B,
    FakeGeti,
    make_config,
    make_image_bytes,
)

IMAGES_ALLOWED = Permissions(allow_image_access=True)


class TestConnectionInfo:
    async def test_reports_capabilities_and_scope(self, service) -> None:
        info = await service.connection_info()
        assert info.reachable is True
        assert info.api_version == "3.2.0"
        assert info.compatible is True
        assert info.permissions.allow_training is False
        assert info.project_scope.policy == "allowlist"
        assert info.project_scope.authorized_project_ids == [PROJECT_A]

    async def test_lists_training_devices_for_start_training(self, service) -> None:
        info = await service.connection_info()
        assert [device.device for device in info.training_devices] == ["cpu", "xpu-0"]

    async def test_unreachable_backend_is_reported_not_raised(self, fake_geti: FakeGeti, make_service) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        fake_geti.override(r".*", refuse)
        info = await make_service().connection_info()
        assert info.reachable is False
        assert info.warnings

    async def test_states_which_api_versions_were_tested(self, service) -> None:
        info = await service.connection_info()
        assert info.tested_api_versions == ["3.2"]

    async def test_untested_backend_version_is_surfaced_as_a_warning(self, fake_geti: FakeGeti, make_service) -> None:
        fake_geti.api_version = "9.9.9"
        info = await make_service().connection_info()
        assert info.compatible is False
        assert any("9.9.9" in warning for warning in info.warnings)


class TestProjects:
    async def test_only_authorized_projects_are_listed(self, service) -> None:
        projects = await service.list_projects()
        assert [project.id for project in projects.projects] == [PROJECT_A]
        assert projects.total_authorized == 1

    async def test_unauthorized_project_names_are_not_disclosed(self, service) -> None:
        projects = await service.list_projects()
        assert "secret-project" not in projects.model_dump_json()

    async def test_all_projects_mode_lists_everything(self, make_service) -> None:
        projects = await make_service(make_config(allowed_projects=None)).list_projects()
        assert {project.id for project in projects.projects} == {PROJECT_A, PROJECT_B}

    async def test_project_detail_includes_labels(self, service) -> None:
        project = await service.get_project(PROJECT_A)
        assert project.task_type == "detection"
        assert [label.name for label in project.labels] == ["cat", "dog"]
        assert project.exclusive_labels is True

    async def test_unauthorized_project_is_denied_before_any_request(self, service, fake_geti: FakeGeti) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await service.get_project(PROJECT_B)
        assert excinfo.value.code is ErrorCode.PERMISSION_DENIED
        assert fake_geti.requests == []


class TestDatasetStatistics:
    async def test_reports_backend_counts_verbatim(self, service) -> None:
        stats = await service.dataset_statistics(PROJECT_A)
        assert (stats.images, stats.videos, stats.video_frames) == (10, 1, 312)
        assert stats.annotated_images == 8
        assert stats.total_instances == 56

    async def test_instances_are_broken_down_per_label(self, service) -> None:
        stats = await service.dataset_statistics(PROJECT_A)
        counts = {entry.label_id: entry.instances for entry in stats.instances_per_label}
        assert counts[LABEL_CAT] == 56

    async def test_findings_are_warnings_not_a_trainability_verdict(self, service) -> None:
        stats = await service.dataset_statistics(PROJECT_A)
        assert not hasattr(stats, "trainable")
        assert not hasattr(stats, "ready_to_train")
        assert isinstance(stats.readiness_warnings, list)

    async def test_unannotated_dataset_produces_a_warning(self, fake_geti: FakeGeti, make_service) -> None:
        fake_geti.override(
            r"/statistics$",
            lambda _: httpx.Response(
                200,
                json={
                    "media_counts": {"images": 4, "videos": 0, "video_frames": 0},
                    "annotations_counts": {
                        "annotated_images": 0,
                        "annotated_videos": 0,
                        "annotated_video_frames": 0,
                        "instances": 0,
                        "instances_per_label": [],
                    },
                },
            ),
        )
        stats = await make_service().dataset_statistics(PROJECT_A)
        assert stats.readiness_warnings


class TestMedia:
    async def test_page_size_is_clamped_to_the_configured_limit(self, service, fake_geti: FakeGeti) -> None:
        await service.list_media(PROJECT_A, limit=10_000)
        assert int(fake_geti.requests[-1].url.params["limit"]) <= 50

    async def test_pagination_is_reported_for_continuation(self, service) -> None:
        page = await service.list_media(PROJECT_A, offset=0, limit=2)
        assert (page.count, page.total, page.next_offset) == (2, 3, 2)

    async def test_last_page_has_no_continuation(self, service) -> None:
        page = await service.list_media(PROJECT_A, offset=2, limit=2)
        assert page.next_offset is None

    async def test_never_returns_image_content(self, service) -> None:
        page = await service.list_media(PROJECT_A)
        assert "base64" not in page.model_dump_json()

    @pytest.mark.parametrize("bad", ["annotated", "ALL", "with-annotations"])
    async def test_invalid_annotation_status_names_the_valid_options(self, service, bad: str) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await service.list_media(PROJECT_A, annotation_status=bad)
        assert excinfo.value.code is ErrorCode.INVALID_INPUT
        assert "with_annotations" in excinfo.value.message

    async def test_invalid_subset_is_rejected(self, service) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await service.list_media(PROJECT_A, subsets=["train"])
        assert excinfo.value.code is ErrorCode.INVALID_INPUT

    async def test_label_filter_must_be_uuids(self, service) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await service.list_media(PROJECT_A, label_ids=["cat"])
        assert excinfo.value.code is ErrorCode.INVALID_INPUT

    async def test_valid_filters_are_forwarded(self, service, fake_geti: FakeGeti) -> None:
        await service.list_media(
            PROJECT_A, annotation_status="missing_annotations", subsets=["training"], label_ids=[LABEL_CAT]
        )
        params = fake_geti.requests[-1].url.params
        assert params["annotation_status"] == "missing_annotations"
        assert params.get_list("subsets") == ["training"]
        assert params.get_list("labels") == [LABEL_CAT]


class TestArchitectures:
    async def test_lists_only_architectures_for_the_project_task(self, service, fake_geti: FakeGeti) -> None:
        result = await service.list_architectures(PROJECT_A)
        assert fake_geti.requests[-1].url.params["task"] == "detection"
        assert result.task_type == "detection"
        assert ARCHITECTURE in [item.id for item in result.architectures]

    async def test_marks_the_backend_recommendation_without_choosing(self, service) -> None:
        result = await service.list_architectures(PROJECT_A)
        recommended = {item.id: item.recommended_for for item in result.architectures}
        assert recommended[ARCHITECTURE] == ["balance"]
        assert recommended["object-detection-dfine-m"] == ["accuracy", "speed"]

    async def test_benchmark_metrics_are_passed_through_unnormalized(self, service) -> None:
        result = await service.list_architectures(PROJECT_A)
        atss = next(item for item in result.architectures if item.id == ARCHITECTURE)
        assert atss.benchmark_metrics is not None
        assert atss.benchmark_metrics.coco_map_50_95 == 42.1
        assert atss.gigaflops == 20.6


class TestTrainingConfiguration:
    async def test_returns_flattened_effective_values(self, service) -> None:
        config = await service.training_configuration(PROJECT_A, ARCHITECTURE)
        entries = {entry.key: entry.value for entry in config.entries}
        assert entries["training.max_epochs"] == "100"
        assert entries["training.early_stopping.patience"] == "10"

    async def test_reports_defaults_and_bounds_so_overrides_can_be_judged(self, service) -> None:
        config = await service.training_configuration(PROJECT_A, ARCHITECTURE)
        epochs = next(entry for entry in config.entries if entry.key == "training.max_epochs")
        assert epochs.name == "Maximum epochs"
        assert epochs.default_value == "200"
        assert (epochs.min_value, epochs.max_value) == (1.0, 1000.0)

    async def test_states_that_values_are_a_non_atomic_read(self, service) -> None:
        config = await service.training_configuration(PROJECT_A, ARCHITECTURE)
        assert config.observed_at
        assert "atomic" in config.note.lower()

    async def test_output_is_bounded(self, make_service, fake_geti: FakeGeti) -> None:
        deep = [
            {
                "key": "group",
                "name": "Group",
                "type": "parameter_group",
                "parameters": [
                    {"key": f"key_{index}", "name": f"Key {index}", "value": index, "value_type": "int"}
                    for index in range(500)
                ],
            }
        ]
        fake_geti.override(r"/training_configuration$", lambda _: httpx.Response(200, json={"parameters": deep}))
        config = await make_service(make_config(limits=Limits(max_items=20))).training_configuration(
            PROJECT_A, ARCHITECTURE
        )
        assert len(config.entries) == 20
        assert config.truncated is True

    async def test_a_malformed_configuration_yields_no_entries_rather_than_garbage(
        self, make_service, fake_geti: FakeGeti
    ) -> None:
        fake_geti.override(r"/training_configuration$", lambda _: httpx.Response(200, json={"unexpected": 1}))
        config = await make_service(make_config()).training_configuration(PROJECT_A, ARCHITECTURE)
        assert config.entries == []


class TestModels:
    async def test_lists_models_with_variants(self, service) -> None:
        models = await service.list_models(PROJECT_A)
        assert len(models.models) == 1
        assert models.models[0].architecture == ARCHITECTURE
        assert [variant.format for variant in models.models[0].variants] == ["openvino"]

    async def test_dataset_revision_filter_is_forwarded(self, service, fake_geti: FakeGeti) -> None:
        result = await service.list_models(PROJECT_A, REVISION_B)
        assert fake_geti.requests[-1].url.params["dataset_revision_id"] == REVISION_B
        assert result.models == []

    async def test_headline_metric_is_only_offered_when_unambiguous(self, service) -> None:
        models = await service.list_models(PROJECT_A)
        headline = models.models[0].headline_metric
        assert headline is not None
        assert headline.name == "mAP"

    async def test_comparability_is_stated_on_every_listing(self, service) -> None:
        models = await service.list_models(PROJECT_A)
        assert "dataset_revision_id" in models.comparison_note

    async def test_metrics_carry_their_evaluation_context(self, service) -> None:
        results = await service.model_results(PROJECT_A, MODEL_A)
        evaluations = [item for group in results.evaluations_by_variant.values() for item in group]
        assert evaluations[0].dataset_revision_id == REVISION_A
        assert evaluations[0].subset == "testing"
        assert {metric.name for metric in evaluations[0].metrics} == {"mAP", "Precision"}

    async def test_mixed_evaluation_contexts_are_enumerated(self, service) -> None:
        results = await service.model_results(PROJECT_A, MODEL_A)
        assert results.distinct_dataset_revisions == [REVISION_A]
        assert results.distinct_subsets == ["testing"]

    async def test_absent_metrics_are_reported_not_invented(self, service, fake_geti: FakeGeti) -> None:
        fake_geti.override(
            r"/models/",
            lambda _: httpx.Response(
                200,
                json={
                    "id": MODEL_A,
                    "name": "untested",
                    "architecture": ARCHITECTURE,
                    "parent_revision": None,
                    "files_deleted": False,
                    "size": None,
                    "training_info": {"status": "successful", "dataset_revision_id": None},
                    "variants": [
                        {
                            "id": "x",
                            "format": "openvino",
                            "precision": "fp16",
                            "weights_size": None,
                            "files_deleted": False,
                            "quantization_info": None,
                            "optimal_confidence_threshold": None,
                            "evaluations": [],
                        }
                    ],
                },
            ),
        )
        results = await service.model_results(PROJECT_A, MODEL_A)
        assert results.evaluations_by_variant == {}
        assert results.distinct_dataset_revisions == []
        assert results.missing_data


class TestImageDisclosure:
    async def test_previews_require_an_explicit_opt_in(self, service, fake_geti: FakeGeti) -> None:
        with pytest.raises(GetiMcpError) as excinfo:
            await service.view_media(PROJECT_A, MEDIA_A, None)
        assert excinfo.value.code is ErrorCode.PERMISSION_DENIED
        assert fake_geti.requests == []

    async def test_preview_is_downscaled_and_reports_the_transform(self, make_service) -> None:
        service = make_service(make_config(permissions=IMAGES_ALLOWED))
        preview = await service.view_media(PROJECT_A, MEDIA_A, None)
        assert (preview.transform.source_width, preview.transform.source_height) == (1600, 900)
        assert max(preview.transform.preview_width, preview.transform.preview_height) <= 768
        assert preview.transform.applied is True
        assert 0 < preview.transform.scale < 1

    async def test_preview_stays_within_the_byte_budget(self, make_service) -> None:
        service = make_service(make_config(permissions=IMAGES_ALLOWED, limits=Limits(preview_max_bytes=20_000)))
        preview = await service.view_media(PROJECT_A, MEDIA_A, None)
        assert preview.preview_bytes <= 20_000

    async def test_preview_carries_a_disclosure_notice(self, make_service) -> None:
        service = make_service(make_config(permissions=IMAGES_ALLOWED))
        preview = await service.view_media(PROJECT_A, MEDIA_A, None)
        assert preview.disclosure_notice

    async def test_small_image_is_not_upscaled(self, make_service, fake_geti: FakeGeti) -> None:
        fake_geti.image_bytes = make_image_bytes(64, 48)
        service = make_service(make_config(permissions=IMAGES_ALLOWED))
        preview = await service.view_media(PROJECT_A, MEDIA_A, None)
        assert (preview.transform.preview_width, preview.transform.preview_height) == (64, 48)
        assert preview.transform.applied is False
        assert preview.transform.scale == 1.0

    async def test_unsupported_media_type_is_refused_clearly(self, make_service, fake_geti: FakeGeti) -> None:
        fake_geti.image_content_type = "application/zip"
        fake_geti.image_bytes = b"PK\x03\x04not-an-image"
        service = make_service(make_config(permissions=IMAGES_ALLOWED))
        with pytest.raises(GetiMcpError) as excinfo:
            await service.view_media(PROJECT_A, MEDIA_A, None)
        assert excinfo.value.code is ErrorCode.UNSUPPORTED_MEDIA

    async def test_oversized_source_image_is_refused_before_decoding(self, make_service, fake_geti: FakeGeti) -> None:
        fake_geti.override(
            r"/binary$",
            lambda _: httpx.Response(200, content=b"x" * 200_000, headers={"content-type": "image/jpeg"}),
        )
        service = make_service(make_config(permissions=IMAGES_ALLOWED, limits=Limits(max_response_bytes=50_000)))
        with pytest.raises(GetiMcpError) as excinfo:
            await service.view_media(PROJECT_A, MEDIA_A, None)
        assert excinfo.value.code is ErrorCode.LIMIT_EXCEEDED

    async def test_frame_index_is_forwarded_for_videos(self, make_service, fake_geti: FakeGeti) -> None:
        service = make_service(make_config(permissions=IMAGES_ALLOWED))
        await service.view_media(PROJECT_A, MEDIA_A, 42)
        assert fake_geti.requests[-1].url.params["frame_index"] == "42"
