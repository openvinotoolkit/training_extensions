# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, call, patch
from uuid import uuid4

import pytest
from fastapi import status

from app.api.dependencies import (
    get_inference_server,
    get_media_service,
    get_model_service,
    get_training_configuration_service,
)
from app.api.schemas import TrainingConfigurationView
from app.models import DatasetItemSubset, EvaluationResult, ModelRevision, ModelVariant, TrainingInfo, TrainingStatus
from app.models.model_manifest import License
from app.models.model_revision import ModelFormat, ModelPrecision
from app.services import MediaService, ModelService, ResourceInUseError, ResourceNotFoundError, ResourceType
from app.services.inference import InferenceServer
from app.services.training_configuration_service import TrainingConfigurationService


@pytest.fixture
def fxt_model() -> ModelRevision:
    model_revision_id = uuid4()
    dataset_revision_id = uuid4()
    openvino_variant_id = uuid4()
    return ModelRevision(
        id=model_revision_id,
        name="YOLOX-X (abc123)",
        architecture="object-detection-yolox-x",
        parent_revision=uuid4(),
        training_info=TrainingInfo(
            status=TrainingStatus.NOT_STARTED,
            label_schema_revision={"labels": [{"name": "dog", "id": "05db02c9-6a54-4089-bd8e-b85dfe3bec03"}]},
            dataset_revision_id=dataset_revision_id,
        ),
        variants=[
            ModelVariant(
                id=openvino_variant_id,
                model_revision_id=model_revision_id,
                format=ModelFormat.OPENVINO,
                precision=ModelPrecision.FP16,
                evaluations=[
                    EvaluationResult(
                        model_revision_id=model_revision_id,
                        model_variant_id=openvino_variant_id,
                        dataset_revision_id=dataset_revision_id,
                        subset=DatasetItemSubset.TESTING,
                        metrics={
                            "map": 0.85,
                            "map_50": 0.90,
                            "map_75": 0.80,
                            "mar_1": 0.88,
                            "mar_10": 0.90,
                            "mar_100": 0.92,
                        },
                    )
                ],
            )
        ],
        files_deleted=False,
    )


@pytest.fixture
def fxt_model_variants() -> list[ModelVariant]:
    model_revision_id = uuid4()
    return [
        ModelVariant(
            id=uuid4(),
            model_revision_id=model_revision_id,
            format=ModelFormat.OPENVINO,
            precision=ModelPrecision.FP16,
            weights_size=100000,
            optimal_confidence_threshold=0.35,
        ),
        ModelVariant(
            id=uuid4(),
            model_revision_id=model_revision_id,
            format=ModelFormat.ONNX,
            precision=ModelPrecision.FP16,
            weights_size=100100,
            optimal_confidence_threshold=0.35,
        ),
        ModelVariant(
            id=uuid4(),
            model_revision_id=model_revision_id,
            format=ModelFormat.PYTORCH,
            precision=ModelPrecision.FP32,
            weights_size=100200,
        ),
    ]


@pytest.fixture
def fxt_model_service(fxt_app) -> MagicMock:
    model_service = MagicMock(spec=ModelService)
    fxt_app.dependency_overrides[get_model_service] = lambda: model_service
    return model_service


@pytest.fixture(autouse=True)
def fxt_media_service(fxt_app) -> MagicMock:
    media_service = MagicMock(spec=MediaService)
    media_service.list_media.return_value = []
    fxt_app.dependency_overrides[get_media_service] = lambda: media_service
    return media_service


@pytest.fixture
def fxt_training_configuration_service(fxt_app) -> MagicMock:
    training_configuration_service = MagicMock(spec=TrainingConfigurationService)
    fxt_app.dependency_overrides[get_training_configuration_service] = lambda: training_configuration_service
    return training_configuration_service


@pytest.fixture(autouse=True)
def fxt_inference_server(fxt_app) -> MagicMock:
    inference_server = MagicMock(spec=InferenceServer)
    fxt_app.dependency_overrides[get_inference_server] = lambda: inference_server
    return inference_server


class TestModelEndpoints:
    def test_list_model_success(self, fxt_model, fxt_model_variants, fxt_get_project, fxt_model_service, fxt_client):
        fxt_model_service.list_models.return_value = [fxt_model] * 2
        fxt_model_service.get_model_size_in_bytes.side_effect = [300200] * 2
        fxt_model_service.get_model_variants.side_effect = [fxt_model_variants, fxt_model_variants]

        dataset_revision_id = uuid4()
        response = fxt_client.get(
            f"/api/projects/{fxt_get_project.id}/models?dataset_revision_id={dataset_revision_id}"
        )

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 2
        for model_view in response.json():
            thresholds_by_format = {
                variant["format"]: variant["optimal_confidence_threshold"] for variant in model_view["variants"]
            }
            assert thresholds_by_format == {"openvino": 0.35, "onnx": 0.35, "pytorch": None}
        fxt_model_service.list_models.assert_called_once_with(
            project_id=fxt_get_project.id, dataset_revision_id=dataset_revision_id
        )
        fxt_model_service.get_model_size_in_bytes.assert_has_calls(
            [
                call(project_id=fxt_get_project.id, model_id=fxt_model.id),
                call(project_id=fxt_get_project.id, model_id=fxt_model.id),
            ]
        )
        fxt_model_service.get_model_variants.assert_has_calls(
            [
                call(project_id=fxt_get_project.id, model_id=fxt_model.id),
                call(project_id=fxt_get_project.id, model_id=fxt_model.id),
            ]
        )

    def test_list_model_project_not_found(self, fxt_model, fxt_get_project, fxt_model_service, fxt_client):
        project_id = uuid4()
        fxt_model_service.list_models.side_effect = ResourceNotFoundError(ResourceType.PROJECT, str(project_id))

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.list_models.assert_called_once_with(project_id=fxt_get_project.id, dataset_revision_id=None)

    def test_list_model_invalid_id(self, fxt_model, fxt_model_service, fxt_client):
        response = fxt_client.get("/api/projects/invalid-id/models")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_model_service.list_models.assert_not_called()

    def test_get_model_success(self, fxt_model, fxt_model_variants, fxt_get_project, fxt_model_service, fxt_client):
        fxt_model_service.get_model.return_value = fxt_model
        fxt_model_service.get_model_size_in_bytes.return_value = 300100
        fxt_model_service.get_model_variants.return_value = fxt_model_variants

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}")

        assert response.status_code == status.HTTP_200_OK
        fxt_model_service.get_model.assert_called_once_with(project_id=fxt_get_project.id, model_id=fxt_model.id)
        fxt_model_service.get_model_size_in_bytes.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=fxt_model.id
        )
        fxt_model_service.get_model_variants.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=fxt_model.id
        )
        response_data = response.json()
        assert response_data["id"] == str(fxt_model.id)
        assert response_data["name"] == fxt_model.name
        assert response_data["architecture"] == fxt_model.architecture
        assert response_data["parent_revision"] == str(fxt_model.parent_revision)
        assert response_data["training_info"]["status"] == fxt_model.training_info.status.value
        assert response_data["training_info"]["label_schema_revision"] == fxt_model.training_info.label_schema_revision
        assert response_data["training_info"]["dataset_revision_id"] == str(fxt_model.training_info.dataset_revision_id)
        assert len(response_data["variants"]) == len(fxt_model_variants)
        for resp_variant, expected_variant in zip(response_data["variants"], fxt_model_variants):
            assert resp_variant["id"] == str(expected_variant.id)
            assert resp_variant["format"] == expected_variant.format
            assert resp_variant["precision"] == expected_variant.precision
            assert resp_variant["weights_size"] == expected_variant.weights_size
            assert resp_variant["optimal_confidence_threshold"] == expected_variant.optimal_confidence_threshold
        assert response_data["size"] == 300100

    @pytest.mark.parametrize(
        "http_method, service_method",
        [
            ("get", "get_model"),
            ("delete", "delete_model"),
            ("patch", "rename_model"),
        ],
    )
    def test_model_invalid_ids(self, http_method, service_method, fxt_get_project, fxt_model_service, fxt_client):
        response = getattr(fxt_client, http_method)(f"/api/projects/{fxt_get_project.id}/models/invalid-id")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        getattr(fxt_model_service, service_method).assert_not_called()

    def test_get_model_not_found(self, fxt_get_project, fxt_model_service, fxt_client):
        model_id = uuid4()
        fxt_model_service.get_model.side_effect = ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.get_model.assert_called_once_with(project_id=fxt_get_project.id, model_id=model_id)

    def test_delete_model_success(
        self, fxt_get_project, fxt_model, fxt_model_service, fxt_inference_server, fxt_client
    ):
        response = fxt_client.delete(f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_model_service.delete_model.assert_called_once_with(project_id=fxt_get_project.id, model_id=fxt_model.id)
        fxt_inference_server.invalidate_model.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=fxt_model.id
        )

    def test_delete_model_files_only_success(
        self, fxt_get_project, fxt_model, fxt_model_service, fxt_inference_server, fxt_client
    ):
        response = fxt_client.delete(f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}?files_only=true")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_model_service.delete_model_files.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=fxt_model.id
        )
        fxt_inference_server.invalidate_model.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=fxt_model.id
        )

    def test_delete_model_not_found(self, fxt_get_project, fxt_model_service, fxt_client):
        model_id = uuid4()
        fxt_model_service.delete_model.side_effect = ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        response = fxt_client.delete(f"/api/projects/{fxt_get_project.id}/models/{model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.delete_model.assert_called_once_with(project_id=fxt_get_project.id, model_id=model_id)

    def test_delete_model_in_use(self, fxt_get_project, fxt_model_service, fxt_client):
        model_id = uuid4()
        err = ResourceInUseError(ResourceType.MODEL, str(model_id))
        fxt_model_service.delete_model.side_effect = err

        response = fxt_client.delete(f"/api/projects/{fxt_get_project.id}/models/{model_id}")

        assert response.status_code == status.HTTP_409_CONFLICT
        assert str(err) == response.json()["detail"]
        fxt_model_service.delete_model.assert_called_once_with(project_id=fxt_get_project.id, model_id=model_id)

    @pytest.mark.parametrize(
        "model_format, model_precision",
        [
            (ModelFormat.OPENVINO, "fp16"),
            (ModelFormat.ONNX, "fp16"),
            (ModelFormat.PYTORCH, "fp32"),
        ],
    )
    def test_download_model_binary_success(
        self, model_format, model_precision, fxt_get_project, fxt_model, fxt_model_service, fxt_client, tmp_path
    ):
        import zipfile
        from io import BytesIO

        model_variant_id = uuid4()
        # Create mock model files
        model_dir = tmp_path / "models" / str(fxt_model.id) / "variants" / str(model_variant_id)
        model_dir.mkdir(parents=True)
        bin_content = b"binary model data"
        if model_format == ModelFormat.OPENVINO:
            xml_content = "<xml>model data</xml>"
            (model_dir / "model.xml").write_text(xml_content)
            (model_dir / "model.bin").write_bytes(bin_content)
        elif model_format == ModelFormat.ONNX:
            (model_dir / "model.onnx").write_bytes(bin_content)
        elif model_format == ModelFormat.PYTORCH:
            (model_dir / "model.pt").write_bytes(bin_content)

        fxt_model_service.get_model_binary_files.return_value = True, tuple(model_dir.glob("*"))
        fxt_model_service.get_variant.return_value = ModelVariant(
            id=model_variant_id,
            model_revision_id=fxt_model.id,
            format=model_format,
            precision=ModelPrecision(model_precision),
        )
        fxt_model_service.get_model_license.return_value = License(
            name="AGPL-3.0", url="https://www.ultralytics.com/legal/agpl-3-0-software-license"
        )
        fxt_model_service.get_model_attribution.return_value = None

        response = fxt_client.get(
            f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}/variants/{model_variant_id}/binary"
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/zip"
        assert "content-disposition" in response.headers
        assert (
            f"model-{str(fxt_model.id).split('-')[0]}-{model_format.value}-{model_precision}.zip"
            in response.headers["content-disposition"]
        )

        # Verify zip file contents
        zip_data = BytesIO(response.content)
        with zipfile.ZipFile(zip_data, "r") as zip_file:
            namelist = zip_file.namelist()
            if model_format == ModelFormat.OPENVINO:
                assert "model.xml" in namelist
                assert "model.bin" in namelist
                assert zip_file.read("model.xml").decode() == xml_content  # pyrefly: ignore[unbound-name]
                assert zip_file.read("model.bin") == bin_content
            elif model_format == ModelFormat.ONNX:
                assert "model.onnx" in namelist
                assert zip_file.read("model.onnx") == bin_content
            elif model_format == ModelFormat.PYTORCH:
                assert "model.pt" in namelist
                assert zip_file.read("model.pt") == bin_content

            # Auxiliary deployment files are bundled only for the deployable formats.
            if model_format in (ModelFormat.OPENVINO, ModelFormat.ONNX):
                assert "demo.py" in namelist
                assert "demo_async.py" in namelist
                assert "pyproject.toml" in namelist
                assert "README.md" in namelist
                assert "LICENSE" in namelist
                assert "## Licensing" in zip_file.read("README.md").decode()
                fxt_model_service.get_model_attribution.assert_called_once_with(
                    project_id=fxt_get_project.id, model_id=fxt_model.id
                )
            else:
                assert "demo.py" not in namelist
                assert "demo_async.py" not in namelist
                assert "pyproject.toml" not in namelist
                assert "README.md" not in namelist
                assert "LICENSE" not in namelist

        fxt_model_service.get_model_binary_files.assert_called_once_with(
            project_id=fxt_get_project.id,
            model_id=fxt_model.id,
            model_variant_id=model_variant_id,
        )

    def test_download_model_binary_not_found(self, fxt_get_project, fxt_model_service, fxt_client):
        model_id = uuid4()
        model_variant_id = uuid4()
        fxt_model_service.get_model_binary_files.side_effect = ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        response = fxt_client.get(
            f"/api/projects/{fxt_get_project.id}/models/{model_id}/variants/{model_variant_id}/binary"
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.get_model_binary_files.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=model_id, model_variant_id=model_variant_id
        )

    def test_download_directory_backed_model_preserves_paths(
        self, fxt_get_project, fxt_model, fxt_model_service, fxt_client, tmp_path
    ) -> None:
        import zipfile
        from io import BytesIO

        model_variant_id = uuid4()
        variant_dir = tmp_path / "variants" / str(model_variant_id)
        checkpoint_dir = variant_dir / "model"
        (checkpoint_dir / "weights").mkdir(parents=True)
        config_path = checkpoint_dir / "config.json"
        weights_path = checkpoint_dir / "weights" / "model.safetensors"
        config_path.write_text("{}")
        weights_path.write_bytes(b"weights")
        fxt_model_service.get_model_binary_files.return_value = True, (config_path, weights_path)
        fxt_model_service.get_variant.return_value = ModelVariant(
            id=model_variant_id,
            model_revision_id=fxt_model.id,
            format=ModelFormat.PYTORCH,
            precision=ModelPrecision.FP32,
        )

        response = fxt_client.get(
            f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}/variants/{model_variant_id}/binary"
        )

        assert response.status_code == status.HTTP_200_OK
        with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
            assert zip_file.namelist() == ["model/config.json", "model/weights/model.safetensors"]

    def test_download_model_binary_invalid_id(self, fxt_get_project, fxt_model_service, fxt_client):
        model_id = uuid4()
        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{model_id}/variants/invalid-id/binary")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_model_service.get_model.assert_not_called()

    def test_rename_model_success(self, fxt_model, fxt_get_project, fxt_model_service, fxt_client):
        fxt_model_service.rename_model.return_value = fxt_model
        fxt_model_service.get_model_size_in_bytes.return_value = 1024
        fxt_model_service.get_model_variants.return_value = []

        response = fxt_client.patch(
            f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}", json={"name": "New name"}
        )

        assert response.status_code == status.HTTP_200_OK
        fxt_model_service.rename_model.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=fxt_model.id, model_metadata={"name": "New name"}
        )
        fxt_model_service.get_model_size_in_bytes.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=fxt_model.id
        )
        fxt_model_service.get_model_variants.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=fxt_model.id
        )

    def test_rename_model_not_found(self, fxt_get_project, fxt_model_service, fxt_client):
        model_id = uuid4()
        fxt_model_service.rename_model.side_effect = ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        response = fxt_client.patch(f"/api/projects/{fxt_get_project.id}/models/{model_id}", json={"name": "New name"})

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.rename_model.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=model_id, model_metadata={"name": "New name"}
        )

    def test_get_training_metrics_success(self, fxt_model, fxt_get_project, fxt_model_service, fxt_client):
        mock_statistics = [
            {
                "header": "Training total loss",
                "type": "line",
                "key": "Training total loss",
                "value": {
                    "x_axis_label": "Timestamp",
                    "y_axis_label": "Training total loss",
                    "line_data": [
                        {
                            "header": "Training total loss",
                            "key": "Training total loss",
                            "points": [{"x": 1.0, "y": 1.0, "type": "point"}],
                        }
                    ],
                },
            }
        ]
        fxt_model_service.get_model_training_metrics.return_value = mock_statistics

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}/training_metrics")

        assert response.status_code == status.HTTP_200_OK
        fxt_model_service.get_model_training_metrics.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=fxt_model.id
        )

    def test_get_training_metrics_not_found(self, fxt_get_project, fxt_model_service, fxt_client):
        model_id = uuid4()
        fxt_model_service.get_model_training_metrics.side_effect = ResourceNotFoundError(
            ResourceType.MODEL, str(model_id)
        )

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{model_id}/training_metrics")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.get_model_training_metrics.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=model_id
        )

    @pytest.mark.parametrize(
        "accept_header, expected_content_type",
        [
            ("application/json", "application/x-ndjson"),
            ("application/x-ndjson", "application/x-ndjson"),
            ("text/plain", "text/plain; charset=utf-8"),
            (None, "application/x-ndjson"),  # Default to JSON if no Accept header is provided
        ],
    )
    def test_get_training_logs_success(
        self, accept_header, expected_content_type, fxt_get_project, fxt_model, fxt_model_service, fxt_client, tmp_path
    ):
        log_file = tmp_path / "training.log"
        log_content = "Training started\nEpoch 1/10\nLoss: 0.5\n"
        log_file.write_text(log_content, newline="\n")

        if accept_header == "text/plain":
            fxt_model_service.get_logs.return_value = log_content
        else:
            fxt_model_service.get_logs.return_value = log_file

        headers = {"Accept": accept_header} if accept_header is not None else {}
        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}/logs", headers=headers)

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == expected_content_type
        assert response.text == log_content
        fxt_model_service.get_logs.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=fxt_model.id, as_text="text/plain" in expected_content_type
        )

    def test_get_training_logs_not_found(self, fxt_get_project, fxt_model_service, fxt_client):
        model_id = uuid4()
        fxt_model_service.get_logs.side_effect = ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{model_id}/logs")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.get_logs.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=model_id, as_text=False
        )

    def test_get_training_logs_invalid_id(self, fxt_get_project, fxt_model_service, fxt_client):
        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/invalid-id/logs")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_model_service.get_logs.assert_not_called()

    def test_get_training_logs_not_available(self, fxt_get_project, fxt_model, fxt_model_service, fxt_client):
        fxt_model_service.get_logs.side_effect = ValueError(
            "Logs are not available for models that have not started or are currently in progress of training"
        )

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}/logs")

        assert response.status_code == status.HTTP_409_CONFLICT
        fxt_model_service.get_logs.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=fxt_model.id, as_text=False
        )

    def test_get_training_logs_file_not_exists(self, fxt_get_project, fxt_model, fxt_model_service, fxt_client):
        fxt_model_service.get_logs.return_value = None

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}/logs")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.json() == {"detail": "Log file not found"}
        fxt_model_service.get_logs.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=fxt_model.id, as_text=False
        )

    def test_get_model_training_configuration_success(
        self, fxt_model, fxt_get_project, fxt_model_service, fxt_training_configuration_service, fxt_client
    ):
        mock_training_config = MagicMock()
        fxt_training_configuration_service.get_by_model_revision.return_value = mock_training_config
        fxt_model_service.get_model_revision_architecture.return_value = "object-detection-yolox-x"

        mock_default_config = MagicMock()
        mock_view = MagicMock(spec=TrainingConfigurationView)
        mock_view.model_dump.return_value = {"parameters": []}

        with (
            patch.object(
                TrainingConfigurationService,
                "get_default_by_model_architecture",
                return_value=mock_default_config,
            ) as mock_get_default,
            patch.object(TrainingConfigurationView, "from_training_configuration", return_value=mock_view) as mock_from,
        ):
            response = fxt_client.get(
                f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}/training_configuration"
            )

            fxt_training_configuration_service.get_by_model_revision.assert_called_once_with(
                project_id=fxt_get_project.id, model_revision_id=fxt_model.id
            )
            fxt_model_service.get_model_revision_architecture.assert_called_once_with(
                project_id=fxt_get_project.id, model_id=fxt_model.id
            )
            mock_get_default.assert_called_once_with(model_architecture_id="object-detection-yolox-x")
            mock_from.assert_called_once_with(config=mock_training_config, default_config=mock_default_config)

        assert response.status_code == status.HTTP_200_OK

    def test_get_model_training_configuration_model_not_found(
        self, fxt_get_project, fxt_model_service, fxt_training_configuration_service, fxt_client
    ):
        model_id = uuid4()
        fxt_training_configuration_service.get_by_model_revision.side_effect = ResourceNotFoundError(
            ResourceType.MODEL, str(model_id)
        )

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{model_id}/training_configuration")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_training_configuration_service.get_by_model_revision.assert_called_once_with(
            project_id=fxt_get_project.id, model_revision_id=model_id
        )

    def test_get_model_training_configuration_invalid_id(
        self, fxt_get_project, fxt_training_configuration_service, fxt_client
    ):
        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/invalid-id/training_configuration")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_training_configuration_service.get_by_model_revision.assert_not_called()
