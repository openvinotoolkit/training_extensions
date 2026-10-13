# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
from collections.abc import Generator
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import yaml
from fastapi import status

from app.api.dependencies import get_sink, get_sink_service
from app.api.schemas.sink import FolderSinkConfigCreate, FolderSinkConfigView, MqttSinkConfigView
from app.models import OutputFormat, SinkType
from app.models.sink import FolderConfig, MqttConfig, Sink, SinkTestResult
from app.services import (
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
    ResourceValidationError,
    ResourceWithNameAlreadyExistsError,
    SinkService,
)


@pytest.fixture
def fxt_folder_sink_create() -> FolderSinkConfigCreate:
    return FolderSinkConfigCreate(
        id=uuid4(),
        sink_type=SinkType.FOLDER,
        name="Test Folder Sink",
        rate_limit=0.1,
        output_formats=[OutputFormat.PREDICTIONS],
        folder_path="/test/path",
    )


@pytest.fixture
def fxt_folder_sink_view() -> FolderSinkConfigView:
    return FolderSinkConfigView(
        id=uuid4(),
        sink_type=SinkType.FOLDER,
        name="Test Folder Sink",
        rate_limit=0.1,
        output_formats=[OutputFormat.PREDICTIONS],
        config_data=FolderConfig(folder_path="/test/path"),
    )


@pytest.fixture
def fxt_get_sink(fxt_app, fxt_folder_sink_view) -> Generator[Sink]:
    fxt_app.dependency_overrides[get_sink] = lambda: fxt_folder_sink_view
    yield fxt_folder_sink_view
    del fxt_app.dependency_overrides[get_sink]


@pytest.fixture
def fxt_mqtt_sink_view() -> MqttSinkConfigView:
    return MqttSinkConfigView(
        id=uuid4(),
        sink_type=SinkType.MQTT,
        name="Test MQTT Sink",
        rate_limit=0.2,
        output_formats=[OutputFormat.IMAGE_WITH_PREDICTIONS],
        config_data=MqttConfig(
            broker_host="localhost",
            broker_port=1883,
            topic="test_topic",
        ),
    )


@pytest.fixture
def fxt_sink_service(fxt_app) -> MagicMock:
    sink_service = MagicMock(spec=SinkService)
    fxt_app.dependency_overrides[get_sink_service] = lambda: sink_service
    return sink_service


class TestSinkEndpoints:
    def test_create_sink_success(self, fxt_folder_sink_create, fxt_sink_service, fxt_client):
        sink_id = str(fxt_folder_sink_create.id)
        fxt_sink_service.create_sink.return_value = fxt_folder_sink_create

        response = fxt_client.post("/api/sinks", json=fxt_folder_sink_create.model_dump(exclude={"id"}))

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["id"] == sink_id
        assert response.json()["name"] == fxt_folder_sink_create.name
        fxt_sink_service.create_sink.assert_called_once()

    def test_create_sink_disconnected_fails(self, fxt_sink_service, fxt_client):
        response = fxt_client.post("/api/sinks", json={"sink_type": SinkType.DISCONNECTED, "name": "Disconnected Sink"})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_sink_service.create_sink.assert_not_called()

    def test_create_sink_validation_error(self, fxt_sink_service, fxt_client):
        response = fxt_client.post("/api/sinks", json={"name": ""})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_sink_service.create_sink.assert_not_called()

    @pytest.mark.parametrize(
        ("field", "value"),
        [("timeout", False), ("timeout", 0), ("timeout", -1), ("rate_limit", 0)],
    )
    def test_create_webhook_sink_rejects_invalid_numeric_config(self, field, value, fxt_sink_service, fxt_client):
        payload = {
            "sink_type": "webhook",
            "name": "Webhook",
            "output_formats": ["predictions"],
            "webhook_url": "https://example.com/webhook",
            "http_method": "POST",
            "headers": None,
            "timeout": 10,
        }
        payload[field] = value

        response = fxt_client.post("/api/sinks", json=payload)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_sink_service.create_sink.assert_not_called()

    def test_create_sink_exists(self, fxt_folder_sink_create, fxt_sink_service, fxt_client):
        fxt_sink_service.create_sink.side_effect = ResourceWithNameAlreadyExistsError(
            resource_type=ResourceType.SINK, resource_name="New Config"
        )
        response = fxt_client.post("/api/sinks", json=fxt_folder_sink_create.model_dump(exclude={"id"}))

        assert response.status_code == status.HTTP_409_CONFLICT
        fxt_sink_service.create_sink.assert_called_once()

    def test_create_sink_not_reachable(self, fxt_folder_sink_create, fxt_sink_service, fxt_client):
        err = ResourceValidationError(
            ResourceType.SINK, str(fxt_folder_sink_create.id), "Directory not found: /test/path"
        )
        fxt_sink_service.create_sink.side_effect = err

        response = fxt_client.post("/api/sinks", json=fxt_folder_sink_create.model_dump(exclude={"id"}))

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert response.json()["detail"] == str(err)
        fxt_sink_service.create_sink.assert_called_once()

    def test_list_sinks(self, fxt_folder_sink_view, fxt_mqtt_sink, fxt_sink_service, fxt_client):
        fxt_sink_service.list_all.return_value = [fxt_folder_sink_view, fxt_mqtt_sink]

        response = fxt_client.get("/api/sinks")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 2
        fxt_sink_service.list_all.assert_called_once()

    def test_get_sink_success(self, fxt_folder_sink_view, fxt_sink_service, fxt_client):
        sink_id = str(fxt_folder_sink_view.id)
        fxt_sink_service.get_by_id.return_value = fxt_folder_sink_view

        response = fxt_client.get(f"/api/sinks/{sink_id}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["id"] == sink_id
        fxt_sink_service.get_by_id.assert_called_once_with(fxt_folder_sink_view.id)

    def test_get_sink_not_found(self, fxt_sink_service, fxt_client):
        sink_id = uuid4()
        fxt_sink_service.get_by_id.side_effect = ResourceNotFoundError(ResourceType.SINK, str(sink_id))

        response = fxt_client.get(f"/api/sinks/{str(sink_id)}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_sink_service.get_by_id.assert_called_once_with(sink_id)

    def test_get_sink_invalid_uuid(self, fxt_sink_service, fxt_client):
        response = fxt_client.get("/api/sinks/invalid-uuid")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_update_sink_success(self, fxt_folder_sink_create, fxt_get_sink, fxt_sink_service, fxt_client):
        update_data = {"folder_path": "/new/path"}
        sink_id = str(fxt_folder_sink_create.id)
        updated_config = fxt_folder_sink_create.model_copy(update=update_data)
        fxt_sink_service.update_sink.return_value = updated_config

        response = fxt_client.patch(f"/api/sinks/{sink_id}", json=update_data)

        assert response.status_code == status.HTTP_200_OK
        key, value = next(iter(update_data.items()))
        assert response.json()[key] == value
        fxt_sink_service.update_sink.assert_called_once_with(
            sink=fxt_get_sink,
            new_name="Test Folder Sink",
            new_rate_limit=0.1,
            new_config_data=FolderConfig(
                folder_path="/new/path",
            ),
            new_output_formats=[OutputFormat.PREDICTIONS],
        )

    def test_update_sink_type_forbidden(self, fxt_folder_sink_create, fxt_sink_service, fxt_client):
        sink_id = str(fxt_folder_sink_create.id)

        response = fxt_client.patch(f"/api/sinks/{sink_id}", json={"sink_type": "mqtt"})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "_type" in response.json()["detail"]
        fxt_sink_service.update_sink.assert_not_called()

    def test_update_sink_not_reachable(self, fxt_get_sink, fxt_sink_service, fxt_client):
        sink_id = str(fxt_get_sink.id)
        err = ResourceValidationError(ResourceType.SINK, sink_id, "Directory not found: /new/path")
        fxt_sink_service.update_sink.side_effect = err

        response = fxt_client.patch(f"/api/sinks/{sink_id}", json={"folder_path": "/new/path"})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert response.json()["detail"] == str(err)

    def test_delete_sink_success(self, fxt_folder_sink_view, fxt_get_sink, fxt_sink_service, fxt_client):
        sink_id = str(fxt_folder_sink_view.id)
        fxt_sink_service.delete_sink.side_effect = None

        response = fxt_client.delete(f"/api/sinks/{sink_id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_sink_service.delete_sink.assert_called_once_with(fxt_get_sink)

    def test_delete_sink_invalid_id(self, fxt_sink_service, fxt_client):
        fxt_sink_service.delete_sink.side_effect = None

        response = fxt_client.delete("/api/sinks/invalid-id")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_sink_service.delete_sink.assert_not_called()

    def test_delete_sink_not_found(self, fxt_sink_service, fxt_client):
        sink_id = str(uuid4())
        fxt_sink_service.get_by_id.side_effect = ResourceNotFoundError(ResourceType.SINK, sink_id)

        response = fxt_client.delete(f"/api/sinks/{sink_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_sink_in_use(self, fxt_folder_sink_view, fxt_sink_service, fxt_client):
        sink_id = str(fxt_folder_sink_view.id)
        err = ResourceInUseError(ResourceType.SINK, sink_id)
        fxt_sink_service.delete_sink.side_effect = err

        response = fxt_client.delete(f"/api/sinks/{sink_id}")

        assert response.status_code == status.HTTP_409_CONFLICT
        assert str(err) == response.json()["detail"]

    def test_export_sink_success(self, fxt_folder_sink_view, fxt_sink_service, fxt_client):
        sink_id = str(fxt_folder_sink_view.id)
        fxt_sink_service.get_by_id.return_value = fxt_folder_sink_view

        response = fxt_client.post(f"/api/sinks/{sink_id}:export")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/x-yaml"
        assert f"sink_{sink_id}.yaml" in response.headers["content-disposition"]
        assert (
            response.text == "folder_path: /test/path\nname: Test Folder Sink\noutput_formats:\n"
            "- predictions\nrate_limit: 0.1\nsink_type: folder\n"
        )
        fxt_sink_service.get_by_id.assert_called_once_with(fxt_folder_sink_view.id)

    def test_import_sink_success(self, fxt_folder_sink_create, fxt_sink_service, fxt_client):
        sink_data = fxt_folder_sink_create.model_dump(exclude={"id"}, mode="json")
        yaml_content = yaml.safe_dump(sink_data)
        fxt_sink_service.create_sink.return_value = fxt_folder_sink_create

        files = {"yaml_file": ("test.yaml", io.BytesIO(yaml_content.encode()), "application/x-yaml")}
        response = fxt_client.post("/api/sinks:import", files=files)

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["id"] == str(fxt_folder_sink_create.id)
        fxt_sink_service.create_sink.assert_called_once()

    def test_import_sink_exists(self, fxt_folder_sink_create, fxt_sink_service, fxt_client):
        sink_data = fxt_folder_sink_create.model_dump(exclude={"id"}, mode="json")
        yaml_content = yaml.safe_dump(sink_data)
        fxt_sink_service.create_sink.side_effect = ResourceWithNameAlreadyExistsError(
            resource_type=ResourceType.SINK, resource_name="New Config"
        )

        files = {"yaml_file": ("test.yaml", io.BytesIO(yaml_content.encode()), "application/x-yaml")}
        response = fxt_client.post("/api/sinks:import", files=files)

        assert response.status_code == status.HTTP_409_CONFLICT
        fxt_sink_service.create_sink.assert_called_once()

    def test_import_sink_not_reachable(self, fxt_folder_sink_create, fxt_sink_service, fxt_client):
        sink_data = fxt_folder_sink_create.model_dump(exclude={"id"}, mode="json")
        yaml_content = yaml.safe_dump(sink_data)
        err = ResourceValidationError(
            ResourceType.SINK, str(fxt_folder_sink_create.id), "Directory not found: /test/path"
        )
        fxt_sink_service.create_sink.side_effect = err

        files = {"yaml_file": ("test.yaml", io.BytesIO(yaml_content.encode()), "application/x-yaml")}
        response = fxt_client.post("/api/sinks:import", files=files)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert response.json()["detail"] == str(err)
        fxt_sink_service.create_sink.assert_called_once()

    def test_import_sink_invalid_yaml(self, fxt_client):
        invalid_yaml = "invalid: yaml: content: ["
        files = {"yaml_file": ("test.yaml", io.BytesIO(invalid_yaml.encode()), "application/x-yaml")}

        response = fxt_client.post("/api/sinks:import", files=files)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid YAML format" in response.json()["detail"]

    def test_import_disconnected_sink_fails(self, fxt_sink_service, fxt_client):
        config_data = {"sink_type": "disconnected", "name": "Test"}
        yaml_content = yaml.safe_dump(config_data)
        files = {"yaml_file": ("test.yaml", io.BytesIO(yaml_content.encode()), "application/x-yaml")}

        response = fxt_client.post("/api/sinks:import", files=files)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_sink_service.create_sink.assert_not_called()

    def test_test_sink_reachable(self, fxt_folder_sink_view, fxt_get_sink, fxt_sink_service, fxt_client):
        sink_id = str(fxt_folder_sink_view.id)
        fxt_sink_service.test_sink.return_value = SinkTestResult.success(5.2)

        response = fxt_client.post(f"/api/sinks/{sink_id}:test")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["reachable"] is True
        assert response.json()["latency_ms"] == 5.2
        assert response.json()["error"] is None
        fxt_sink_service.test_sink.assert_called_once_with(fxt_get_sink)

    def test_test_sink_not_reachable(self, fxt_folder_sink_view, fxt_get_sink, fxt_sink_service, fxt_client):
        sink_id = str(fxt_folder_sink_view.id)
        fxt_sink_service.test_sink.return_value = SinkTestResult.failure("Directory not found: /test/path")

        response = fxt_client.post(f"/api/sinks/{sink_id}:test")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["reachable"] is False
        assert response.json()["error"] == "Directory not found: /test/path"
        assert response.json()["latency_ms"] is None
        fxt_sink_service.test_sink.assert_called_once_with(fxt_get_sink)

    def test_test_sink_not_found(self, fxt_sink_service, fxt_client):
        sink_id = str(uuid4())
        fxt_sink_service.get_by_id.side_effect = ResourceNotFoundError(ResourceType.SINK, sink_id)

        response = fxt_client.post(f"/api/sinks/{sink_id}:test")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_test_sink_invalid_id(self, fxt_sink_service, fxt_client):
        response = fxt_client.post("/api/sinks/invalid-id:test")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
