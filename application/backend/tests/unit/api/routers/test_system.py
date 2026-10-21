# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.api.dependencies import get_system_service
from app.models.system import DeviceInfo, DeviceType
from app.services import SystemService


@pytest.fixture
def fxt_system_service(fxt_app) -> Mock:
    system_service = Mock(spec=SystemService)
    fxt_app.dependency_overrides[get_system_service] = lambda: system_service
    return system_service


class TestSystemEndpoints:
    def test_get_training_devices_with_all_devices(self, fxt_system_service: Mock, fxt_client: TestClient):
        """Test GET /api/system/devices/training with all device types"""
        fxt_system_service.training_devices.return_value = [
            DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None),
            DeviceInfo(type=DeviceType.XPU, name="Intel(R) Graphics [0x7d41]", memory=36022263808, index=0),
            DeviceInfo(type=DeviceType.CUDA, name="NVIDIA GeForce RTX 4090", memory=25769803776, index=0),
        ]

        response = fxt_client.get("/api/system/devices/training")

        assert response.status_code == status.HTTP_200_OK
        devices = response.json()
        assert len(devices) == 3
        assert devices[0]["type"] == "cpu"
        assert devices[1]["type"] == "xpu"
        assert devices[1]["name"] == "Intel(R) Graphics [0x7d41]"
        assert devices[1]["memory"] == 36022263808
        assert devices[1]["index"] == 0
        assert devices[2]["type"] == "cuda"
        assert devices[2]["name"] == "NVIDIA GeForce RTX 4090"
        assert devices[2]["memory"] == 25769803776
        assert devices[2]["index"] == 0

    def test_get_inference_devices_with_xpu(self, fxt_system_service: Mock, fxt_client: TestClient):
        """Test GET /api/system/devices/inference with Intel XPU"""
        fxt_system_service.inference_devices.return_value = [
            DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None),
            DeviceInfo(type=DeviceType.XPU, name="Intel(R) Graphics [0x7d41]", memory=36022263808, index=0),
        ]

        response = fxt_client.get("/api/system/devices/inference")

        assert response.status_code == status.HTTP_200_OK
        devices = response.json()
        assert len(devices) == 2
        assert devices[0]["type"] == "cpu"
        assert devices[1]["type"] == "xpu"
        assert devices[1]["name"] == "Intel(R) Graphics [0x7d41]"
        assert devices[1]["memory"] == 36022263808
        assert devices[1]["index"] == 0

    def test_get_camera_devices(self, fxt_system_service: Mock, fxt_client: TestClient):
        """Test GET /api/system/devices/camera"""
        fxt_system_service.list_cameras.return_value = [
            {"index": 0, "name": "Integrated USB Camera"},
            {"index": 1, "name": "USB Camera"},
        ]

        response = fxt_client.get("/api/system/devices/camera")

        assert response.status_code == status.HTTP_200_OK
        cameras = response.json()
        assert len(cameras) == 2
        assert cameras[0]["index"] == 0
        assert cameras[0]["name"] == "Integrated USB Camera"
        assert cameras[1]["index"] == 1
        assert cameras[1]["name"] == "USB Camera"

    def test_get_memory(self, fxt_system_service: Mock, fxt_client: TestClient):
        """Test GET /api/system/metrics/memory"""
        fxt_system_service.get_memory_usage.return_value = (1024.5, 8192.0)

        response = fxt_client.get("/api/system/metrics/memory")

        assert response.status_code == status.HTTP_200_OK
        memory = response.json()
        assert memory["used"] == 1024
        assert memory["total"] == 8192

    def test_download_system_logs(self, fxt_app, fxt_client: TestClient, tmp_path):
        """Test GET /api/system/logs"""
        # We need to mock get_log_dir to return our tmp_path
        from app.api.dependencies import get_log_dir

        fxt_app.dependency_overrides[get_log_dir] = lambda: tmp_path

        # Create a dummy log file
        log_file = tmp_path / "test.log"
        log_file.write_text("dummy log content")

        response = fxt_client.get("/api/system/logs")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/zip"
        assert response.headers["content-disposition"] == 'attachment; filename="geti_logs.zip"'

        import io
        import zipfile

        zf = zipfile.ZipFile(io.BytesIO(response.content))
        assert "test.log" in zf.namelist()
        assert zf.read("test.log") == b"dummy log content"
