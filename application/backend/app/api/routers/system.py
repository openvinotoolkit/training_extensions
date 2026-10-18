# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""System API Endpoints"""

import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from starlette.background import BackgroundTask
from starlette.responses import FileResponse

from app.api.dependencies import get_license_service, get_log_dir, get_system_service
from app.api.schemas.system import CameraInfoView, DeviceInfoView, PlatformType, SystemInfoView
from app.services import SystemService
from app.services.license_service import LicenseService

router = APIRouter(prefix="/api/system", tags=["System"])


def _get_platform() -> PlatformType:
    """Return the current operating system platform."""
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


@router.get("/info")
async def get_system_info(
    license_service: Annotated[LicenseService, Depends(get_license_service)],
) -> SystemInfoView:
    """Returns system information including license status and platform."""
    return SystemInfoView(license_accepted=license_service.is_accepted(), platform=_get_platform())


@router.get("/devices/inference")
async def get_inference_devices(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> list[DeviceInfoView]:
    """Returns the list of available compute devices (CPU, Intel XPU)."""
    inference_devices = system_service.inference_devices()
    return [DeviceInfoView.model_validate(device, from_attributes=True) for device in inference_devices]


@router.get("/devices/training")
async def get_training_devices(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> list[DeviceInfoView]:
    """Returns the list of available training devices (CPU, Intel XPU, NVIDIA CUDA)."""
    training_devices = system_service.training_devices()
    return [DeviceInfoView.model_validate(device, from_attributes=True) for device in training_devices]


@router.get("/devices/camera")
async def get_camera_devices(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> list[CameraInfoView]:
    """Returns the list of available camera devices."""
    camera_devices = system_service.list_cameras()
    return [CameraInfoView.model_validate(device, from_attributes=True) for device in camera_devices]


@router.get("/metrics/memory")
async def get_memory(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> dict:
    """Returns the used memory in MB and total available memory in MB."""
    used, total = system_service.get_memory_usage()
    return {"used": int(used), "total": int(total)}


@router.get("/logs", response_class=FileResponse)
def download_system_logs(log_dir: Annotated[Path, Depends(get_log_dir)]) -> FileResponse:
    """Downloads all system logs as a zip archive."""
    fd, temp_name = tempfile.mkstemp(suffix=".zip")
    os.close(fd)

    try:
        with zipfile.ZipFile(temp_name, "w", zipfile.ZIP_DEFLATED) as zf:
            if log_dir.exists():
                for root, _, files in os.walk(log_dir):
                    for file in files:
                        file_path = Path(root) / file
                        # store path relative to log_dir
                        arcname = file_path.relative_to(log_dir)
                        zf.write(file_path, arcname)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise

    return FileResponse(
        path=temp_name,
        filename="geti_logs.zip",
        media_type="application/zip",
        background=BackgroundTask(os.unlink, temp_name),
    )
