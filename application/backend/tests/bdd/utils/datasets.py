# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path
from uuid import UUID

import requests
from requests import Session

from app.api.schemas import StagedDatasetView

_CHUNK_SIZE = 1 << 20  # 1 MiB


def download_file(url: str, dest_dir: Path) -> Path:
    """Download *url* into *dest_dir* and return the local file path (cached across scenarios)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / url.rsplit("/", 1)[-1]
    if not dest.exists():
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with dest.open("wb") as fh:
                shutil.copyfileobj(response.raw, fh, _CHUNK_SIZE)
    return dest


def upload_staged_dataset(session: Session, base_url: str, archive_path: Path) -> UUID:
    """Upload a local dataset archive to the staging area and return its staged dataset ID."""
    with archive_path.open("rb") as fh:
        response = session.post(
            f"{base_url}/api/staged_datasets",
            files={"file": (archive_path.name, fh, "application/zip")},
        )
    assert response.status_code == 201, (
        f"Expected staged dataset upload to succeed, got {response.status_code}, response: {response.text}"
    )
    return StagedDatasetView.model_validate(response.json()).id
