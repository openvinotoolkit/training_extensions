# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.params import Depends
from starlette.responses import StreamingResponse

from app.api.dependencies import get_staged_dataset_service
from app.api.io_utils import file_iterator
from app.api.schemas import StagedDatasetView
from app.api.validators import StagedDatasetID
from app.services import StagedDatasetService

router = APIRouter(prefix="/api/staged_datasets", tags=["Dataset Import/Export"])


@router.post(
    "",
    response_model=StagedDatasetView,
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_201_CREATED: {"description": "Dataset archive uploaded successfully"}},
)
async def upload_archive(
    file: Annotated[UploadFile, File()],
    staged_datasets_service: Annotated[StagedDatasetService, Depends(get_staged_dataset_service)],
) -> StagedDatasetView:
    """Upload dataset archive to the staging area"""
    try:
        if not file.filename or not file.filename.endswith(".zip"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Only zip files are allowed.",
            )
        staged_dataset = await staged_datasets_service.upload(filename="dataset.zip", file_obj=file.file)
        return StagedDatasetView.model_validate(staged_dataset, from_attributes=True)
    finally:
        await file.close()


@router.get(
    "",
    response_model=list[StagedDatasetView],
    responses={
        status.HTTP_200_OK: {
            "description": "List of staged dataset archives",
        },
    },
)
async def list_datasets(
    staged_dataset_service: Annotated[StagedDatasetService, Depends(get_staged_dataset_service)],
) -> list[StagedDatasetView]:
    """List all datasets from the staging area"""
    return [StagedDatasetView.model_validate(item, from_attributes=True) for item in staged_dataset_service.list_all()]


@router.get(
    "/{staged_dataset_id}",
    response_model=StagedDatasetView,
    responses={
        status.HTTP_200_OK: {"description": "Staged dataset found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid staged dataset ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Staged dataset not found"},
    },
)
def get_dataset(
    staged_dataset_id: StagedDatasetID,
    staged_dataset_service: Annotated[StagedDatasetService, Depends(get_staged_dataset_service)],
) -> StagedDatasetView:
    """Get info about the staged dataset from the staging area"""
    staged_dataset = staged_dataset_service.find_by_id(staged_dataset_id)
    if not staged_dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Staged dataset with ID '{staged_dataset_id}' not found.",
        )
    return StagedDatasetView.model_validate(staged_dataset, from_attributes=True)


@router.get(
    "/{staged_dataset_id}/zip",
    responses={
        status.HTTP_200_OK: {"description": "Staged dataset found"},
        status.HTTP_404_NOT_FOUND: {"description": "Staged dataset not found"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Staged dataset is not in zip format ready for download"
        },
    },
)
def download_archive(
    staged_dataset_id: StagedDatasetID,
    staged_dataset_service: Annotated[StagedDatasetService, Depends(get_staged_dataset_service)],
) -> StreamingResponse:
    """Download the staged dataset archive from the staging area"""
    staged_dataset = staged_dataset_service.find_by_id(staged_dataset_id)
    if staged_dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Staged dataset with ID `{staged_dataset_id}` not found.",
        )

    if not staged_dataset.compressed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Staged dataset is not in zip format ready for download.",
        )

    file_path = Path(staged_dataset.filename)
    return StreamingResponse(
        file_iterator(file_path),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename*=utf-8''{quote(file_path.name)}",
        },
    )


@router.delete(
    "/{staged_dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Staged dataset successfully deleted"},
        status.HTTP_404_NOT_FOUND: {"description": "Staged dataset not found"},
    },
)
def delete_dataset(
    staged_dataset_id: StagedDatasetID,
    staged_dataset_service: Annotated[StagedDatasetService, Depends(get_staged_dataset_service)],
) -> None:
    """Delete the staged dataset from the staging area"""
    deleted = staged_dataset_service.delete_by_id(staged_dataset_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Staged dataset with ID '{staged_dataset_id}' not found.",
        )
