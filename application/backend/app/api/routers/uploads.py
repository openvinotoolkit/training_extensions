# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Resumable uploads following the TUS 1.0.0 protocol (https://tus.io/protocols/resumable-upload)."""

from collections.abc import AsyncIterator
from email.utils import format_datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from loguru import logger
from starlette.requests import ClientDisconnect

from app.api.dependencies import get_upload_service
from app.api.schemas import UploadView
from app.api.upload_utils import HTTP_460_CHECKSUM_MISMATCH, upload_error_to_http
from app.models import Upload
from app.services.upload_service import TUS_CHECKSUM_ALGORITHMS, TUS_EXTENSIONS, TUS_VERSION, UploadError, UploadService

OFFSET_OCTET_STREAM = "application/offset+octet-stream"


def _reject_method_override(request: Request) -> None:
    """Refuse method overrides: they would let clients reach PATCH/DELETE through POST and bypass method controls."""
    if "X-HTTP-Method-Override" in request.headers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="The X-HTTP-Method-Override header is not supported."
        )


router = APIRouter(prefix="/api/uploads", tags=["Uploads"], dependencies=[Depends(_reject_method_override)])


def _require_tus_version(request: Request) -> None:
    if request.headers.get("Tus-Resumable") != TUS_VERSION:
        raise HTTPException(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            detail=f"The Tus-Resumable header must be '{TUS_VERSION}'.",
            headers={"Tus-Version": TUS_VERSION},
        )


def _parse_non_negative_int(request: Request, header: str) -> int | None:
    value = request.headers.get(header)
    if value is None:
        return None
    if not value.isascii() or not value.isdecimal():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"The {header} header must be a non-negative integer."
        )
    return int(value)


def _media_type(request: Request) -> str:
    return request.headers.get("Content-Type", "").split(";")[0].strip().lower()


def _has_body(request: Request) -> bool:
    return request.headers.get("Content-Length", "0") != "0" or "Transfer-Encoding" in request.headers


async def _request_body(request: Request) -> AsyncIterator[bytes]:
    """Stream the request body; a client disconnection simply ends it, so the bytes received so far are kept."""
    try:
        async for chunk in request.stream():
            yield chunk
    except ClientDisconnect:
        logger.info("Client disconnected while uploading; keeping the bytes received so far")


def _upload_headers(upload: Upload) -> dict[str, str]:
    headers = {
        "Tus-Resumable": TUS_VERSION,
        "Upload-Offset": str(upload.offset),
        "Upload-Expires": format_datetime(upload.expires_at, usegmt=True),
        "Cache-Control": "no-store",
    }
    if upload.size is None:
        headers["Upload-Defer-Length"] = "1"
    else:
        headers["Upload-Length"] = str(upload.size)
    return headers


def _tus_error(error: UploadError) -> HTTPException:
    return upload_error_to_http(error, headers={"Tus-Resumable": TUS_VERSION})


# ---- OpenAPI documentation of the header-based TUS semantics, which FastAPI cannot infer -----------------------

_STRING = {"type": "string"}
_INTEGER = {"type": "integer", "minimum": 0}


def _header(name: str, description: str, schema: dict[str, Any], required: bool = False) -> dict[str, Any]:
    return {"name": name, "in": "header", "required": required, "description": description, "schema": schema}


_TUS_RESUMABLE = _header(
    "Tus-Resumable", "Version of the TUS protocol used by the client", {"type": "string", "enum": [TUS_VERSION]}, True
)
_UPLOAD_CHECKSUM = _header(
    "Upload-Checksum",
    f"Checksum of the request body: '<algorithm> <base64 digest>', algorithm in {', '.join(TUS_CHECKSUM_ALGORITHMS)}",
    _STRING,
)
_RESPONSE_HEADERS: dict[str, dict[str, Any]] = {
    "Tus-Resumable": {"description": "Version of the TUS protocol used by the server", "schema": _STRING},
    "Tus-Version": {"description": "Comma-separated list of TUS versions supported by the server", "schema": _STRING},
    "Tus-Extension": {"description": "Comma-separated list of supported TUS extensions", "schema": _STRING},
    "Tus-Max-Size": {"description": "Maximum allowed size of an upload in bytes", "schema": _INTEGER},
    "Tus-Checksum-Algorithm": {"description": "Comma-separated list of checksum algorithms", "schema": _STRING},
    "Location": {"description": "URL of the created upload", "schema": _STRING},
    "Upload-Offset": {"description": "Number of bytes received and acknowledged by the server", "schema": _INTEGER},
    "Upload-Length": {"description": "Total size of the upload in bytes", "schema": _INTEGER},
    "Upload-Defer-Length": {"description": "'1' if the size of the upload is not known yet", "schema": _STRING},
    "Upload-Expires": {"description": "RFC 9110 date after which the upload expires", "schema": _STRING},
    "Cache-Control": {"description": "Always 'no-store'", "schema": _STRING},
}


def _response_headers(*names: str) -> dict[str, dict[str, Any]]:
    return {name: _RESPONSE_HEADERS[name] for name in names}


_UPLOAD_STATE_HEADERS = _response_headers(
    "Tus-Resumable", "Upload-Offset", "Upload-Length", "Upload-Defer-Length", "Upload-Expires", "Cache-Control"
)
_BINARY_BODY = {OFFSET_OCTET_STREAM: {"schema": {"type": "string", "format": "binary"}}}
_ERRORS: dict[str, dict[str, Any]] = {
    "400": {"description": "Malformed TUS headers or metadata"},
    "404": {"description": "Upload not found"},
    "410": {"description": "Upload has expired or has already been consumed"},
    "412": {"description": "Unsupported TUS protocol version", "headers": _response_headers("Tus-Version")},
    "423": {"description": "Another request is currently operating on the upload"},
}


def _errors(*codes: str) -> dict[str, dict[str, Any]]:
    return {code: _ERRORS[code] for code in codes}


# ---- TUS protocol -------------------------------------------------------------------------------------------------


def _capabilities_response(max_size: int) -> Response:
    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
        headers={
            "Tus-Resumable": TUS_VERSION,
            "Tus-Version": TUS_VERSION,
            "Tus-Extension": ",".join(TUS_EXTENSIONS),
            "Tus-Max-Size": str(max_size),
            "Tus-Checksum-Algorithm": ",".join(TUS_CHECKSUM_ALGORITHMS),
        },
    )


_CAPABILITIES_OPENAPI = {
    "responses": {
        "204": {
            "description": "TUS capabilities of the server",
            "headers": _response_headers(
                "Tus-Resumable", "Tus-Version", "Tus-Extension", "Tus-Max-Size", "Tus-Checksum-Algorithm"
            ),
        }
    }
}


@router.options(
    "", status_code=status.HTTP_204_NO_CONTENT, response_class=Response, openapi_extra=_CAPABILITIES_OPENAPI
)
async def get_upload_capabilities(
    upload_service: Annotated[UploadService, Depends(get_upload_service)],
) -> Response:
    """Advertise the supported TUS version, extensions and maximum upload size."""
    return _capabilities_response(upload_service.max_size)


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_class=Response,
    openapi_extra={
        "parameters": [
            _TUS_RESUMABLE,
            _header("Upload-Length", "Total size of the upload in bytes", _INTEGER),
            _header("Upload-Defer-Length", "'1' to declare the size later, in a PATCH request", _STRING),
            _header(
                "Upload-Metadata",
                "Comma-separated 'key base64value' pairs; 'filename' and 'filetype' are recognised",
                _STRING,
            ),
            _UPLOAD_CHECKSUM,
        ],
        "requestBody": {"required": False, "content": _BINARY_BODY},
        "responses": {
            "201": {
                "description": "Upload created; the body, if any, was stored as the first chunk",
                "headers": {**_response_headers("Location"), **_UPLOAD_STATE_HEADERS},
            },
            **_errors("400", "412"),
            "413": {"description": "Upload-Length exceeds Tus-Max-Size"},
            "415": {"description": "Unsupported Content-Type for the first chunk"},
            str(HTTP_460_CHECKSUM_MISMATCH): {"description": "Checksum mismatch"},
        },
    },
)
async def create_upload(
    request: Request,
    upload_service: Annotated[UploadService, Depends(get_upload_service)],
) -> Response:
    """Create a resumable upload (TUS 'creation' and 'creation-with-upload' extensions)."""
    _require_tus_version(request)
    size = _parse_non_negative_int(request, "Upload-Length")
    defer_length = request.headers.get("Upload-Defer-Length")
    if (size is None) == (defer_length is None) or (defer_length is not None and defer_length != "1"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Exactly one of Upload-Length or 'Upload-Defer-Length: 1' must be provided.",
        )
    body = None
    if _media_type(request) == OFFSET_OCTET_STREAM:
        body = _request_body(request)
    elif _has_body(request):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"The first chunk must be sent with Content-Type: {OFFSET_OCTET_STREAM}.",
        )
    try:
        upload = await upload_service.create(
            size=size,
            metadata_header=request.headers.get("Upload-Metadata"),
            body=body,
            checksum_header=request.headers.get("Upload-Checksum"),
        )
    except UploadError as error:
        raise _tus_error(error) from error
    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"{router.prefix}/{upload.id}", **_upload_headers(upload)},
    )


@router.head(
    "/{upload_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    openapi_extra={
        "parameters": [_TUS_RESUMABLE],
        "responses": {
            "204": {"description": "Offset to resume the upload from", "headers": _UPLOAD_STATE_HEADERS},
            **_errors("404", "410", "412"),
        },
    },
)
async def get_upload_offset(
    upload_id: UUID,
    request: Request,
    upload_service: Annotated[UploadService, Depends(get_upload_service)],
) -> Response:
    """Get the offset from which the upload can be resumed."""
    _require_tus_version(request)
    try:
        upload = await upload_service.get_status(upload_id)
    except UploadError as error:
        raise _tus_error(error) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT, headers=_upload_headers(upload))


@router.patch(
    "/{upload_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    openapi_extra={
        "parameters": [
            _TUS_RESUMABLE,
            _header("Upload-Offset", "Offset at which the body must be written", _INTEGER, True),
            _header("Upload-Length", "Total size of the upload, if it was deferred at creation", _INTEGER),
            _UPLOAD_CHECKSUM,
        ],
        "requestBody": {"required": True, "content": _BINARY_BODY},
        "responses": {
            "204": {"description": "Chunk stored", "headers": _UPLOAD_STATE_HEADERS},
            **_errors("400", "404", "410", "412", "423"),
            "409": {"description": "Upload-Offset does not match the offset of the upload"},
            "413": {"description": "The body goes past Upload-Length or Tus-Max-Size"},
            "415": {"description": f"Content-Type is not {OFFSET_OCTET_STREAM}"},
            str(HTTP_460_CHECKSUM_MISMATCH): {"description": "Checksum mismatch; the chunk was discarded"},
        },
    },
)
async def append_to_upload(
    upload_id: UUID,
    request: Request,
    upload_service: Annotated[UploadService, Depends(get_upload_service)],
) -> Response:
    """Write a chunk of data at the current offset of the upload."""
    _require_tus_version(request)
    if _media_type(request) != OFFSET_OCTET_STREAM:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Content-Type must be {OFFSET_OCTET_STREAM}.",
        )
    offset = _parse_non_negative_int(request, "Upload-Offset")
    if offset is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The Upload-Offset header is required.")
    try:
        upload = await upload_service.append(
            upload_id=upload_id,
            offset=offset,
            chunks=_request_body(request),
            checksum_header=request.headers.get("Upload-Checksum"),
            length=_parse_non_negative_int(request, "Upload-Length"),
        )
    except UploadError as error:
        raise _tus_error(error) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT, headers=_upload_headers(upload))


@router.delete(
    "/{upload_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    openapi_extra={
        "parameters": [_TUS_RESUMABLE],
        "responses": {
            "204": {"description": "Upload deleted", "headers": _response_headers("Tus-Resumable")},
            **_errors("404", "412", "423"),
        },
    },
)
async def delete_upload(
    upload_id: UUID,
    request: Request,
    upload_service: Annotated[UploadService, Depends(get_upload_service)],
) -> Response:
    """Cancel an upload and discard the bytes received so far (TUS 'termination' extension)."""
    _require_tus_version(request)
    try:
        await upload_service.delete(upload_id)
    except UploadError as error:
        raise _tus_error(error) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT, headers={"Tus-Resumable": TUS_VERSION})


# ---- Non-TUS JSON views -------------------------------------------------------------------------------------------


@router.get("", response_model=list[UploadView])
async def list_uploads(
    upload_service: Annotated[UploadService, Depends(get_upload_service)],
) -> list[UploadView]:
    """List the resumable uploads, most recent first."""
    return [UploadView.model_validate(upload) for upload in await upload_service.list_all()]


@router.get(
    "/{upload_id}",
    response_model=UploadView,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Upload not found"}},
)
async def get_upload(
    upload_id: UUID,
    upload_service: Annotated[UploadService, Depends(get_upload_service)],
) -> UploadView:
    """Get the state of a resumable upload."""
    try:
        upload = await upload_service.get(upload_id)
    except UploadError as error:
        raise upload_error_to_http(error) from error
    return UploadView.model_validate(upload)
