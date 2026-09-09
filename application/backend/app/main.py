# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# export no_proxy="localhost, 127.0.0.1, ::1"
# Start with:
#  - uv run app/main.py
# or use docker
#  - docker compose up

import sys

if getattr(sys, "frozen", False) and __name__ == "__main__":
    import multiprocessing

    # Pyinstaller requires this method to be called in "frozen" applications if multiprocessing module is
    # used to prevent issues. This line must be called before any attempt to use multiprocessing module, so it makes
    # sense to put it in the very beginning.
    # https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html#multi-processing
    multiprocessing.freeze_support()

import asyncio
import logging
import ssl
from collections.abc import Awaitable, Callable
from os import getenv
from pathlib import Path
from typing import cast

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from hypercorn.asyncio import serve
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework
from loguru import logger

from app.api.cache_utils import CachedStaticFiles
from app.api.routers import (
    dataset_ie,
    dataset_revisions,
    dataset_views,
    datasets,
    jobs,
    media,
    model_architectures,
    models,
    pipelines,
    projects,
    sinks,
    source_media,
    sources,
    system,
    training_configurations,
    webrtc,
)
from app.api.routers import license as license_api
from app.core.certs import ensure_certs_exist
from app.core.logging import InterceptHandler, setup_hypercorn_logging
from app.lifecycle import lifespan
from app.services.base import ResourceNotFoundError, ResourceWithNameAlreadyExistsError
from app.settings import get_settings

settings = get_settings()
logging.basicConfig(handlers=[InterceptHandler()], level=settings.log_level, force=True)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=settings.description,
        openapi_url=settings.openapi_url,
        redoc_url=None,
        docs_url=None,
        lifespan=lifespan,
        # TODO add contact info
        # TODO add license
    )

    app.add_middleware(  # TODO restrict settings in production
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include all API routers from the routers package
    app.include_router(dataset_ie.router)
    app.include_router(dataset_revisions.router)
    app.include_router(dataset_views.router)
    app.include_router(datasets.router)
    app.include_router(jobs.router)
    app.include_router(license_api.router)
    app.include_router(media.router)
    app.include_router(model_architectures.router)
    app.include_router(models.router)
    app.include_router(pipelines.router)
    app.include_router(projects.router)
    app.include_router(sinks.router)
    app.include_router(source_media.router)
    app.include_router(sources.router)
    app.include_router(system.router)
    app.include_router(training_configurations.router)
    app.include_router(webrtc.router)

    cur_dir = Path(__file__).parent

    @app.get("/api/docs", include_in_schema=False)
    async def get_scalar_docs() -> FileResponse:
        """Shows docs for our OpenAPI specification using scalar"""
        return FileResponse(cur_dir / "static" / "scalar.html")

    @app.get("/stream", include_in_schema=False)
    async def get_webrtc_stream() -> FileResponse:
        """Get webrtc player"""
        return FileResponse(cur_dir / "static" / "webrtc.html")

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint"""
        return {"status": "ok"}

    @app.middleware("http")
    async def security_headers_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Add COEP and COOP security headers to all HTTP responses."""
        response = await call_next(request)
        response.headers.setdefault("Cross-Origin-Embedder-Policy", "credentialless")
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        return response

    @app.exception_handler(ResourceNotFoundError)
    async def resource_not_found_exception_handler(request: Request, exc: ResourceNotFoundError) -> JSONResponse:  # noqa: ARG001
        """Catch resource not found errors and return 404 response"""
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )

    @app.exception_handler(ResourceWithNameAlreadyExistsError)
    async def resource_with_name_already_exists_exception_handler(
        request: Request,  # noqa: ARG001
        exc: ResourceWithNameAlreadyExistsError,
    ) -> JSONResponse:
        """Catch resource-name-conflict errors and return 409 response"""
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(exc)},
        )

    @app.exception_handler(NotImplementedError)
    async def not_implemented_exception_handler(request: Request, exc: NotImplementedError) -> JSONResponse:  # noqa: ARG001
        """Catch not-implemented errors (features under construction) and return a 501 response"""
        return JSONResponse(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            content={"detail": str(exc) or "This feature is not implemented yet."},
        )

    static_dir = settings.static_files_dir
    if static_dir is not None and static_dir.is_dir() and any(static_dir.iterdir()):
        asset_prefix = getenv("ASSET_PREFIX", "/html")
        logger.info("Serving static files from {} by context {}", static_dir, asset_prefix)

        app.mount(asset_prefix, CachedStaticFiles(directory=static_dir), name="static")

        @app.get("/", include_in_schema=False)
        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa() -> FileResponse:
            """Serve the Single Page Application (SPA) index.html file for any path."""
            return FileResponse(cast(Path, static_dir) / "index.html")

    return app


# Substrings identifying benign TLS connection-teardown errors raised when a client
# (e.g. the WebView2-based desktop UI) closes an HTTP/2-over-TLS connection abruptly.
_BENIGN_TLS_SHUTDOWN_MARKERS = (
    "SSL shutdown timed out",
    "APPLICATION_DATA_AFTER_CLOSE_NOTIFY",
    "application data after close notify",
)


def _is_benign_tls_shutdown_error(exc: BaseException | None, message: str) -> bool:
    """Return True for harmless TLS teardown errors that should not be logged as errors."""
    if not isinstance(exc, TimeoutError | ssl.SSLError):
        return False
    text = f"{message} {exc}"
    return any(marker in text for marker in _BENIGN_TLS_SHUTDOWN_MARKERS)


def _asyncio_exception_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """Filter out benign TLS-shutdown noise; defer all other errors to the default handler."""
    exc = context.get("exception")
    message = context.get("message", "")
    if _is_benign_tls_shutdown_error(exc, message):
        logger.debug("Ignoring benign TLS shutdown error: {}", message or exc)
        return
    loop.default_exception_handler(context)


async def main_async() -> None:
    """Async main application entry point for Hypercorn"""
    app = create_app()
    logger.info(
        "Starting {} version {} in {} mode via Hypercorn (HTTP/2)",
        settings.app_name,
        settings.version,
        settings.environment,
    )

    # WebView2 / browser clients frequently drop HTTP/2-over-TLS connections without a
    # clean TLS close, so asyncio's graceful SSL shutdown either times out or observes a
    # late record. Hypercorn surfaces these as "Unhandled exception in client_connected_cb"
    # at ERROR level even though they are harmless connection teardown noise. Install an
    # exception handler that downgrades just these cases and delegates everything else.
    asyncio.get_running_loop().set_exception_handler(_asyncio_exception_handler)

    setup_hypercorn_logging(settings.log_level)
    config = Config()
    config.bind = [f"{settings.host}:{settings.port}"]
    certfile = settings.data_dir / settings.certfile
    keyfile = settings.data_dir / settings.keyfile
    ensure_certs_exist(certfile, keyfile)
    config.certfile = str(certfile)
    config.keyfile = str(keyfile)
    config.accesslog = "-"
    config.errorlog = "-"
    config.loglevel = settings.log_level.upper()

    await serve(cast(ASGIFramework, app), config)


def main() -> None:
    """Synchronous wrapper to start the async loop"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Application shutdown cleanly via KeyboardInterrupt")


if __name__ == "__main__":
    main()
