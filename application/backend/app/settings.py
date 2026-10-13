# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application configuration management"""

import os
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _read_version() -> str:
    """Read the application version, preferring installed package metadata.

    In the Docker image (and any installed environment) the ``geti`` package is
    installed and its version is stamped from the ``VERSION`` file at build time,
    so it can be read from the package metadata. When running from source without
    an installed distribution, fall back to reading the ``VERSION`` file directly.
    """
    try:
        return _pkg_version("geti")
    except PackageNotFoundError:
        # settings.py -> app -> backend -> application/VERSION
        version_file = Path(__file__).resolve().parents[2] / "VERSION"
        try:
            return version_file.read_text(encoding="utf-8").strip()
        except OSError:
            return "0.0.0"


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    # Application
    app_name: str = "Geti"
    version: str = Field(default_factory=_read_version)
    summary: str = "Geti server"
    description: str = (
        "Geti allows to fine-tune computer vision models at the edge. "
        "It is a lightweight application that covers the complete model AI lifecycle, "
        "including advanced features for inference, model monitoring and data collection."
    )
    openapi_url: str = "/api/openapi.json"
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: Literal["dev", "prod"] = "dev"
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    log_dir: Path = Field(default=Path("logs"), alias="LOG_DIR")
    worker_dir: Path | None = None
    job_dir: Path | None = None
    staged_datasets_dir: Path | None = None
    source_media_dir: Path | None = None
    uploads_dir: Path | None = None  # default: data_dir / "uploads"
    upload_max_size: int = Field(
        default=50 * 1024**3, gt=0, description="Maximum size in bytes of a resumable upload (Tus-Max-Size)"
    )
    upload_ttl_hours: int = Field(
        default=24, gt=0, description="Hours of inactivity after which an unfinished resumable upload expires"
    )

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")  # noqa: S104
    port: int = Field(default=7860, alias="PORT")
    static_files_dir: Path | None = Field(
        default=None,
        alias="STATIC_FILES_DIR",
        description="Directory containing static UI files",
    )

    # CORS
    cors_origins: str = Field(
        default="http://localhost:3000, https://localhost:7860",
        alias="CORS_ORIGINS",
    )

    # Database
    database_file: str = Field(default="geti.db", alias="DATABASE_FILE", description="Database filename")
    db_echo: bool = Field(default=False, alias="DB_ECHO")

    # Alembic
    alembic_config_path: str = "app/alembic.ini"
    alembic_script_location: str = "app/alembic"

    # Proxy settings
    no_proxy: str = Field(default="localhost,127.0.0.1,::1", alias="no_proxy")

    gpu_slots: int = Field(default=1, alias="GPU_SLOTS", description="Number of GPU slots available for model tuning")

    # WebRTC
    webrtc_advertise_ip: str | None = Field(default=None, alias="WEBRTC_ADVERTISE_IP")

    # Simplified WebRTC config
    coturn_host: str | None = Field(default=None, alias="COTURN_HOST")
    coturn_port: int = Field(default=443, alias="COTURN_PORT")
    coturn_username: str = Field(default="user", alias="COTURN_USERNAME")
    coturn_password: str = Field(default="password", alias="COTURN_PASSWORD")
    stun_server: str | None = Field(default=None, alias="STUN_SERVER")

    # Inference
    inference_media_limit: int = Field(
        default=64,
        alias="INFERENCE_MEDIA_LIMIT",
        description="Maximum number for images or video frames passed for inference",
    )
    inference_model_ttl: int = Field(
        default=120,
        alias="INFERENCE_MODEL_TTL",
        description="Time to live for a model loaded for inference, before unloading",
    )
    inference_keyframe_stride: int = Field(
        default=5,
        alias="INFERENCE_KEYFRAME_STRIDE",
        description=(
            "This stride value controls which video frames are considered 'key frames' and therefore "
            "sent to the model for inference, while predictions for other frames are interpolated based "
            "on key frames. Key frames satisfy the condition frame_index % stride == 0. "
            "Additionally, the first and last frames are also considered key frames."
        ),
        gt=0,
    )
    inference_max_models: int = Field(
        default=2,
        alias="INFERENCE_MAX_MODELS",
        description="Maximum number of models the inference server keeps loaded in memory at the same time",
        ge=1,
    )

    # SAM (Segment Anything Model)
    sam_encoder_xml_path: Path = Field(
        # The OpenVINO SAM encoder ships inside the ``app/services/sam`` Python module, so the
        # default is resolved relative to this file rather than the current working directory.
        # settings.py -> app; the encoder lives at app/services/sam/mobile_sam.encoder.xml
        default=Path(__file__).resolve().parent / "services" / "sam" / "mobile_sam.encoder.xml",
        alias="SAM_ENCODER_XML_PATH",
        description="Path to the OpenVINO SAM image-encoder model XML file",
    )
    sam_ov_cache_path: Path | None = Field(
        default=None,
        alias="SAM_OV_CACHE_PATH",
        description="Directory for the OpenVINO model cache used by the SAM encoder. "
        "Defaults to <data_dir>/.sam_ov_cache.",
    )

    # Video
    video_cache_ttl: float = Field(
        default=30.0,
        alias="VIDEO_CACHE_TTL",
        description="Time-to-live in seconds for cached video handles before eviction",
    )
    video_cache_cleanup_interval: float = Field(
        default=5.0,
        alias="VIDEO_CACHE_CLEANUP_INTERVAL",
        description="Interval in seconds between cache cleanup sweeps",
    )

    # Certificates
    certfile: Path = Field(
        default=Path("certs/localhost.pem"),
        alias="CERTFILE",
        description="Path to certificate file",
    )
    keyfile: Path = Field(
        default=Path("certs/localhost-key.pem"),
        alias="KEYFILE",
        description="Path to private key file",
    )

    @property
    def ice_servers(self) -> list[dict]:
        """Compute ICE servers from coturn and STUN configuration."""
        servers = []
        if self.coturn_host:
            servers.append(
                {
                    "urls": f"turn:{self.coturn_host}:{self.coturn_port}?transport=tcp",
                    "username": self.coturn_username,
                    "credential": self.coturn_password,
                }
            )

        if self.stun_server:
            servers.append({"urls": self.stun_server})

        return servers

    @property
    def database_url(self) -> str:
        """Get database URL"""
        return f"sqlite:///{self.data_dir / self.database_file}"

    @property
    def cors_allowed_origins(self) -> list[str]:
        """Parsed list of allowed CORS origins.

        Always includes localhost / 127.0.0.1 entries for the configured server port,
        so that browser requests from a forwarded / non-default port are accepted.
        """
        origins = [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        for host in ("localhost", "127.0.0.1"):
            for scheme in ("http", "https"):
                entry = f"{scheme}://{host}:{self.port}"
                if entry not in origins:
                    origins.append(entry)
        return origins

    @model_validator(mode="after")
    def set_default_dirs(self) -> "Settings":
        """Set default directories based on log_dir"""
        if self.worker_dir is None:
            self.worker_dir = self.log_dir / "workers"
        if self.job_dir is None:
            self.job_dir = self.log_dir / "jobs"
        if self.staged_datasets_dir is None:
            self.staged_datasets_dir = self.data_dir / "staged_datasets"
        if self.source_media_dir is None:
            self.source_media_dir = self.data_dir / "source_media"
        if self.uploads_dir is None:
            self.uploads_dir = self.data_dir / "uploads"
        if self.sam_ov_cache_path is None:
            self.sam_ov_cache_path = self.data_dir / ".sam_ov_cache"

        return self

    def ensure_dirs_exist(self) -> None:
        """Create all directories if they don't exist."""
        for d in [
            self.data_dir,
            self.log_dir,
            self.worker_dir,
            self.job_dir,
            self.staged_datasets_dir,
            self.source_media_dir,
            self.uploads_dir,
            self.sam_ov_cache_path,
        ]:
            if d:
                d.mkdir(parents=True, exist_ok=True)

    @field_validator("static_files_dir", "alembic_config_path", "alembic_script_location", mode="after")
    def prefix_paths(cls, v: str | Path | None) -> str | Path | None:
        # In "frozen" pyinstaller applications data paths must be prefixed with the absolute path to the bundle folder
        # which is stored in  sys._MEIPASS attribute.
        # https://pyinstaller.org/en/stable/runtime-information.html
        if v and getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            # If application is running in pyinstaller bundle, adjust the path accordingly.
            prefixed_path = os.path.join(getattr(sys, "_MEIPASS", ""), v)
            return Path(prefixed_path) if isinstance(v, Path) else prefixed_path
        return v

    @field_validator("stun_server")
    def validate_stun_server(cls, v: str | None) -> str | None:
        """Ensure that STUN server, if set, is a full ICE URL starting with stun: or stuns:."""
        if v:
            if not isinstance(v, str):
                raise TypeError("stun_server must be a string.")
            if not (v.startswith(("stun:", "stuns:"))):
                raise ValueError("stun_server must be a full ICE URL starting with 'stun:' or 'stuns:'.")
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()
