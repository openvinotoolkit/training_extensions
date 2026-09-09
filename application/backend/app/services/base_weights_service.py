# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import hashlib
import shutil
from pathlib import Path
from typing import cast

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from app.models import ModelManifest, TaskType
from app.models.model_manifest import DirectLinkPretrainedWeights

from .model_manifest_service import ModelManifestService


class BaseWeightsService:
    """Service for downloading and managing pretrained model weights from external archives."""

    REQUEST_TIMEOUT = (5, 600)  # (connect timeout, read timeout) in seconds
    RETRY_TOTAL = 2  # total number of retries for failed requests
    RETRY_CONNECT = 1  # retries specifically for connection failures (fail fast on unreachable hosts)
    RETRY_BACKOFF_FACTOR = 0.5  # exponential backoff factor for retries (e.g., 0.5s, 1s)

    def __init__(self, data_dir: Path) -> None:
        self.pretrained_weights_dir = data_dir / "pretrained_weights"
        for task in TaskType:
            task_dir = self.pretrained_weights_dir / task.name.lower()
            task_dir.mkdir(parents=True, exist_ok=True)

    def get_remote_weights_path(self, task: TaskType, model_manifest_id: str) -> str:
        """
        Return the remote location of the weights as configured in the manifest.

        Args:
            task: The task type (used for validation)
            model_manifest_id: The unique identifier of the model architecture

        Returns:
            str: The remote URL of the pretrained weights
        """
        manifest = self._get_and_validate_model_manifest(task, model_manifest_id)
        pretrained_weights = cast(DirectLinkPretrainedWeights, manifest.pretrained_weights)
        return pretrained_weights.url

    def get_local_weights_path(self, task: TaskType, model_manifest_id: str, allow_download: bool = True) -> Path:
        """
        Return the location of the weights (.pt file).

        If not already present and allow_download is enabled, downloads the weights from remote.

        Args:
            task: The task type
            model_manifest_id: The unique identifier of the model architecture
            allow_download: Whether to download weights if not present locally

        Returns:
            Path: Path to the local weights file

        Raises:
            FileNotFoundError: If weights are not found locally and allow_download is False
        """
        manifest = self._get_and_validate_model_manifest(task, model_manifest_id)

        pretrained_weights = cast(DirectLinkPretrainedWeights, manifest.pretrained_weights)
        local_filename = pretrained_weights.local_filename
        local_path = self.pretrained_weights_dir / task.name.lower() / local_filename
        if local_path.exists():
            if self._verify_file_integrity(file_path=local_path, sha_sum=pretrained_weights.sha_sum):
                logger.info("Using cached weights for {}: {}", model_manifest_id, local_path)
                return local_path

            logger.warning("Cached weights for {} failed integrity check, will re-download", model_manifest_id)
            local_path.unlink()

        if not allow_download:
            raise FileNotFoundError(f"Weights not found locally for model {model_manifest_id} and download is disabled")

        logger.info("Downloading pretrained weights for {}", model_manifest_id)
        urls = [pretrained_weights.url]
        if pretrained_weights.mirror_url is not None:
            urls.append(pretrained_weights.mirror_url)
        self._download_weights(
            urls=urls,
            local_path=local_path,
            sha_sum=pretrained_weights.sha_sum,
        )

        return local_path

    def remove_local_weights(self, task: TaskType, model_manifest_id: str) -> bool:
        """
        Delete the local base weights of a specific model architecture to free space on disk.

        Args:
            task: The task type
            model_manifest_id: The unique identifier of the model architecture

        Returns:
            bool: True if weights were successfully removed, False if they didn't exist
        """
        manifest = self._get_and_validate_model_manifest(task, model_manifest_id)
        pretrained_weights = cast(DirectLinkPretrainedWeights, manifest.pretrained_weights)
        local_path = self.pretrained_weights_dir / task.name.lower() / pretrained_weights.local_filename
        if local_path.exists():
            try:
                local_path.unlink()
                logger.info("Removed local weights for {}: {}", model_manifest_id, local_path)
                return True
            except OSError as e:
                logger.error("Failed to remove weights file {}: {}", local_path, e)

        return False

    def remove_all_local_weights(self) -> int:
        """
        Remove all locally cached pretrained weights to free space on disk.

        Returns:
            int: Number of weight files that were removed
        """
        removed_count = 0
        if not self.pretrained_weights_dir.exists():
            return 0

        try:
            for weights_file in self.pretrained_weights_dir.rglob("*"):
                if weights_file.is_file() and not weights_file.name.startswith("."):
                    try:
                        weights_file.unlink()
                        removed_count += 1
                        logger.debug("Removed weights file: {}", weights_file)
                    except OSError as e:
                        logger.error("Failed to remove weights file {}: {}", weights_file, e)

            logger.info("Removed {} cached weight files", removed_count)
            return removed_count
        except OSError as e:
            logger.error("Failed to remove cached weights: {}", e)
            return 0

    @staticmethod
    def _verify_file_integrity(file_path: Path, sha_sum: str) -> bool:
        """
        Verify the integrity of a downloaded file using SHA256 checksum.

        Args:
            file_path: Path to the file to verify
            sha_sum: Expected SHA256 checksum

        Returns:
            bool: True if the file integrity is valid, False otherwise
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

            actual_sha_sum = sha256_hash.hexdigest()
            return actual_sha_sum == sha_sum
        except Exception as e:
            logger.error("Failed to verify file integrity for {}: {}", file_path, e)
            return False

    def _check_disk_space(self, remote_url: str, safety_margin_gb: float = 1.0) -> None:
        """
        Check if there's sufficient disk space for downloading the remote file.

        Args:
            remote_url: URL of the file to download
            safety_margin_gb: Additional safety margin in GB

        Raises:
            OSError: If there's insufficient disk space
        """
        # Use a conservative default (500MB) if the remote size cannot be queried.
        file_size = 500 * 1024 * 1024
        # Try with and without proxies, as some environments may have proxy issues that cause the HEAD request to fail
        for use_env_proxy in (True, False):
            try:
                with self._build_retry_session(use_env_proxy=use_env_proxy) as session:
                    response = session.head(remote_url, allow_redirects=True, timeout=self.REQUEST_TIMEOUT)
                    response.raise_for_status()

                    content_length = response.headers.get("content-length")
                    if content_length:
                        file_size = int(content_length)
                    else:
                        logger.warning("Could not determine file size for {}, assuming 500MB", remote_url)
                    break
            except Exception as e:
                # Log per mode so we can see whether proxy or direct access failed.
                proxy_mode = "env-proxy" if use_env_proxy else "direct"
                logger.warning("Could not check remote file size for {} via {}: {}", remote_url, proxy_mode, e)
        else:
            logger.warning("Could not check remote file size for {}; assuming 500MB", remote_url)

        stat = shutil.disk_usage(self.pretrained_weights_dir)
        available_space = stat.free
        required_space = file_size + (safety_margin_gb * 1024 * 1024 * 1024)

        if available_space < required_space:
            raise OSError(
                f"Insufficient disk space. Required: {required_space / (1024**3):.2f} GB, "
                f"Available: {available_space / (1024**3):.2f} GB"
            )

    def _download_weights(self, urls: list[str], local_path: Path, sha_sum: str) -> None:
        """
        Download weights from the first working URL among the given candidates (e.g. primary then mirror).
        Before download, checks if there is enough space on disk.

        Args:
            urls: Candidate URLs in priority order.
            local_path: Local path to save the file
            sha_sum: Expected SHA256 checksum for verification

        Raises:
            RuntimeError: If none of the candidate URLs yield a file that passes integrity verification.
        """
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = local_path.with_suffix(".tmp")

        last_error: Exception | None = None
        try:
            for url in urls:
                logger.info("Downloading pretrained weights from {}", url)
                try:
                    self._check_disk_space(url)
                    self._download_from_url(url, temp_path)
                except requests.RequestException as e:
                    last_error = e
                    logger.warning("Failed to download weights from {}: {}", url, e)
                    continue

                if self._verify_file_integrity(temp_path, sha_sum):
                    temp_path.rename(local_path)
                    logger.info("Successfully downloaded and verified weights: {}", local_path)
                    return

                last_error = RuntimeError(f"Integrity check failed for weights downloaded from {url}")
                logger.warning("Downloaded weights from {} failed integrity check, trying next candidate URL", url)
                temp_path.unlink(missing_ok=True)

            raise RuntimeError(
                f"Failed to download weights from any of the candidate URLs {urls}: {last_error}"
            ) from last_error

        finally:
            # Clean up temporary file if it exists
            if temp_path.exists():
                temp_path.unlink()

    def _download_from_url(self, url: str, temp_path: Path) -> None:
        """
        Download a single URL to temp_path, retrying with & without the environment proxy on failure.

        Raises:
            requests.RequestException: If both the proxied and direct attempts fail.
        """
        last_error: Exception | None = None
        for use_env_proxy in (True, False):
            try:
                with (
                    self._build_retry_session(use_env_proxy=use_env_proxy) as session,
                    session.get(url, stream=True, timeout=self.REQUEST_TIMEOUT) as response,
                ):
                    response.raise_for_status()
                    with open(temp_path, "wb") as f:
                        for data in response.iter_content(chunk_size=4096):
                            f.write(data)
                return
            except requests.RequestException as e:
                last_error = e
                proxy_mode = "env-proxy" if use_env_proxy else "direct"
                logger.warning("Weight download failed via {} for {}: {}", proxy_mode, url, e)
        raise requests.RequestException(f"Failed to download from {url}") from last_error

    def _build_retry_session(self, use_env_proxy: bool) -> requests.Session:
        session = requests.Session()
        # trust_env=False disables HTTP(S)_PROXY and NO_PROXY from environment.
        session.trust_env = use_env_proxy

        retry = Retry(
            total=self.RETRY_TOTAL,
            connect=self.RETRY_CONNECT,
            read=self.RETRY_TOTAL,
            backoff_factor=self.RETRY_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["HEAD", "GET"]),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @staticmethod
    def _get_and_validate_model_manifest(task: TaskType, model_manifest_id: str) -> ModelManifest:
        """
        Validate or retrieve the default model manifest ID for a given task.

        Args:
            task: The task type (e.g., classification, detection)
            model_manifest_id: The provided model manifest ID, or None to use default

        Returns:
            ModelManifest: The validated model manifest

        Raises:
            ValueError: If the model manifest is not found, task type mismatch, or doesn't have pretrained weights
        """
        manifest = ModelManifestService.get_model_manifest_by_id(model_manifest_id)
        if manifest.task != task:
            raise ValueError(f"Task mismatch: expected '{task.name.lower()}', got '{manifest.task.lower()}'")
        return manifest
