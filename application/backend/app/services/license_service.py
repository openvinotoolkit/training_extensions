# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""License consent service with file-based persistence."""

from pathlib import Path

from loguru import logger


class LicenseService:
    """Service to track third-party license consent using a version-aware file marker.

    The marker file stores the application version for which the license was accepted.
    When the version changes (e.g., after an upgrade), the license must be re-accepted.
    """

    CONSENT_FILENAME = ".license_accepted"

    def __init__(self, data_dir: Path, app_version: str) -> None:
        self._consent_file = data_dir / self.CONSENT_FILENAME
        self._marker = app_version

    def is_accepted(self) -> bool:
        """Check whether the license has been accepted for the current app version."""
        try:
            if not self._consent_file.exists():
                return False
            accepted_marker = self._consent_file.read_text().strip()
        except OSError as exc:
            logger.warning(
                "Failed to read license consent file {}: {}",
                self._consent_file,
                exc,
            )
            return False
        return accepted_marker == self._marker

    def accept(self) -> None:
        """Record that the user accepted the license terms for the current app version."""
        self._consent_file.write_text(self._marker)
        logger.info("License accepted ({}) — recorded at {}", self._marker, self._consent_file)
