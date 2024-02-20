"""Class modules that manage Workspace."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class Workspace:
    """Represents a workspace for the OTX application.

    Args:
        work_dir (str): The path to the workspace directory. Defaults to "./otx-workspace".
        use_sub_dir (bool): Whether to use a subdirectory within the workspace. Defaults to True.
    """

    def __init__(self, work_dir: Path | str | None = None, use_sub_dir: bool = True):
        if work_dir is None:
            if not (Path.cwd() / ".cache").exists():
                # Without work_dir input & no .cache directory in root
                self.work_dir = Path.cwd() / "otx-workspace"
            else:
                # If Path.cwd is workspace
                self.work_dir = Path.cwd()
        else:
            self.work_dir = Path(work_dir)
        if use_sub_dir:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.work_dir = self.work_dir / f"{timestamp}"
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
