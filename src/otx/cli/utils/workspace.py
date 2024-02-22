"""Class modules that manage Workspace."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class Workspace:
    """Represents a workspace for the OTX application.

    Args:
        work_dir (Path | str | None, optional): The path to the workspace directory. Defaults to None.
        use_sub_dir (bool, optional): Whether to use a subdirectory within the workspace. Defaults to True.
    """

    def __init__(self, work_dir: Path | str = Path.cwd(), use_sub_dir: bool = True):  # noqa: B008
        work_dir = Path(work_dir)
        self.work_dir = (
            work_dir / "otx-workspace"
            # Without work_dir input & no .latest directory in root
            if work_dir == Path.cwd() and not (work_dir / ".latest").exists()
            else work_dir
        )
        if use_sub_dir:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.work_dir = self.work_dir / f"{timestamp}"
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
