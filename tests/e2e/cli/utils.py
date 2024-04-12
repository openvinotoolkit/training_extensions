# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

from otx.cli import main


def run_main(command_cfg: list[str], open_subprocess: bool) -> None:
    if open_subprocess:
        _run_main_with_open_subprocess(command_cfg)
    else:
        _run_main(command_cfg)


def _run_main_with_open_subprocess(command_cfg) -> None:
    completed = subprocess.run(
        [sys.executable, __file__, *command_cfg],  # noqa: S603
        check=True,
    )

    completed.check_returncode()


def _run_main(command_cfg) -> None:
    with patch("sys.argv", command_cfg):
        main()


if __name__ == "__main__":
    _run_main(sys.argv[1:])
