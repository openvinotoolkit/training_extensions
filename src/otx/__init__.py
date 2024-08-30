"""OpenVINO Training Extensions."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

__version__ = "2.2.0rc0"

import os
from pathlib import Path

from otx.core.types import *  # noqa: F403

# Pretrained weight is saved into ~/.cache/torch/hub/checkpoints
os.environ["XDG_CACHE_HOME"] = str(Path.home() / ".cache" / "torch" / "hub" / "checkpoints")

OTX_LOGO: str = """

 ██████╗  ████████╗ ██╗  ██╗
██╔═══██╗ ╚══██╔══╝ ╚██╗██╔╝
██║   ██║    ██║     ╚███╔╝
██║   ██║    ██║     ██╔██╗
╚██████╔╝    ██║    ██╔╝ ██╗
 ╚═════╝     ╚═╝    ╚═╝  ╚═╝

"""
