"""OpenVINO Training Extensions."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

__version__ = "2.2.0rc2"

import os
from pathlib import Path

from otx.core.types import *  # noqa: F403

# Set the value of HF_HUB_CACHE to set the cache folder that stores the pretrained weights for timm and huggingface.
# Refer: huggingface_hub/constants.py::HF_HUB_CACHE
# Default, Pretrained weight is saved into ~/.cache/torch/hub/checkpoints
os.environ["HF_HUB_CACHE"] = os.getenv(
    "HF_HUB_CACHE",
    str(Path.home() / ".cache" / "torch" / "hub" / "checkpoints"),
)

OTX_LOGO: str = """

 ██████╗  ████████╗ ██╗  ██╗
██╔═══██╗ ╚══██╔══╝ ╚██╗██╔╝
██║   ██║    ██║     ╚███╔╝
██║   ██║    ██║     ██╔██╗
╚██████╔╝    ██║    ██╔╝ ██╗
 ╚═════╝     ╚═╝    ╚═╝  ╚═╝

"""
