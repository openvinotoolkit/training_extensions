"""OpenVINO Training Extensions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from . import algo

__version__ = "1.5.0rc0"
# NOTE: Sync w/ src/otx/api/usecases/exportable_code/demo/requirements.txt on release

__all__ = ["algo"]
