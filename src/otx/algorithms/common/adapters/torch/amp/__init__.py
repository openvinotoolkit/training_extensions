"""Custom AMP (Automatic Mixed Precision package) in OTX."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

try:
    from .xpu_grad_scaler import XPUGradScaler  # noqa: F401
except:  # noqa: E722
    pass
