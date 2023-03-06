"""OpenVINO Training Extensions."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

__version__ = "1.0.0"
# NOTE: Sync w/ otx/api/usecases/exportable_code/demo/requirements.txt

MMCLS_AVAILABLE = True
MMDET_AVAILABLE = True
MMSEG_AVAILABLE = True
MMACTION_AVAILABLE = True

try:
    import mmcls
except ImportError:
    MMCLS_AVAILABLE = False

try:
    import mmdet
except ImportError:
    MMDET_AVAILABLE = False

try:
    import mmseg
except ImportError:
    MMSEG_AVAILABLE = False

try:
    import mmaction
except ImportError:
    MMACTION_AVAILABLE = False
