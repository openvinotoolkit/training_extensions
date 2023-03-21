"""OpenVINO Training Extensions."""

# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

__version__ = "1.1.0rc1"
# NOTE: Sync w/ otx/api/usecases/exportable_code/demo/requirements.txt on release

MMCLS_AVAILABLE = True
MMDET_AVAILABLE = True
MMSEG_AVAILABLE = True
MMACTION_AVAILABLE = True

try:
    import mmcls  # noqa: F401
except ImportError:
    MMCLS_AVAILABLE = False

try:
    import mmdet  # noqa: F401
except ImportError:
    MMDET_AVAILABLE = False

try:
    import mmseg  # noqa: F401
except ImportError:
    MMSEG_AVAILABLE = False

try:
    import mmaction  # noqa: F401
except ImportError:
    MMACTION_AVAILABLE = False
