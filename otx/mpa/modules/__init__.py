# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# flake8: noqa

try:
    import openvino
except ImportError:
    pass
else:
    from . import ov
