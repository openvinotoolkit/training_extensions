# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib

from otx.v2.adapters.torch.mmengine.mmdeploy.utils.mmdeploy import (
    is_mmdeploy_enabled,
)


def test_is_mmdeploy_enabled() -> None:
    assert (importlib.util.find_spec("mmdeploy") is not None) == is_mmdeploy_enabled()
