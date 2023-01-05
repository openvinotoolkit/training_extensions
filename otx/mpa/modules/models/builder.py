# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmseg.models.builder import MODELS

SCALAR_SCHEDULERS = MODELS


def build_scalar_scheduler(cfg, default_value=None):
    if cfg is None:
        if default_value is not None:
            assert isinstance(default_value, (int, float))
            cfg = dict(type="ConstantScalarScheduler", scale=float(default_value))
        else:
            return None
    elif isinstance(cfg, (int, float)):
        cfg = dict(type="ConstantScalarScheduler", scale=float(cfg))

    return SCALAR_SCHEDULERS.build(cfg)
