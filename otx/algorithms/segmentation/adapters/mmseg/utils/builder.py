"""MMseg model builder."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from typing import Optional, Union

import torch
from mmcv.runner import load_checkpoint
from mmcv.utils import Config, ConfigDict
from mmseg.models.builder import MODELS

SCALAR_SCHEDULERS = MODELS


def build_scalar_scheduler(cfg, default_value=None):
    """Build scalar scheduler."""
    if cfg is None:
        if default_value is not None:
            assert isinstance(default_value, (int, float))
            cfg = dict(type="ConstantScalarScheduler", scale=float(default_value))
        else:
            return None
    elif isinstance(cfg, (int, float)):
        cfg = dict(type="ConstantScalarScheduler", scale=float(cfg))

    return SCALAR_SCHEDULERS.build(cfg)


def build_segmentor(
    config: Config,
    train_cfg: Optional[Union[Config, ConfigDict]] = None,
    test_cfg: Optional[Union[Config, ConfigDict]] = None,
    checkpoint: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    cfg_options: Optional[Union[Config, ConfigDict]] = None,
    from_scratch: bool = False,
) -> torch.nn.Module:
    """A builder function for mmseg model.

    Creates a model, based on the configuration in config.
    Note that this function updates 'load_from' attribute of 'config'.
    """

    # fmt: off
    # isort: off
    # false positive (mypy, pylint)
    # pylint: disable-next=no-name-in-module
    from mmseg.models import build_segmentor as origin_build_segmentor  # type: ignore[attr-defined]
    # isort: on
    # fmt: on

    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    model_cfg = deepcopy(config.model)
    model = origin_build_segmentor(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    model = model.to(device)

    checkpoint = checkpoint if checkpoint else config.pop("load_from", None)
    if checkpoint is not None and not from_scratch:
        load_checkpoint(model, checkpoint, map_location=device)
    config.load_from = checkpoint

    return model
