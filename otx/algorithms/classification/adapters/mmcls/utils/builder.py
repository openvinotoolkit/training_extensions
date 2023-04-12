"""MMcls model builder."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from typing import Optional, Union

import torch
from mmcv.runner import load_checkpoint
from mmcv.utils import Config, ConfigDict, get_logger

from otx.algorithms.common.utils.logger import LEVEL

logger = get_logger("mmcls")


def build_classifier(
    config: Config,
    checkpoint: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    cfg_options: Optional[Union[Config, ConfigDict]] = None,
    from_scratch: bool = False,
) -> torch.nn.Module:
    """A builder function for mmcls model.

    Creates a model, based on the configuration in config.
    Note that this function updates 'load_from' attribute of 'config'.
    """

    from mmcls.models import build_classifier as origin_build_classifier

    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    model_cfg = deepcopy(config.model)
    model = origin_build_classifier(model_cfg)
    logger.setLevel("WARNING")
    model.init_weights()
    logger.setLevel(LEVEL)
    model = model.to(device)

    checkpoint = checkpoint if checkpoint else config.pop("load_from", None)
    if checkpoint is not None and not from_scratch:
        load_checkpoint(model, checkpoint, map_location=device)
    config.load_from = checkpoint

    return model
