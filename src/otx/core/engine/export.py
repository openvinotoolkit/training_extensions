# MIT License

# Copyright (c) 2023 Intel Corporation
# Copyright (c) 2021 ashleve

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This source code is borrowed from https://github.com/ashleve/lightning-hydra-template

"""Engine component to training pipeline."""
from __future__ import annotations

import logging as log

import hydra

from otx.core.config import TrainConfig
from otx.core.model.entity.base import OTXModel


def export(
    cfg: TrainConfig,
    otx_model: OTXModel | None = None,
):
    """Exports the model. Can additionally evaluate on a testset, using exported model.

    Args:
        cfg: A DictConfig configuration composed by Hydra.
        otx_model: If it is not `None`, the given OTX model will be overrided.

    Returns:
        A tuple with Pytorch Lightning Trainer and Python dict of metrics
    """
    if otx_model is not None:
        log.info(f"Instantiating model <{cfg.model}>")
        if not isinstance(otx_model, OTXModel):
            raise TypeError(otx_model)
        model = otx_model
    else:
        model: OTXModel = hydra.utils.instantiate(cfg.model.otx_model)

    model.export(cfg.base.output_dir, cfg.deploy, test_pipeline=cfg.data.test_subset.transforms)
