"""A file for a function build_data_parallel()."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# NOTE: a workaround for https://github.com/python/mypy/issues/5028

import os
from typing import Literal, Union, overload

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, DataContainer

from otx.algorithms.common.utils import is_xpu_available

@overload
def build_data_parallel(
    model: torch.nn.Module,
    config: Config,
    *,
    distributed: Literal[True],
) -> MMDistributedDataParallel:
    ...


@overload
def build_data_parallel(
    model: torch.nn.Module,
    config: Config,
    *,
    distributed: Literal[False] = False,
) -> MMDataParallel:
    ...


@overload
def build_data_parallel(
    model: torch.nn.Module,
    config: Config,
    *,
    distributed: bool,
) -> Union[MMDataParallel, MMDistributedDataParallel]:
    ...


def build_data_parallel(
    model: torch.nn.Module,
    config: Config,
    *,
    distributed: bool = False,
) -> Union[MMDataParallel, MMDistributedDataParallel]:
    """Prepare model for execution.

    Return MMDataParallel or MMDistributedDataParallel model.

    :param model: Model.
    :param config: config.
    :param distributed: Enable distributed training mode.
    :return:
    """
    if is_xpu_available() and config.get("gpu_ids", []):
        model = model.xpu()
        model = XPUDataParallel(model, device_ids=config.gpu_ids)
    elif torch.cuda.is_available() and config.get("gpu_ids", []):
        if distributed:
            model = model.cuda()
            # put model on gpus
            find_unused_parameters = config.get("find_unused_parameters", False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model,
                device_ids=[int(os.environ["LOCAL_RANK"])],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            model = model.cuda(config.gpu_ids[0])
            model = MMDataParallel(model, device_ids=config.gpu_ids)
    else:
        # temporarily disable cuda for cpu data parallel
        bak = torch.cuda.is_available
        setattr(torch.cuda, "is_available", lambda: False)
        model = MMDataParallel(model, device_ids=[])
        torch.cuda.is_available = bak
    return model


class XPUDataParallel(MMDataParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def scatter(self, inputs, kwargs,
                device_ids):
        for k in kwargs:
            if k == 'img_metas':
                if isinstance(kwargs[k], list):
                    kwargs[k] = kwargs[k][0].data
                else:
                    kwargs[k] = kwargs[k].data
            if k == 'img':
                kwargs[k][0] = kwargs[k][0].xpu()

        for x in inputs:
            if isinstance(x, dict):
                for k in x:
                    if isinstance(x[k], torch.Tensor):
                        x[k] = x[k].to("xpu")

        return (inputs,), (kwargs, )