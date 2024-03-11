"""A file for a function build_data_parallel()."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# NOTE: a workaround for https://github.com/python/mypy/issues/5028

import os
from typing import Literal, Union, overload

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from otx.algorithms.common.utils import is_hpu_available, is_xpu_available


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
    elif is_hpu_available() and config.get("gpu_ids", []):
        model = model.to("hpu")
        model = HPUDataParallel(model, device_ids=config.gpu_ids)
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
    def scatter(self, inputs, kwargs, device_ids):
        inputs, kwargs = super().scatter(inputs, kwargs, [-1])
        target_device = torch.device(f"xpu:{device_ids[0]}")

        def change_tensor_device(obj):
            if isinstance(obj, list):
                obj = list(map(change_tensor_device, obj))
            elif isinstance(obj, tuple):
                obj = tuple(map(change_tensor_device, obj))
            elif isinstance(obj, dict):
                obj = {key: change_tensor_device(val) for key, val in obj.items()}
            elif isinstance(obj, torch.Tensor):
                obj = obj.to(target_device)
            return obj

        inputs = change_tensor_device(inputs)
        kwargs = change_tensor_device(kwargs)

        return inputs, kwargs


class HPUDataParallel(MMDataParallel):
    def __init__(self, *args, enable_autocast: bool = False, put_gt_on_device=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_autocast = enable_autocast
        self.put_gt_on_device = put_gt_on_device
        self.src_device_obj = torch.device("hpu", self.device_ids[0])

    def scatter(self, inputs, kwargs, device_ids):
        inputs, kwargs = super().scatter(inputs, kwargs, [-1])

        for x in inputs:
            if isinstance(x, tuple):
                for val in x:
                    if isinstance(val, dict):
                        for k in val:
                            # don't put annotations on the HPU to proceed
                            # post-processing on the CPU
                            if not self.put_gt_on_device and k.startswith("gt_"):
                                continue
                            if isinstance(val[k], torch.Tensor):
                                val[k] = val[k].to(self.src_device_obj)
                            elif isinstance(val[k], list):
                                for i, item in enumerate(val[k]):
                                    if isinstance(item, torch.Tensor):
                                        val[k][i] = item.to(self.src_device_obj)

        for x in kwargs:
            if isinstance(x, dict):
                for k in x:
                    if isinstance(x[k], torch.Tensor):
                        x[k] = x[k].to(f"hpu:{device_ids[0]}")
                    elif isinstance(x[k], list):
                        for i, item in enumerate(x[k]):
                            if isinstance(item, torch.Tensor):
                                x[k][i] = item.to(self.src_device_obj)

        return inputs, kwargs

    def forward(self, *inputs, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.enable_autocast):
            return super().forward(*inputs, **kwargs)

    def train_step(self, *inputs, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.enable_autocast):
            return super().train_step(*inputs, **kwargs)

    def val_step(self, *inputs, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.enable_autocast):
            return super().val_step(*inputs, **kwargs)
