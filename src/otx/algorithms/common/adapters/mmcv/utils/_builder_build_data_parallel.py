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

from otx.algorithms.common.utils import is_xpu_available, is_hpu_available
import habana_frameworks.torch as htorch
from torch._utils import _get_device_index


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
        model = model.hpu()
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
    def __init__(self, *args, enable_autocast: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_autocast = enable_autocast

    def scatter(self, inputs, kwargs, device_ids):
        inputs, kwargs = super().scatter(inputs, kwargs, [-1])
        target_device = torch.device(f"xpu:{device_ids[0]}")

        for x in inputs:
            if isinstance(x, tuple):
                for val in x:
                    if isinstance(val, dict):
                        for k in val:
                            if isinstance(val[k], torch.Tensor):
                                val[k] = val[k].to(target_device)
                            elif isinstance(val[k], list):
                                for i, item in enumerate(val[k]):
                                    if isinstance(item, torch.Tensor):
                                        val[k][i] = item.to(target_device)

        for x in kwargs:
            if isinstance(x, dict):
                for k in x:
                    if isinstance(x[k], torch.Tensor):
                        x[k] = x[k].to(target_device)
                    elif isinstance(x[k], list):
                        for i, item in enumerate(x[k]):
                            if isinstance(item, torch.Tensor):
                                x[k][i] = item.to(target_device)

        return inputs, kwargs

    def forward(self, *inputs, **kwargs):
        # we have to apply autocast here, because the original mmcv's fp16 decorator is hard to override.
        # Perhaps, one global autocast is not as accurate as original mmcv's approach
        with torch.autocast(device_type="xpu", dtype=torch.bfloat16, enabled=self.enable_autocast):
            return super().forward(*inputs, **kwargs)

    def train_step(self, *inputs, **kwargs):
        with torch.autocast(device_type="xpu", dtype=torch.bfloat16, enabled=self.enable_autocast):
            return super().train_step(*inputs, **kwargs)

    def val_step(self, *inputs, **kwargs):
        with torch.autocast(device_type="xpu", dtype=torch.bfloat16, enabled=self.enable_autocast):
            return super().val_step(*inputs, **kwargs)
    
    
def _get_available_device_type():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
        return "xpu"
    if is_hpu_available():
        return "hpu"
    # add more available device types here
    return None


def _get_device_attr(get_member):
    device_type = _get_available_device_type()
    if device_type and device_type.lower() == "cuda":
        return get_member(torch.cuda)
    if device_type and device_type.lower() == "xpu":
        return get_member(torch.xpu)  # type: ignore[attr-defined]
    if device_type and device_type.lower() == "hpu":
        return get_member(htorch.hpu)
    # add more available device types here
    return None


def _get_all_device_indices():
    # all device index
    return _get_device_attr(lambda m: list(range(m.device_count())))


class HPUDataParallel(MMDataParallel):
    def __init__(self, *args, enable_autocast: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_autocast = enable_autocast
        self.src_device_obj = torch.device("hpu", self.device_ids[0])
        self.module.to(self.src_device_obj)

    def scatter(self, inputs, kwargs, device_ids):
        inputs, kwargs = super().scatter(inputs, kwargs, [-1])

        for x in inputs:
            if isinstance(x, tuple):
                for val in x:
                    if isinstance(val, dict):
                        for k in val:
                            if isinstance(val[k], torch.Tensor):
                                val[k] = val[k].to(torch.device(f"hpu:{device_ids[0]}"))
                            elif isinstance(val[k], list):
                                for i, item in enumerate(val[k]):
                                    if isinstance(item, torch.Tensor):
                                        val[k][i] = item.to(torch.device(f"hpu:{device_ids[0]}"))

        for x in kwargs:
            if isinstance(x, dict):
                for k in x:
                    if isinstance(x[k], torch.Tensor):
                        x[k] = x[k].to("hpu")
                    elif isinstance(x[k], list):
                        for i, item in enumerate(x[k]):
                            if isinstance(item, torch.Tensor):
                                x[k][i] = item.to(torch.device(f"hpu:{device_ids[0]}"))

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
