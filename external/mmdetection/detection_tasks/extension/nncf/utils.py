# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import importlib
from collections import OrderedDict
from contextlib import contextmanager

import torch

from mmcv.parallel import MMDataParallel
from mmcv.parallel import MMDistributedDataParallel
from ..utils import MMDataCPU
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)


_is_nncf_enabled = importlib.util.find_spec('nncf') is not None


def is_nncf_enabled():
    return _is_nncf_enabled


def check_nncf_is_enabled():
    if not is_nncf_enabled():
        raise RuntimeError('Tried to use NNCF, but NNCF is not installed')


def get_nncf_version():
    if not is_nncf_enabled():
        return None
    import nncf
    return nncf.__version__


def load_checkpoint(model, filename, map_location=None, strict=False):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    from nncf.torch import load_state

    checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    _ = load_state(model, state_dict, strict)
    return checkpoint


@contextmanager
def nullcontext():
    """
    Context which does nothing
    """
    yield


def no_nncf_trace():
    """
    Wrapper for original NNCF no_nncf_trace() context
    """

    if is_nncf_enabled():
        from nncf.torch.dynamic_graph.context import no_nncf_trace as original_no_nncf_trace
        return original_no_nncf_trace()
    return nullcontext()


def is_in_nncf_tracing():
    if not is_nncf_enabled():
        return False

    from nncf.torch.dynamic_graph.context import get_current_context

    ctx = get_current_context()

    if ctx is None:
        return False
    return ctx.is_tracing

def is_accuracy_aware_training_set(nncf_config):
    if not is_nncf_enabled():
        return False
    from nncf.config.utils import is_accuracy_aware_training
    is_acc_aware_training_set = is_accuracy_aware_training(nncf_config)
    return is_acc_aware_training_set

def prepare_mmdet_model_for_execution(model, cfg, distributed=False):
    """
    Prepare model for execution.
    Return model MMDistributedDataParallel, MMDataParallel or MMDataCPU.

    :param model: Model.
    :param cfg: training mmdet config.
    :param distributed: Enable distributed training mode.
    :return:
    """
    if torch.cuda.is_available():
        if distributed:
            # put model on gpus
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDataCPU(model)
    return model

def build_val_dataloader(cfg, distributed):
    # Support batch_size > 1 in validation
    val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
    if val_samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.val.pipeline = replace_ImageToTensor(
            cfg.data.val.pipeline)
    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    val_dataloader = build_dataloader(
        val_dataset,
        samples_per_gpu=val_samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    return val_dataloader
