"""NNCF utils."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import inspect
import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import DataContainer, collate, scatter
from torch import nn
from torch.utils.data import DataLoader

from otx.algorithms.common.adapters.mmcv.nncf.runners import NNCF_META_KEY
from otx.algorithms.common.adapters.mmcv.utils.builder import build_data_parallel
from otx.algorithms.common.adapters.nncf.compression import (
    is_checkpoint_nncf,
    is_state_nncf,
)
from otx.algorithms.common.adapters.nncf.utils import (
    check_nncf_is_enabled,
    load_checkpoint,
    no_nncf_trace,
)
from otx.algorithms.common.utils import get_arg_spec
from otx.utils.logger import get_logger

logger = get_logger()


def get_fake_input(
    preprocessor: Callable[..., Dict[str, Any]],
    data: Optional[np.ndarray] = None,
    shape: Tuple[int, ...] = (128, 128, 3),
    device: Union[str, torch.device] = "cpu",
):
    """A function to generate fake data."""

    if isinstance(device, str):
        device = torch.device(device)

    if data is None:
        data = dict(img=np.zeros(shape, dtype=np.uint8))
    else:
        data = dict(img=data)
    data = preprocessor(data)

    for key, value in data.items():
        if not isinstance(value, list):
            data[key] = [value]

    if device.type == "cpu":
        data = scatter(collate([data], samples_per_gpu=1), [-1])[0]
    elif device.type == "cuda":
        data = scatter(collate([data], samples_per_gpu=1), [device.index])[0]
    elif device.type == "xpu":
        data = scatter(collate([data], samples_per_gpu=1), [-1])[0]
    else:
        raise NotImplementedError()
    return data


def model_eval(
    model: nn.Module,
    *,
    config: Config,
    val_dataloader: DataLoader,
    evaluate_fn: Callable,
    distributed: bool,
):
    """A model evaluation function for NNCF.

    Runs evaluation of the model on the validation set and
    returns the target metric value.
    Used to evaluate the original model before compression
    if NNCF-based accuracy-aware training is used.
    """
    if val_dataloader is None:
        raise RuntimeError(
            "Cannot perform model evaluation on the validation "
            "dataset since the validation data loader was not passed "
            "to wrap_nncf_model"
        )

    nncf_config = config.get("nncf_config")
    metric_name = nncf_config.get("target_metric_name")
    prepared_model = build_data_parallel(model, config, distributed=distributed)

    logger.info("Calculating an original model accuracy")

    evaluation_cfg = deepcopy(config.evaluation)
    spec = get_arg_spec(val_dataloader.dataset.evaluate)
    for key in list(evaluation_cfg.keys()):
        if key not in spec:
            evaluation_cfg.pop(key)
    evaluation_cfg["metric"] = metric_name

    if distributed:  # pylint: disable=no-else-return
        dist_eval_res: List[Dict[str, Any]] = [{}]
        results = evaluate_fn(prepared_model, val_dataloader, gpu_collect=True)
        if torch.distributed.get_rank() == 0:
            eval_res = val_dataloader.dataset.evaluate(results, **evaluation_cfg)
            if metric_name not in eval_res:
                raise RuntimeError(f"Cannot find {metric_name} metric in the evaluation result dict")
            dist_eval_res[0] = eval_res

        torch.distributed.broadcast_object_list(dist_eval_res, src=0)
        return dist_eval_res[0][metric_name]
    else:
        results = evaluate_fn(prepared_model, val_dataloader, show=False)
        eval_res = val_dataloader.dataset.evaluate(results, **evaluation_cfg)

        if metric_name not in eval_res:
            raise RuntimeError(f"Cannot find {metric_name} metric in the evaluation result dict {eval_res.keys()}")

        return eval_res[metric_name]


def nncf_state_dict_pre_hook(state_dict, prefix, *args, **kwargs):
    """NNCF-specific state dict pre-hook.

    This hook removes extra prefixes from nncf-related parameters
    before loading to NNCF-ready model.
    """
    for key in list(state_dict.keys()):
        val = state_dict.pop(key)
        if "_nncf" in key:
            if key.startswith("backbone"):
                key = key.replace("backbone.", "", 1)
        state_dict[key] = val

    return state_dict


def nncf_state_dict_hook(module, state_dict, prefix, *args, **kwargs):
    """NNCF-specific state dict post-hook.

    This hook prevents extra buffers from being saved to state dict,
    reverting this behavior, introduced by mmcv.
    """
    for key in list(state_dict.keys()):
        val = state_dict.pop(key)
        if "_level_high" in key or "_level_low" in key:
            continue

        state_dict[key] = val

    return state_dict


# pylint: disable-next=too-many-branches,too-many-statements,too-many-locals
def wrap_nncf_model(  # noqa: C901
    config: Config,
    model: nn.Module,
    *,
    model_eval_fn: Optional[Callable] = None,
    dummy_forward_fn: Optional[Callable] = None,
    get_fake_input_fn: Optional[Callable] = None,
    wrap_inputs_fn: Optional[Callable] = None,
    dataloader_for_init: Optional[DataLoader] = None,
    init_state_dict: Optional[Dict[Any, Any]] = None,
    is_accuracy_aware: bool = False,
):
    """The function wraps mmcv model by NNCF."""

    check_nncf_is_enabled()

    from nncf import NNCFConfig
    from nncf.config.utils import is_accuracy_aware_training
    from nncf.torch import (
        create_compressed_model,
        load_state,
        register_default_init_args,
    )
    from nncf.torch.dynamic_graph.io_handling import nncf_model_input
    from nncf.torch.initialization import PTInitializingDataLoader

    class _MMInitializeDataLoader(PTInitializingDataLoader):
        def get_inputs(self, dataloader_output):
            # redefined PTInitializingDataLoader because
            # of DataContainer format in mmdet
            kwargs = {k: v.data[0] if isinstance(v, DataContainer) else v for k, v in dataloader_output.items()}
            # TODO: Check ignore scopes for models!
            # We substituted training to validation inference pipeline.
            # The graphs for model have been changed. Now, we may don't need some ignored scopes in NNCF config.
            kwargs["return_loss"] = False
            kwargs["img_metas"] = [kwargs["img_metas"]]
            kwargs["img"] = [kwargs["img"]]
            # delete labels
            new_kwargs = dict()
            for key, val in kwargs.items():
                if not key.startswith("gt_"):
                    new_kwargs[key] = val

            return (), new_kwargs

    nncf_config = NNCFConfig(config.nncf_config)
    resuming_state_dict = None

    if dataloader_for_init:
        wrapped_loader = _MMInitializeDataLoader(dataloader_for_init)
        eval_fn = model_eval_fn if is_accuracy_aware else None
        nncf_config = register_default_init_args(
            nncf_config,
            wrapped_loader,
            model_eval_fn=eval_fn,
            device=next(model.parameters()).device,
        )

    if config.get("resume_from"):
        checkpoint_path = config.get("resume_from")
        assert is_checkpoint_nncf(checkpoint_path), (
            "It is possible to resume training with NNCF compression from NNCF checkpoints only. "
            'Use "load_from" with non-compressed model for further compression by NNCF.'
        )
    elif config.get("load_from"):
        checkpoint_path = config.get("load_from")
        if not is_checkpoint_nncf(checkpoint_path):
            logger.info("Received non-NNCF checkpoint to start training -- initialization of NNCF fields will be done")
    else:
        checkpoint_path = None

    if not dataloader_for_init and not checkpoint_path and not init_state_dict:
        logger.warning(
            "Either dataloader_for_init or NNCF pre-trained "
            "model checkpoint should be set. Without this, "
            "quantizers will not be initialized"
        )

    if init_state_dict:
        assert is_state_nncf(init_state_dict)
        meta_state = init_state_dict["meta"][NNCF_META_KEY]
        resuming_state_dict = init_state_dict["state_dict"]
        compression_state = meta_state.compression_ctrl
    elif checkpoint_path:
        logger.info(f"Loading NNCF checkpoint from {checkpoint_path}")
        logger.info(
            "Please, note that this first loading is made before addition of "
            "NNCF FakeQuantize nodes to the model, so there may be some "
            "warnings on unexpected keys"
        )
        compression_state, resuming_state_dict = load_checkpoint(model, checkpoint_path)
        logger.info(f"Loaded NNCF checkpoint from {checkpoint_path}")
    else:
        compression_state = None

    if dummy_forward_fn is None:
        assert get_fake_input_fn is not None

        def _get_fake_data_for_forward(nncf_config):
            device = next(model.parameters()).device

            if nncf_config.get("input_info", None) and nncf_config.get("input_info").get("sample_size", None):
                input_size = nncf_config.get("input_info").get("sample_size")
                assert len(input_size) == 4 and input_size[0] == 1
                H, W, C = input_size[2], input_size[3], input_size[1]  # pylint: disable=invalid-name
                shape = tuple([H, W, C])
            else:
                shape = (128, 128, 3)

            with no_nncf_trace():
                return get_fake_input_fn(shape=shape, device=device)

        def dummy_forward_fn(model):
            fake_data = _get_fake_data_for_forward(nncf_config)
            img, img_metas = fake_data["img"], fake_data["img_metas"]

            ctx = model.nncf_trace_context(img_metas)
            with ctx:
                # The device where model is could be changed under this context
                img = [i.to(next(model.parameters()).device) for i in img]
                # Marking data as NNCF network input must be after device movement
                img = [nncf_model_input(i) for i in img]
                model(img)

    if wrap_inputs_fn is None:

        def wrap_inputs_fn(args, kwargs):
            img = kwargs.get("img") if "img" in kwargs else args[0]
            if isinstance(img, list):
                assert len(img) == 1, "Input list must have a length 1"
                assert torch.is_tensor(img[0]), "Input for a model must be a tensor"
                img[0] = nncf_model_input(img[0])
            else:
                assert torch.is_tensor(img), "Input for a model must be a tensor"
                img = nncf_model_input(img)
            if "img" in kwargs:
                kwargs["img"] = img
            else:
                args = (img, *args[1:])
            return args, kwargs

    if "log_dir" in nncf_config:
        os.makedirs(nncf_config["log_dir"], exist_ok=True)

    uncompressed_model_accuracy = None
    if is_accuracy_aware_training(nncf_config) and model_eval_fn is not None:
        # Evaluate model before compressing
        uncompressed_model_accuracy = model_eval_fn(model)

    model._register_state_dict_hook(nncf_state_dict_hook)
    model._register_load_state_dict_pre_hook(nncf_state_dict_pre_hook)

    compression_ctrl, model = create_compressed_model(
        model,
        nncf_config,
        dummy_forward_fn=dummy_forward_fn,
        wrap_inputs_fn=wrap_inputs_fn,
        compression_state=compression_state,
    )

    if uncompressed_model_accuracy is not None:
        model.nncf._uncompressed_model_accuracy = uncompressed_model_accuracy

    # Hiding signature of the forward method is required for model export to work
    model.__class__.forward.__signature__ = inspect.Signature(
        [
            inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
            inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
        ]
    )

    if resuming_state_dict:
        load_state(model, resuming_state_dict, is_resume=True)

    return compression_ctrl, model
