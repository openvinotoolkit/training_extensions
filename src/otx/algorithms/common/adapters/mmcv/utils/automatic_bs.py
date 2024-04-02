"""Algorithm to find a proper batch size which is fit to current GPU device for tasks using mmcv."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
from copy import copy
from importlib import import_module
from math import sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import wrap_fp16_model
from torch import distributed as dist
from torch.cuda import is_available as cuda_available
from torch.utils.data import Dataset

from otx.algorithms.common.adapters.mmcv.utils.config_utils import OTXConfig
from otx.algorithms.common.adapters.torch.utils import BsSearchAlgo, sync_batchnorm_2_batchnorm
from otx.algorithms.common.utils import is_xpu_available
from otx.core.data import caching
from otx.utils.logger import get_logger

logger = get_logger()


def _set_value_at_dict_in_dict(target: Dict, key_path: str, value):
    """Set value at dictionary hierarchy structure.

    This function is for setting a value at leaf dictionary node in dictionary hierarchy structure.
    If key doesn't exist in the middle node dictionaray, then make a new dictionary at that and keep going.
    For example, if you want to set value at target["a"]["b"]["c"], then you can call the function as below.
    _set_value_at_dict_in_dict(target, "a.b.c", value)

    Args:
        target (Dict): Target variable.
        key_path (str): Dot delimited dictionary key string.
        value : Value to set.
    """
    keys = key_path.split(".")
    for key in keys[:-1]:
        if key not in target:
            target[key] = {}
        target = target[key]

    target[keys[-1]] = value


def _build_model(model_builder: Callable, cfg: Config) -> torch.nn.Module:
    model = model_builder(cfg)
    if cfg.get("fp16", False):
        wrap_fp16_model(model)
    return model


NNCF_PATCH_MODULE = {
    "mmcls": "otx.algorithms.classification.adapters.mmcls.nncf.patches",
    "mmdet": "otx.algorithms.detection.adapters.mmdet.nncf.patches",
    "mmseg": "otx.algorithms.segmentation.adapters.mmseg.nncf.patches",
}


def _train_func_single_iter(
    batch_size: int,
    train_func: Callable,
    datasets: List[Dataset],
    cfg: OTXConfig,
    is_nncf: bool = False,
    meta: Optional[Dict[str, Any]] = None,
    model: Optional[torch.nn.Module] = None,
    model_builder: Optional[Callable] = None,
) -> None:
    caching.MemCacheHandlerSingleton.create("null", 0)  # initialize mem cache
    _set_batch_size(cfg, batch_size)
    _set_max_epoch(cfg, 1)  # setup for training a single iter to save time

    new_dataset = [SubDataset(datasets[0], batch_size)]

    validate = is_nncf  # nncf needs eval hooks
    if is_nncf:
        pkg_name = inspect.getmodule(train_func).__package__
        for framework in ["mmcls", "mmdet", "mmseg"]:
            if framework in pkg_name:
                import_module(NNCF_PATCH_MODULE[framework])
                break
        else:
            framework = None

        if framework == "mmcls":
            validate = False  # classification task has own custom eval hook

    if model is None:
        model = _build_model(model_builder, cfg)

    if is_nncf:
        model.nncf._uncompressed_model_accuracy = 0

    sync_batchnorm_2_batchnorm(model)

    train_func(
        model=model,
        dataset=new_dataset,
        cfg=cfg,
        distributed=False,
        validate=validate,
        meta=meta,
    )


def _save_nncf_model_weight(model: torch.nn.Module, cfg: OTXConfig, save_path: Path) -> str:
    """Save nncf model weight after nncf finishes to build a model.

    NNCF analyzes and get some statistics when buliding a model, which is time consuming.
    To skip this part, nncf model weight is saved and load it on new process.
    """
    from otx.algorithms.common.adapters.nncf.compression import NNCFMetaState

    file_path = save_path / "nncf_model.pth"
    for custom_hook in cfg.custom_hooks:
        if custom_hook["type"] == "CompressionHook":
            compression_ctrl = custom_hook["compression_ctrl"].get_compression_state()
            break
    else:
        msg = "CompressionHook doesn't exist in custom hooks."
        raise RuntimeError(msg)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {
                "nncf_meta": NNCFMetaState(
                    state_to_build=cfg.runner.nncf_meta.state_to_build,
                    data_to_build=cfg.runner.nncf_meta.data_to_build,
                    compression_ctrl=compression_ctrl,
                ),
                "nncf_enable_compression": True,
            },
        },
        file_path,
    )

    return str(file_path)


def _organize_custom_hooks(custom_hooks: List, is_nncf: bool = False) -> None:
    # Remove hooks due to reasons below
    # for nncf task
    # OTXProgressHook and CompressionHook are added when building a model. Need to remove them to avoid duplication.
    # for normal task
    # OTXProgressHook => prevent progress bar from being 0 and 100 repeatably
    # CancelInterfaceHook => avoid segmentation fault
    # earlystoppinghook => if eval hook is excluded, this hook makes an error due to absence of score history
    # CustomEvalHook => exclude validation in classification task

    if is_nncf:
        hooks_to_remove = ["OTXProgressHook", "CompressionHook"]
    else:
        hooks_to_remove = ["OTXProgressHook", "earlystoppinghook", "CustomEvalHook", "CancelInterfaceHook"]

    idx_hooks_to_remove = []
    for i, hook in enumerate(custom_hooks):
        if not is_nncf and hook["type"] == "AdaptiveTrainSchedulingHook":
            hook["enable_eval_before_run"] = False
        for hook_to_remove in hooks_to_remove:
            if hook_to_remove.lower() in hook["type"].lower():
                idx_hooks_to_remove.append(i)

    if idx_hooks_to_remove:
        idx_hooks_to_remove.sort()
        for i in reversed(idx_hooks_to_remove):
            custom_hooks.pop(i)


def adapt_batch_size(
    train_func: Callable,
    model: torch.nn.Module,
    datasets: List[Dataset],
    cfg: OTXConfig,
    distributed: bool = False,
    is_nncf: bool = False,
    meta: Optional[Dict[str, Any]] = None,
    not_increase: bool = True,
    model_builder: Optional[Callable] = None,
) -> None:
    """Decrease batch size if default batch size isn't fit to current GPU device.

    This function just setup for single iteration training to reduce time for adapting.
    The core part of adapting batch size is done in adapt_batch_size in the torch.utils package.

    Args:
        train_func (Callable): The function to train a model.
            Only cfg, dataset and meta are passed to the function when invoking it.
        model (torch.nn.Module): Model to train.
        datasets (List): List of datasets.
        cfg (OTXConfig): Configuration of a training.
        distributed (bool): whether distributed training or not.
        is_nncf (bool): Whether nncf or not.
        meta (Optional[Dict[str, Any]]): meta information.
        not_increase (bool) : Whether adapting batch size to larger value than default value or not.
        model_builder (Optional[Callable]):
            Function for building a model. If it exsits, a model build from model_builder is used instead of the model
            in the argument. It's required for nncf because nncf changes model , which prevent model from pickling.
    """

    if not (cuda_available() or is_xpu_available()):
        logger.warning("Skip Auto-adaptive batch size: Adaptive batch size supports CUDA or XPU.")
        return

    copied_cfg = copy(cfg)
    copied_cfg.custom_hooks = copy(cfg.custom_hooks)
    copied_cfg.pop("algo_backend", None)

    if is_nncf:
        if model_builder is None:
            msg = "model_builder should be possed for building a nncf model."
            raise RuntimeError(msg)
        temp_dir = TemporaryDirectory("adaptive-bs")
        copied_cfg.load_from = _save_nncf_model_weight(model, cfg, Path(temp_dir.name))

    _organize_custom_hooks(copied_cfg.custom_hooks, is_nncf)

    default_bs = _get_batch_size(cfg)
    if not distributed or (rank := dist.get_rank()) == 0:
        train_func_kwargs = {
            "train_func": train_func,
            "datasets": datasets,
            "cfg": copied_cfg,
            "is_nncf": is_nncf,
            "meta": meta,
        }
        if model_builder is None:
            train_func_kwargs["model"] = model
        else:
            train_func_kwargs["model_builder"] = model_builder

        bs_search_algo = BsSearchAlgo(
            train_func=_train_func_single_iter,
            train_func_kwargs=train_func_kwargs,
            default_bs=default_bs,
            max_bs=len(datasets[0]),
        )
        if not_increase:
            new_batch_size = bs_search_algo.auto_decrease_batch_size()
        else:
            drop_last = cfg.data.get("train_dataloader", {}).get("drop_last", False)
            new_batch_size = bs_search_algo.find_big_enough_batch_size(drop_last)

    if distributed:
        if rank == 0:
            total_try_result = torch.tensor([new_batch_size], dtype=torch.int)
        else:
            total_try_result = torch.empty(1, dtype=torch.int)
        total_try_result = total_try_result.cuda() if torch.cuda.is_available() else total_try_result.xpu()
        dist.broadcast(total_try_result, src=0)
        new_batch_size = total_try_result[0].item()

    if default_bs != new_batch_size:
        _set_batch_size(cfg, new_batch_size)
        origin_lr = cfg.optimizer.lr
        bs_change_ratio = new_batch_size / default_bs
        cfg.optimizer.lr *= sqrt(bs_change_ratio)  # Using root scale instead of linear scale

        logger.info("Adapting batch size is done.")
        logger.info(f"Batch size is adapted : {default_bs} -> {new_batch_size}")
        logger.info(f"learning rate is adapted : {origin_lr} -> {cfg.optimizer.lr}")
    else:
        logger.info("Adapting batch size is done. Batch size isn't changed.")


def _get_batch_size(cfg) -> int:
    if "action" in str(cfg.domain).lower():
        return cfg.data.videos_per_gpu
    return cfg.data.train_dataloader["samples_per_gpu"]


def _set_batch_size(cfg, batch_size: int):
    if "action" in str(cfg.domain).lower():
        cfg.data.videos_per_gpu = batch_size
    else:
        cfg.data.train_dataloader["samples_per_gpu"] = batch_size
        for custom_hook in cfg.custom_hooks:
            if custom_hook["type"] == "AdaptiveRepeatDataHook":
                custom_hook["train_batch_size"] = batch_size


def _set_max_epoch(cfg, max_epoch: int):
    if cfg.runner.get("type") == "AccuracyAwareRunner":  # nncf case
        if "nncf_config" in cfg.runner:
            _set_value_at_dict_in_dict(
                cfg.runner["nncf_config"], "accuracy_aware_training.params.maximal_total_epochs", max_epoch
            )
    else:
        runner_type = cfg.runner.get("type")
        if runner_type is not None and "iterbased" in runner_type.lower():
            cfg.runner["max_iters"] = max_epoch
        else:
            cfg.runner["max_epochs"] = max_epoch


class SubDataset:
    """Wrapper class to make dataset pretend to have specified number of images.

    Args:
        fullset: Original dataset.
        num_samples (int): Number of images to pretend to have. It should be positive.
    """

    def __init__(self, fullset, num_samples: int):
        if num_samples <= 0:
            raise ValueError(f"num_samples should be positive. But, current value is {num_samples}.")

        self.fullset = fullset
        self.num_samples = num_samples
        self._img_indices = {  # for class incremental case
            "old": [i for i in range(num_samples // 2)],
            "new": [i for i in range(num_samples // 2, num_samples)],
        }

    @property
    def img_indices(self):
        """img_indices getter."""
        img_indices = copy(getattr(self.fullset, "img_indices", {}))
        img_indices.update(self._img_indices)
        return img_indices

    def __len__(self) -> int:
        """Get length of subset."""
        return self.num_samples

    def __getitem__(self, indx) -> dict:
        """Get dataset at index."""
        return self.fullset[indx]

    def __getattr__(self, name):
        """When trying to get other attributes, not dataset, get values from fullset."""
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.fullset, name)

    @property
    def flag(self):
        """Getter of flag for detection task.

        Sampler of the detection task decides length of dataset checking sum of flag array.
        To consider that case, return flag array with length of num_samples.

        """
        return np.zeros(self.num_samples, dtype=np.uint8)
