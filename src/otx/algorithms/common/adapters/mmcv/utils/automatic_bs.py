"""Algorithm to find a proper batch size which is fit to current GPU device for tasks using mmcv."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from math import sqrt
from typing import Callable, Dict, List

import numpy as np

from otx.algorithms.common.adapters.torch.utils import BsSearchAlgo
from otx.algorithms.common.utils.logger import get_logger

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


def adapt_batch_size(train_func: Callable, cfg, datasets: List, validate: bool = False, not_increase: bool = True):
    """Decrease batch size if default batch size isn't fit to current GPU device.

    This function just setup for single iteration training to reduce time for adapting.
    The core part of adapting batch size is done in adapt_batch_size in the torch.utils package.

    Args:
        train_func (Callable): The function to train a model.
            Only cfg, dataset and meta are passed to the function when invoking it.
        cfg: Configuration of a training.
        meta (Dict): A dict records some meta information of a training.
        datasets (List): List of datasets.
        validate (bool): Whether do vlidation or not.
        not_increase (bool) : Whether adapting batch size to larger value than default value or not.
    """

    def train_func_single_iter(batch_size):
        copied_cfg = deepcopy(cfg)
        _set_batch_size(copied_cfg, batch_size)
        _set_max_epoch(copied_cfg, 1)  # setup for training a single iter to reduce time

        # Remove hooks due to reasons below
        # OTXProgressHook => prevent progress bar from being 0 and 100 repeatably
        # earlystoppinghook => if eval hook is excluded, this hook makes an error due to absence of score history
        # CustomEvalHook => exclude validation in classification task
        idx_hooks_to_remove = []
        hooks_to_remove = ["OTXProgressHook", "earlystoppinghook", "CustomEvalHook"]
        for i, hook in enumerate(copied_cfg.custom_hooks):
            if not validate and hook["type"] == "AdaptiveTrainSchedulingHook":
                hook["enable_eval_before_run"] = False
            for hook_to_remove in hooks_to_remove:
                if hook_to_remove.lower() in hook["type"].lower():
                    idx_hooks_to_remove.append(i)

        if idx_hooks_to_remove:
            idx_hooks_to_remove.sort()
            for i in reversed(idx_hooks_to_remove):
                del copied_cfg.custom_hooks[i]

        new_datasets = [SubDataset(datasets[0], batch_size)]

        train_func(
            dataset=new_datasets,
            cfg=copied_cfg,
            validate=validate,
        )

    default_bs = _get_batch_size(cfg)

    bs_search_algo = BsSearchAlgo(
        train_func=train_func_single_iter,
        default_bs=default_bs,
        max_bs=len(datasets[0]),
    )
    if not_increase:
        new_batch_size = bs_search_algo.auto_decrease_batch_size()
    else:
        drop_last = cfg.data.get("train_dataloader", {}).get("drop_last", False)
        new_batch_size = bs_search_algo.find_big_enough_batch_size(drop_last)

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
        self.img_indices = {  # for class incremental case
            "old": [i for i in range(num_samples // 2)],
            "new": [i for i in range(num_samples // 2, num_samples)],
        }

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
