"""Module for defining a hook for IB loss using mmengine."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from torch.utils.data import Dataset


@HOOKS.register_module()
class IBLossHook(Hook):
    """Hook for IB loss.

    It passes the number of data per class and current epoch to IB loss class.
    """

    def __init__(self, dst_classes: list) -> None:
        """Initialize the IBLossHook.

        Args:
            dst_classes (list): A list of classes including new_classes to be newly learned
        """
        self.cls_num_list = None
        self.dst_classes = dst_classes

    def before_train_epoch(self, runner: Runner) -> None:
        """Get loss from model and pass the number of data per class and current epoch to IB loss."""
        model_loss = self._get_model_loss(runner)
        if runner.epoch == 0:
            dataset = runner.data_loader.dataset
            num_data = self._get_num_data(dataset)
            model_loss.update_weight(num_data)
        model_loss.cur_epoch = runner.epoch

    def _get_num_data(self, dataset: Dataset) -> list:
        return [len(dataset.img_indices[data_cls]) for data_cls in self.dst_classes]

    def _get_model_loss(self, runner: Runner) -> torch.nn.Module:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        return model.head.loss_module
