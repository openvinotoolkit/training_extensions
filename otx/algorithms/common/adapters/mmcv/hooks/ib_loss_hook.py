"""Module for defining a hook for IB loss using mmcls."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class IBLossHook(Hook):
    """Hook for IB loss.

    It passes the number of data per class and current epoch to IB loss class.
    """

    def __init__(self, dst_classes):
        """Initialize the IBLossHook.

        Args:
            dst_classes (list): A list of classes including new_classes to be newly learned
        """
        self.cls_num_list = None
        self.dst_classes = dst_classes

    def before_train_epoch(self, runner):
        """Get loss from model and pass the number of data per class and current epoch to IB loss."""
        model_loss = self._get_model_loss(runner)
        if runner.epoch == 0:
            dataset = runner.data_loader.dataset
            num_data = self._get_num_data(dataset)
            model_loss.update_weight(num_data)
        model_loss.cur_epoch = runner.epoch

    def _get_num_data(self, dataset):
        return [len(dataset.img_indices[data_cls]) for data_cls in self.dst_classes]

    def _get_model_loss(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        return model.head.compute_loss
