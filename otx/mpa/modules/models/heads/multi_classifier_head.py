# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import HEADS
from mmcls.models.heads.cls_head import ClsHead
from mmcv.cnn import normal_init


@HEADS.register_module()
class MultiClsHead(ClsHead):
    def __init__(
        self, tasks, in_channels, loss=dict(type="CrossEntropyLoss", loss_weight=1.0), topk=(1,), num_classes=None
    ):
        super(MultiClsHead, self).__init__(loss=loss, topk=topk)
        self.in_channels = in_channels
        self.tasks = tasks
        if not isinstance(self.tasks, dict):
            raise TypeError("tasks type must be dict")
        if len(self.tasks) <= 0:
            raise ValueError("at least one task must be exist.")

        self._init_layers()

    def _init_layers(self):
        self.classifiers = torch.nn.ModuleDict()
        for t_key in self.tasks:
            cls = self.tasks[t_key]
            if len(cls) == 0:
                raise ValueError(f"task={t_key} is empty task.")
            self.classifiers[t_key] = nn.Linear(self.in_channels, len(cls))

    def init_weights(self):
        for t_key in self.tasks:
            normal_init(self.classifiers[t_key], mean=0, std=0.01, bias=0)

    def simple_test(self, img):
        """Test without augmentation."""
        if torch.onnx.is_in_onnx_export():
            preds = []
        else:
            preds = dict()

        for t_key in self.tasks:
            cls_score = self.classifiers[t_key](img)
            if isinstance(cls_score, list):
                cls_score = sum(cls_score) / float(len(cls_score))
            pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
            if torch.onnx.is_in_onnx_export():
                preds.append(pred)
            else:
                preds[t_key] = pred.detach().cpu().numpy()
        # if torch.onnx.is_in_onnx_export():
        #     return preds
        return preds

    def forward_train(self, x, gt_labels):
        cls_scores = []
        for t_key in self.classifiers:
            cls_scores.append(self.classifiers[t_key](x))
        losses = self.loss(cls_scores, gt_labels)
        return losses

    def loss(self, cls_scores, gt_labels):
        losses = dict()
        loss = 0.0
        gt_labels = gt_labels.t()
        losses["accuracy"] = {}
        for cls_score, gt_label, task_name in zip(cls_scores, gt_labels, self.tasks):
            num_samples = len(cls_score)

            # compute loss
            loss += self.compute_loss(cls_score, gt_label.contiguous(), avg_factor=num_samples)
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses["accuracy"].update({f"{task_name} top-{k}": a for k, a in zip(self.topk, acc)})

        losses["loss"] = loss / len(cls_scores)
        return losses
