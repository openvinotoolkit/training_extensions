# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import HEADS, build_loss

from .multi_classifier_head import MultiClsHead


@HEADS.register_module()
class TaskIncLwfHead(MultiClsHead):
    def __init__(self, old_tasks=None, distillation_loss=dict(type="LwfLoss", T=2.0, loss_weight=1.0), **kwargs):
        super(TaskIncLwfHead, self).__init__(**kwargs)
        self.old_tasks = old_tasks
        if self.old_tasks is not None:  # old_tasks=None and do this by_han_5
            if len(self.old_tasks) <= 0:
                raise ValueError("at least one task must be exist.")
            self._init_old_layers()

        self.compute_dist_loss = build_loss(distillation_loss)

    def _init_old_layers(self):
        for t_key in self.old_tasks:
            cls = self.old_tasks[t_key]
            if t_key in self.tasks.keys():
                raise Warning(f"existing task {t_key} is overwritten with new {t_key} task")
            else:
                if len(cls) == 0:
                    raise ValueError(f"task={t_key} is empty task.")
                self.classifiers[t_key] = nn.Linear(self.in_channels, len(cls))

    def extract_prob(self, img):
        """Test without augmentation."""
        preds = dict()
        for t_key in self.classifiers:
            cls_score = self.classifiers[t_key](img)
            preds[t_key] = cls_score.detach().cpu().numpy()
        return preds

    def forward_train(self, x, gt_labels, soft_labels):
        cls_scores = dict()
        for t_key in self.classifiers:
            cls_scores[t_key] = self.classifiers[t_key](x)
        losses = self.loss(cls_scores, gt_labels, soft_labels)
        return losses

    def loss(self, cls_scores, gt_labels, soft_labels):
        losses = dict()
        cls_loss = 0.0
        dist_loss = 0.0
        gt_labels = gt_labels.t()
        losses["accuracy"] = {}
        for gt_label, t_key in zip(gt_labels, self.tasks):
            num_samples = len(cls_scores[t_key])
            cls_score = cls_scores[t_key]

            # compute loss
            cls_loss += self.compute_loss(cls_score, gt_label.contiguous(), avg_factor=num_samples)
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses["accuracy"].update({f"{t_key} top-{k}": a for k, a in zip(self.topk, acc)})

        for t_key in self.old_tasks:
            num_samples = len(cls_scores[t_key])
            cls_score = cls_scores[t_key]
            soft_label = soft_labels[t_key]
            if not torch.is_tensor(soft_label):
                soft_label = torch.from_numpy(soft_label)
                if torch.cuda.is_available():
                    soft_label = soft_label.cuda()
            # compute loss
            dist_loss += self.compute_dist_loss(cls_score, soft_label, avg_factor=num_samples)

        losses["new_loss"] = cls_loss / len(self.tasks)
        losses["old_loss"] = dist_loss / len(self.old_tasks)
        return losses

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
        if self.old_tasks is not None:  # by_han_5.5 # related to old_tasks=None and do this by_han_5
            for t_key in self.old_tasks:
                cls_score = self.classifiers[t_key](img)
                if isinstance(cls_score, list):
                    cls_score = sum(cls_score) / float(len(cls_score))
                pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
                if torch.onnx.is_in_onnx_export():
                    preds.append(pred)
                else:
                    preds[t_key] = pred.detach().cpu().numpy()
        return preds


# @HEADS.register_module()
# class TaskIncFinetuneHead(MultiClsHead):
#     def __init__(self, old_tasks, **kwargs):
#         super(TaskIncFinetuneHead, self).__init__(**kwargs)
#         self.old_tasks = old_tasks
#         if len(self.old_tasks) <= 0:
#             raise ValueError('at least one task must be exist.')
#         self._init_old_layers()
#
#     def _init_old_layers(self):
#         for t_key in self.old_tasks:
#             cls = self.old_tasks[t_key]
#             if len(cls) == 0:
#                 raise ValueError(
#                     f'task={t_key} is empty task.')
#             self.classifiers[t_key] = nn.Linear(self.in_channels, len(cls))
#             self.classifiers[t_key].eval()
#             for param in self.classifiers[t_key].parameters():
#                 param.requires_grad = False
#
#     def simple_test(self, img):
#         """Test without augmentation."""
#         if torch.onnx.is_in_onnx_export():
#             preds = []
#         else:
#             preds = dict()
#
#         for t_key in self.tasks:
#             cls_score = self.classifiers[t_key](img)
#             if isinstance(cls_score, list):
#                 cls_score = sum(cls_score) / float(len(cls_score))
#             pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
#             if torch.onnx.is_in_onnx_export():
#                 preds.append(pred)
#             else:
#                 preds[t_key] = pred.detach().cpu().numpy()
#         for t_key in self.old_tasks:
#             cls_score = self.classifiers[t_key](img)
#             if isinstance(cls_score, list):
#                 cls_score = sum(cls_score) / float(len(cls_score))
#             pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
#             if torch.onnx.is_in_onnx_export():
#                 preds.append(pred)
#             else:
#                 preds[t_key] = pred.detach().cpu().numpy()
#         return preds
