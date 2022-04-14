# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import numpy as np

from mmcv.utils.registry import build_from_cfg
from mmcls.datasets.builder import DATASETS, PIPELINES
from mmcls.datasets.pipelines import Compose
from mmcls.datasets.base_dataset import BaseDataset
from mpa.utils.logger import get_logger

logger = get_logger()


@DATASETS.register_module()
class MPAClsDataset(BaseDataset):

    def __init__(self, old_new_indices=None, ote_dataset=None, labels=None, **kwargs):
        self.ote_dataset = ote_dataset
        self.labels = labels
        self.CLASSES = list(label.name for label in labels)
        self.gt_labels = []
        pipeline = kwargs['pipeline']
        self.img_indices = dict(old=[], new=[])
        self.num_classes = len(self.CLASSES)

        if old_new_indices is not None:
            self.img_indices['old'] = old_new_indices['old']
            self.img_indices['new'] = old_new_indices['new']

        if isinstance(pipeline, dict):
            self.pipeline = {}
            for k, v in pipeline.items():
                _pipeline = [dict(type='LoadImageFromOTEDataset'), *v]
                _pipeline = [build_from_cfg(p, PIPELINES) for p in _pipeline]
                self.pipeline[k] = Compose(_pipeline)
            self.num_pipes = len(pipeline)
        elif isinstance(pipeline, list):
            self.num_pipes = 1
            _pipeline = [dict(type='LoadImageFromOTEDataset'), *pipeline]
            self.pipeline = Compose([build_from_cfg(p, PIPELINES) for p in _pipeline])
        self.load_annotations()

    def load_annotations(self):
        for dataset_item in self.ote_dataset:
            if dataset_item.get_annotations() == []:
                label = None
            else:
                label = int(dataset_item.get_annotations()[0].get_labels()[0].id_)

            self.gt_labels.append(label)
        self.gt_labels = np.array(self.gt_labels)

    def __getitem__(self, index):
        dataset_item = self.ote_dataset[index]

        if self.pipeline is None:
            return dataset_item

        results = {}
        results['index'] = index
        results['dataset_item'] = dataset_item
        results['height'], results['width'], _ = dataset_item.numpy.shape
        results['gt_label'] = None if self.gt_labels[index] is None else torch.tensor(self.gt_labels[index])
        results = self.pipeline(results)

        return results

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            list[int]: categories for all images.
        """

        return self.gt_labels

    def __len__(self):
        return len(self.ote_dataset)

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        """Evaluate the dataset with new metric 'class_accuracy'

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
                'accuracy', 'precision', 'recall', 'f1_score', 'support', 'class_accuracy'
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5) if self.num_classes >= 5 else (1, )}

        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric

        if 'class_accuracy' in metrics:
            metrics.remove('class_accuracy')
            self.class_acc = True

        eval_results = super().evaluate(results, metrics, metric_options, logger)

        # Add Evaluation Accuracy score per Class
        if self.class_acc:
            results = np.vstack(results)
            gt_labels = self.get_gt_labels()
            accuracies = self.class_accuracy(results, gt_labels)
            eval_results.update({f'{c} accuracy': a for c, a in zip(self.CLASSES, accuracies)})
            eval_results.update({'mean accuracy': np.mean(accuracies)})

        return eval_results

    def class_accuracy(self, results, gt_labels):
        accracies = []
        pred_label = results.argsort(axis=1)[:, -1:][:, ::-1]
        for i in range(self.num_classes):
            cls_pred = pred_label == i
            cls_pred = cls_pred[gt_labels == i]
            cls_acc = np.sum(cls_pred) / len(cls_pred)
            accracies.append(cls_acc)
        return accracies
