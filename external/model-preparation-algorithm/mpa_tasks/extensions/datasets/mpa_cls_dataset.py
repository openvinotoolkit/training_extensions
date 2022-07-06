# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from mmcv.utils.registry import build_from_cfg
from mmcls.core import average_performance, mAP
from mmcls.datasets.builder import DATASETS, PIPELINES
from mmcls.datasets.pipelines import Compose
from mmcls.datasets.base_dataset import BaseDataset
from mpa_tasks.utils.data_utils import get_cls_img_indices, get_old_new_img_indices
from mpa.utils.logger import get_logger

logger = get_logger()


@DATASETS.register_module()
class MPAClsDataset(BaseDataset):

    def __init__(self, ote_dataset=None, labels=None, **kwargs):
        self.ote_dataset = ote_dataset
        self.labels = labels
        self.label_idx = {label.id: i for i, label in enumerate(labels)}
        self.CLASSES = list(label.name for label in labels)
        self.gt_labels = []
        pipeline = kwargs['pipeline']
        self.num_classes = len(self.CLASSES)

        test_mode = kwargs.get('test_mode', False)
        if test_mode is False:
            new_classes = kwargs.pop('new_classes', [])
            self.img_indices = self.get_indices(new_classes)

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

    def get_indices(self, *args):
        return get_cls_img_indices(self.labels, self.ote_dataset)

    def load_annotations(self):
        for dataset_item in self.ote_dataset:
            roi_label = dataset_item.get_roi_labels(self.labels)
            label = [self.label_idx[lbl.id] for lbl in roi_label] if roi_label else [None]
            self.gt_labels.append(label)
        self.gt_labels = np.array(self.gt_labels)

    def __getitem__(self, index):
        dataset = self.ote_dataset
        item = dataset[index]
        ignored_labels = np.array([self.label_idx[lbs.id] for lbs in item.ignored_labels])

        height, width = item.height, item.width

        data_info = dict(dataset_item=item, width=width, height=height, index=index,
                         gt_label=self.gt_labels[index], ignored_labels=ignored_labels)

        if self.pipeline is None:
            return data_info
        else:
            return self.pipeline(data_info)

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

        # Add Evaluation Accuracy score per Class - it can be used only for multi-class dataset.
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


@DATASETS.register_module()
class MPAMultilabelClsDataset(MPAClsDataset):
    def get_indices(self, new_classes):
        return get_old_new_img_indices(self.labels, new_classes, self.ote_dataset)

    def load_annotations(self):
        for dataset_item in self.ote_dataset:
            label = np.zeros(len(self.labels))
            roi_label = dataset_item.get_roi_labels(self.labels)
            for lbl in roi_label:
                label[self.label_idx[lbl.id]] = 1
            self.gt_labels.append(label)
        self.gt_labels = np.array(self.gt_labels)

    def evaluate(self,
                 results,
                 metric='mAP',
                 metric_options=None,
                 indices=None,
                 logger=None):
        """Evaluate the dataset.
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is 'mAP'. Options are 'mAP', 'CP', 'CR', 'CF1',
                'OP', 'OR' and 'OF1'.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'k' and 'thr'. Defaults to None
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None or metric_options == {}:
            metric_options = {'thr': 0.5}

        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ['accuracy-mlc', 'mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        if 'accuracy-mlc' in metrics:
            true_label_idx = []
            pred_label_idx = []
            pos_thr = metric_options.get('thr', 0.5)

            true_label = (gt_labels == 1)
            pred_label = (results > pos_thr)
            cls_index = [i+1 for i in range(len(self.labels))]
            for true_lbl, pred_lbl in zip(true_label, pred_label):
                true_lbl_idx = set(true_lbl * cls_index) - set([0])  # except empty
                pred_lbl_idx = set(pred_lbl * cls_index) - set([0])
                true_label_idx.append(true_lbl_idx)
                pred_label_idx.append(pred_lbl_idx)

            confusion_matrices = []
            for cls_idx in cls_index:
                group_labels_idx = set([cls_idx-1])
                y_true = [int(not group_labels_idx.issubset(true_labels))
                          for true_labels in true_label_idx]
                y_pred = [int(not group_labels_idx.issubset(pred_labels))
                          for pred_labels in pred_label_idx]
                matrix_data = sklearn_confusion_matrix(y_true, y_pred, labels=list(range(len([0, 1]))))
                confusion_matrices.append(matrix_data)

            correct_per_label_group = [
                np.trace(mat) for mat in confusion_matrices
            ]
            total_per_label_group = [
                np.sum(mat) for mat in confusion_matrices
            ]

            acc = np.sum(correct_per_label_group) / np.sum(total_per_label_group)  # MICRO average
            eval_results['accuracy-mlc'] = acc

        if 'mAP' in metrics:
            mAP_value = mAP(results, gt_labels)
            eval_results['mAP'] = mAP_value
        if len(set(metrics) - {'mAP'}) != 0:
            performance_keys = ['CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
            performance_values = average_performance(results, gt_labels,
                                                     **metric_options)
            for k, v in zip(performance_keys, performance_values):
                if k in metrics:
                    eval_results[k] = v

        return eval_results

@DATASETS.register_module()
class MPAHierarchicalClsDataset(MPAMultilabelClsDataset):
    def __init__(self, **kwargs):
        self.hierarchical_class_info = kwargs.pop('hierarchical_class_info', None)
        super().__init__(**kwargs)

    def evaluate(self):
        raise NotImplementedError