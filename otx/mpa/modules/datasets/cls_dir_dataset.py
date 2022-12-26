# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import copy
import numpy as np

from mmcv.utils.registry import build_from_cfg
from mmcls.datasets.builder import DATASETS, PIPELINES
from mmcls.datasets.base_dataset import BaseDataset
from mmcls.datasets.pipelines import Compose

from otx.mpa.utils.logger import get_logger

logger = get_logger()


@DATASETS.register_module()
class ClsDirDataset(BaseDataset):
    """Classification dataset for Finetune, Incremental Learning, Semi-SL
        assumes the data for classification is divided by folders (classes)

    Args:
        data_dir (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        classes (list): List of classes to be used for training,
            If empty, use all classes in the folder list
            Also, if there is new_classes (incremental learning),
            classes are used as a list of old classes
        new_classes (list): List of final classes to be used for incremental learning
        use_labels (bool): dataset with labels or unlabels
    """

    def __init__(self, data_dir, pipeline=[], classes=[], new_classes=[], use_labels=True, **kwargs):
        self.data_dir = data_dir
        self._samples_per_gpu = kwargs.pop('samples_per_gpu', 1)
        self._workers_per_gpu = kwargs.pop('workers_per_gpu', 1)
        self.use_labels = use_labels
        self.img_indices = dict(old=[], new=[])
        self.class_acc = False

        self.new_classes = new_classes
        if not classes:
            self.CLASSES = self.get_classes_from_dir(self.data_dir)
        else:
            self.CLASSES = self.get_classes(classes)
        if isinstance(self.CLASSES, list):
            self.CLASSES.sort()
        self.num_classes = len(self.CLASSES)

        # Pipeline Settings
        if isinstance(pipeline, dict):
            self.pipeline = {}
            for k, v in pipeline.items():
                _pipeline = [dict(type='LoadImageFromFile'), *v]
                _pipeline = [build_from_cfg(p, PIPELINES) for p in _pipeline]
                self.pipeline[k] = Compose(_pipeline)
            self.num_pipes = len(pipeline)
        elif isinstance(pipeline, list):
            self.num_pipes = 1
            _pipeline = [dict(type='LoadImageFromFile'), *pipeline]
            self.pipeline = Compose([build_from_cfg(p, PIPELINES) for p in _pipeline])

        self.data_infos = self.load_annotations()
        self.statistics()

    def statistics(self):
        logger.info(f'ClsDirDataset - {self.num_classes} classes from {self.data_dir}')
        logger.info(f'- Classes: {self.CLASSES}')
        if self.new_classes:
            logger.info(f'- New Classes: {self.new_classes}')
            old_data_length = len(self.img_indices['old'])
            new_data_length = len(self.img_indices['new'])
            logger.info(f'- # of old classes images: {old_data_length}')
            logger.info(f'- # of New classes images: {new_data_length}')
        logger.info(f'- # of images: {len(self)}')

    def _read_dir(self):
        img_path, img_class, img_prefix = [], [], []
        if self.use_labels:
            cls_dir_list = os.listdir(self.data_dir)
            for cls_name in cls_dir_list:
                if cls_name not in self.CLASSES:
                    continue
                cls_dir = os.path.join(self.data_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue

                cls_img_path = os.listdir(cls_dir)
                img_path += cls_img_path
                img_prefix += [cls_dir] * len(cls_img_path)
                img_class += [cls_name] * len(cls_img_path)
        else:
            path_list = os.listdir(self.data_dir)
            for p in path_list:
                current_path = os.path.join(self.data_dir, p)
                if os.path.isdir(current_path):
                    cls_img_path = os.listdir(current_path)
                    img_path += cls_img_path
                    img_prefix += [current_path] * len(cls_img_path)
                    img_class += [self.CLASSES[0]] * len(cls_img_path)
                else:
                    img_path += [p]
                    img_prefix += [self.data_dir]
                    img_class += [self.CLASSES[0]]

        return img_path, img_class, img_prefix

    def load_annotations(self):
        img_path_list, img_class_list, img_prefix_list = self._read_dir()
        data_infos = []
        for i, (img_path, img_cls, img_prefix) in enumerate(zip(img_path_list, img_class_list, img_prefix_list)):
            if self.use_labels:
                gt_label = np.array(self.class_to_idx[img_cls])
            else:
                gt_label = np.array([-1])
            info = {'img_prefix': img_prefix, 'img_info': {'filename': img_path}, 'gt_label': gt_label}
            data_infos.append(info)
            if img_cls in self.new_classes:
                self.img_indices['new'].append(i)
            else:
                self.img_indices['old'].append(i)
        return data_infos

    def get_classes_from_dir(self, root):
        if not self.use_labels:
            return [os.path.basename(root)]
        # classes = self.get_classes(classes)
        # print(classes)
        # if classes is None:
        classes = []
        path_list = os.listdir(root)
        for p in path_list:
            if os.path.isdir(os.path.join(root, p)):
                if p not in classes:
                    classes.append(p)
            else:
                raise ValueError("This folder structure is not suitable for label data")
        return classes

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.pipeline is None:
            return self.data_infos[idx]

        data_infos = [
            copy.deepcopy(self.data_infos[idx]) for _ in range(self.num_pipes)
        ]
        if isinstance(self.pipeline, dict):
            results = {}
            for i, (k, v) in enumerate(self.pipeline.items()):
                results[k] = self.pipeline[k](data_infos[i])
        else:
            results = self.pipeline(data_infos[0])

        return results

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

    @property
    def samples_per_gpu(self):
        return self._samples_per_gpu

    @property
    def workers_per_gpu(self):
        return self._workers_per_gpu
