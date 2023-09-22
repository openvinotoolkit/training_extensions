"""Base Dataset for Classification Task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member

import logging
from typing import Optional, Union

import numpy as np
from mmengine.dataset import Compose
from mmengine.registry import build_from_cfg
from mmpretrain.datasets.base_dataset import BaseDataset
from mmpretrain.datasets.builder import DATASETS
from mmpretrain.evaluation import AveragePrecision, MultiLabelMetric
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from torch.utils.data import Dataset

from otx.v2.adapters.torch.mmengine.mmpretrain.registry import TRANSFORMS
from otx.v2.api.entities.datasets import DatasetEntity
from otx.v2.api.entities.id import ID
from otx.v2.api.utils.data_utils import get_cls_img_indices, get_old_new_img_indices
from otx.v2.api.utils.logger import get_logger

logger = get_logger()


# pylint: disable=too-many-instance-attributes
@DATASETS.register_module()
class OTXClsDataset(BaseDataset):
    """Multi-class classification dataset class."""

    def __init__(
        self,
        otx_dataset: DatasetEntity,
        labels: list,
        empty_label: Optional[list] = None,
        **kwargs,
    ) -> None:  # pylint: disable=super-init-not-called
        self.otx_dataset = otx_dataset
        self.labels = labels
        self.label_names = [label.name for label in self.labels]
        self.label_idx = {label.id: i for i, label in enumerate(labels)}
        self.idx_to_label_id = {v: k for k, v in self.label_idx.items()}
        self.empty_label = empty_label
        self.class_acc = False

        self._metainfo = self._load_metainfo({"classes": [label.name for label in labels]})
        self.gt_labels: list = []
        pipeline = kwargs.get("pipeline", [])
        self.num_classes = len(self.CLASSES)

        test_mode = kwargs.get("test_mode", False)
        if test_mode is False:
            new_classes = kwargs.pop("new_classes", [])
            self.img_indices = self.get_indices(new_classes)

        _pipeline = [{"type": "LoadImageFromOTXDataset"}, *pipeline]
        pipeline_modules = []
        for p in _pipeline:
            if isinstance(p, dict):
                pipeline_modules.append(build_from_cfg(p, TRANSFORMS))
            else:
                pipeline_modules.append(p)
        self.pipeline = Compose(pipeline_modules)
        self.load_annotations()

    def get_indices(self, num_classes: list, *args) -> dict:  # pylint: disable=unused-argument
        """Get indices."""
        return get_cls_img_indices(self.labels, self.otx_dataset)

    def load_annotations(self) -> None:
        """Load annotations."""
        include_empty = self.empty_label in self.labels
        for i, _ in enumerate(self.otx_dataset):
            class_indices = []
            item_labels = self.otx_dataset[i].get_roi_labels(self.labels, include_empty=include_empty)
            ignored_labels = self.otx_dataset[i].ignored_labels
            if item_labels:
                for otx_lbl in item_labels:
                    if otx_lbl not in ignored_labels:
                        class_indices.append(self.label_names.index(otx_lbl.name))
                    else:
                        class_indices.append(-1)
            else:  # this supposed to happen only on inference stage
                class_indices.append(-1)
            self.gt_labels.append(class_indices)
        self.gt_labels = np.array(self.gt_labels)

    def __getitem__(self, index: int) -> dict:
        """Get item from dataset."""
        dataset = self.otx_dataset
        item = dataset[index]
        ignored_labels = np.array([self.label_idx[lbs.id] for lbs in item.ignored_labels])

        height, width = item.height, item.width

        gt_label = self.gt_labels[index]
        data_info = {
            "dataset_item": item,
            "width": width,
            "height": height,
            "index": index,
            "gt_label": gt_label,
            "ignored_labels": ignored_labels,
            "entity_id": getattr(item, "id_", None),
            "label_id": self._get_label_id(gt_label),
        }

        if self.pipeline is None:
            return data_info
        return self.pipeline(data_info)

    def _get_label_id(self, gt_label: np.ndarray) -> Union[ID, list]:
        return self.idx_to_label_id.get(gt_label.item(), ID())

    def get_gt_labels(self) -> np.ndarray:
        """Get all ground-truth labels (categories).

        Returns:
            list[int]: categories for all images.
        """

        return self.gt_labels

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.otx_dataset)

    def evaluate(
        self,
        results: list,
        metric: Union[str, list] = "accuracy",
        metric_options: Optional[dict] = None,
        indices: Optional[list] = None,
        logger: Optional[logging.Logger] = None,
    ) -> dict:  # pylint: disable=redefined-outer-name
        """Evaluate the dataset with new metric class_accuracy.

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
            metric_options = {"topk": (1, 5) if self.num_classes >= 5 else (1,)}

        metrics = [metric] if isinstance(metric, str) else metric

        if "class_accuracy" in metrics:
            metrics.remove("class_accuracy")
            self.class_acc = True

        eval_results = super().evaluate(results, metrics, metric_options, logger=logger)
        for k in metric_options["topk"]:
            eval_results[f"accuracy_top-{k}"] /= 100

        # Add Evaluation Accuracy score per Class - it can be used only for multi-class dataset.
        if self.class_acc:
            results = np.vstack(results)
            gt_labels = self.get_gt_labels()
            accuracies = self.class_accuracy(results, gt_labels)

            if any(np.isnan(accuracies)):
                accuracies = np.nan_to_num(accuracies)

            eval_results.update({f"{c} accuracy": a for c, a in zip(self.CLASSES, accuracies)})
            eval_results.update({"mean accuracy": np.mean(accuracies)})

        eval_results["accuracy"] = eval_results["accuracy_top-1"]
        return eval_results

    def class_accuracy(self, results: np.ndarray, gt_labels: list) -> list:
        """Return per-class accuracy."""
        accracies = []
        pred_label = results.argsort(axis=1)[:, -1:][:, ::-1]
        for i in range(self.num_classes):
            cls_pred = pred_label == i
            cls_pred = cls_pred[gt_labels == i]
            cls_acc = np.sum(cls_pred) / len(cls_pred)
            accracies.append(cls_acc)
        return accracies


@DATASETS.register_module()
class OTXMultilabelClsDataset(OTXClsDataset):
    """Multi-label classification dataset class."""

    def get_indices(self, new_classes: list, *args) -> dict:
        """Get indices."""
        return get_old_new_img_indices(self.labels, new_classes, self.otx_dataset)

    def load_annotations(self) -> None:
        """Load annotations."""
        include_empty = self.empty_label in self.labels
        for i, _ in enumerate(self.otx_dataset):
            item_labels = self.otx_dataset[i].get_roi_labels(self.labels, include_empty=include_empty)
            ignored_labels = self.otx_dataset[i].ignored_labels
            onehot_indices = np.zeros(len(self.labels))
            if item_labels:
                for otx_lbl in item_labels:
                    if otx_lbl not in ignored_labels:
                        onehot_indices[self.label_names.index(otx_lbl.name)] = 1
                    else:
                        # during training we filter ignored classes out,
                        # during validation mmpretrain's mAP also filters -1 labels
                        onehot_indices[self.label_names.index(otx_lbl.name)] = -1

            self.gt_labels.append(onehot_indices)
        self.gt_labels = np.array(self.gt_labels)

    def evaluate(
        self,
        results: list,
        metric: Union[str, list] = "mAP",
        metric_options: Optional[dict] = None,
        indices: Optional[list] = None,
        logger: Optional[logging.Logger] = None,
    ) -> dict:  # pylint: disable=unused-argument, redefined-outer-name, arguments-renamed
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is 'mAP'. Options are 'mAP', 'CP', 'CR', 'CF1',
                'OP', 'OR' and 'OF1'.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'k' and 'thr'. Defaults to None
            indices (list, optional):  Indices to filter the gt label. Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.

        Returns:
            dict: evaluation results
        """
        if metric_options is None or metric_options == {}:
            metric_options = {"thr": 0.5}

        metrics = [metric] if isinstance(metric, str) else metric
        allowed_metrics = ["accuracy-mlc", "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1"]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        len(results)

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f"metric {invalid_metrics} is not supported.")

        if "accuracy-mlc" in metrics:
            true_label_idx = []
            pred_label_idx = []
            pos_thr = metric_options.get("thr", 0.5)

            true_label = gt_labels == 1
            pred_label = results > pos_thr
            cls_index = [i + 1 for i in range(len(self.labels))]
            for true_lbl, pred_lbl in zip(true_label, pred_label):
                true_lbl_idx = set(true_lbl * cls_index) - {0}  # except empty
                pred_lbl_idx = set(pred_lbl * cls_index) - {0}
                true_label_idx.append(true_lbl_idx)
                pred_label_idx.append(pred_lbl_idx)

            confusion_matrices = []
            for cls_idx in cls_index:
                group_labels_idx = {cls_idx - 1}
                y_true = [int(not group_labels_idx.issubset(true_labels)) for true_labels in true_label_idx]
                y_pred = [int(not group_labels_idx.issubset(pred_labels)) for pred_labels in pred_label_idx]
                matrix_data = sklearn_confusion_matrix(y_true, y_pred, labels=list(range(len([0, 1]))))
                confusion_matrices.append(matrix_data)
            correct_per_label_group = [np.trace(mat) for mat in confusion_matrices]
            total_per_label_group = [np.sum(mat) for mat in confusion_matrices]

            acc = np.sum(correct_per_label_group) / np.sum(total_per_label_group)  # MICRO average
            eval_results["accuracy-mlc"] = acc
            eval_results["accuracy"] = eval_results["accuracy-mlc"]

        if "mAP" in metrics:
            map_value = AveragePrecision.calculate(results, gt_labels)
            eval_results["mAP"] = map_value
        if len(set(metrics) - {"mAP"}) != 0:
            performance_keys = ["CP", "CR", "CF1", "OP", "OR", "OF1"]
            performance_values = MultiLabelMetric.calculate(results, gt_labels, **metric_options)
            for k, v in zip(performance_keys, performance_values):
                if k in metrics:
                    eval_results[k] = v

        return eval_results

    def _get_label_id(self, gt_label: np.ndarray) -> Union[ID, list]:
        return [self.idx_to_label_id.get(idx, ID()) for idx, v in enumerate(gt_label) if v == 1]


@DATASETS.register_module()
class OTXHierarchicalClsDataset(OTXMultilabelClsDataset):
    """Hierarchical classification dataset class."""

    def __init__(self, **kwargs) -> None:
        self.hierarchical_info = kwargs.pop("hierarchical_info", {})
        super().__init__(**kwargs)

    def load_annotations(self) -> None:
        """Load annotations."""
        include_empty = self.empty_label in self.labels
        for i, _ in enumerate(self.otx_dataset):
            class_indices = []
            item_labels = self.otx_dataset[i].get_roi_labels(self.labels, include_empty=include_empty)
            ignored_labels = self.otx_dataset[i].ignored_labels
            num_cls_heads = self.hierarchical_info.get("num_multiclass_heads", 0)
            num_ml_classes = self.hierarchical_info.get("num_multilabel_classes", 0)
            if item_labels:
                class_indices = [0] * (num_cls_heads + num_ml_classes)
                for j in range(num_cls_heads):
                    class_indices[j] = -1
                class_to_group_idx = self.hierarchical_info.get("class_to_group_idx", {})
                for otx_lbl in item_labels:
                    group_idx, in_group_idx = class_to_group_idx.get(otx_lbl.name)
                    if group_idx < num_cls_heads:
                        class_indices[group_idx] = in_group_idx
                    elif otx_lbl not in ignored_labels:
                        class_indices[num_cls_heads + in_group_idx] = 1
                    else:
                        class_indices[num_cls_heads + in_group_idx] = -1
            else:  # this supposed to happen only on inference stage or if we have a negative in multilabel data
                class_indices = [-1] * (num_cls_heads + num_ml_classes)
            self.gt_labels.append(class_indices)
        self.gt_labels = np.array(self.gt_labels)

    @staticmethod
    def mean_top_k_accuracy(scores: float, labels: list, k: int = 1) -> np.ndarray:
        """Return mean of top-k accuracy."""
        idx = np.argsort(-scores, axis=-1)[:, :k]
        np_labels = np.array(labels)
        matches = np.any(idx == np_labels.reshape([-1, 1]), axis=-1)

        classes = np.unique(np_labels)

        accuracy_values = []
        for class_id in classes:
            mask = np_labels == class_id
            num_valid = np.sum(mask)
            if num_valid == 0:
                continue

            accuracy_values.append(np.sum(matches[mask]) / float(num_valid))

        return np.mean(accuracy_values) * 100 if len(accuracy_values) > 0 else 1.0

    def evaluate(
        self,
        results: list,
        metric: Union[str, list] = "MHAcc",
        metric_options: Optional[dict] = None,
        indices: Optional[list] = None,
        logger: Optional[logging.Logger] = None,
    ) -> dict:  # pylint: disable=unused-argument, redefined-outer-name
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is 'mAP'. Options are 'mAP', 'CP', 'CR', 'CF1',
                'OP', 'OR' and 'OF1'.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'k' and 'thr'. Defaults to None
            indices (list, optional):  Indices to filter the gt label.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.

        Returns:
            dict: evaluation results
        """
        if metric_options is None or metric_options == {}:
            metric_options = {"thr": 0.5}

        metrics = [metric] if isinstance(metric, str) else metric

        allowed_metrics = ["MHAcc", "avgClsAcc", "mAP"]
        eval_results = {}
        np_results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f"metric {invalid_metrics} is not supported.")

        total_acc = 0.0
        total_acc_sl = 0.0
        num_multiclass_heads = self.hierarchical_info.get("num_multiclass_heads")
        for i in range(num_multiclass_heads):
            multiclass_logit = np_results[
                :,
                self.hierarchical_info["head_idx_to_logits_range"][str(i)][0] : self.hierarchical_info[
                    "head_idx_to_logits_range"
                ][str(i)][1],
            ]
            multiclass_gt = gt_labels[:, i]
            cls_acc = self.mean_top_k_accuracy(multiclass_logit, multiclass_gt, k=1)
            total_acc += cls_acc
            total_acc_sl += cls_acc

        map_value = 0.0
        if num_multiclass_heads and "mAP" in metrics:
            multilabel_logits = np_results[:, self.hierarchical_info["num_single_label_classes"] :]
            multilabel_gt = gt_labels[:, num_multiclass_heads:]
            map_value = AveragePrecision.calculate(multilabel_logits, multilabel_gt)

        total_acc += map_value
        total_acc /= num_multiclass_heads + int(self.hierarchical_info["num_multilabel_classes"] > 0)

        eval_results["MHAcc"] = total_acc
        eval_results["avgClsAcc"] = total_acc_sl / num_multiclass_heads
        eval_results["mAP"] = map_value
        eval_results["accuracy"] = total_acc

        return eval_results


@DATASETS.register_module()
class SelfSLDataset(Dataset):
    """SelfSL dataset that training with two pipelines and no label."""

    CLASSES = None

    def __init__(self, otx_dataset: DatasetEntity, pipeline: dict, **kwargs) -> None:  # pylint: disable=unused-argument
        super().__init__()
        self.otx_dataset = otx_dataset

        self.load_pipeline = build_from_cfg({"type": "LoadImageFromOTXDataset"}, TRANSFORMS)
        self.view0 = Compose([build_from_cfg(p, TRANSFORMS) for p in pipeline["view0"]])
        self.view1 = Compose([build_from_cfg(p, TRANSFORMS) for p in pipeline["view1"]])

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.otx_dataset)

    def __getitem__(self, index: int) -> dict:
        """Get item from dataset."""
        dataset = self.otx_dataset
        item = dataset[index]

        height, width = item.height, item.width

        data_info = {
            "dataset_item": item,
            "width": width,
            "height": height,
            "index": index,
        }

        loaded_results = self.load_pipeline(data_info)
        results1 = self.view0(loaded_results.copy())
        results2 = self.view1(loaded_results.copy())

        results = {}
        for k, v in results1.items():
            results[k + "1"] = v
        for k, v in results2.items():
            results[k + "2"] = v

        return results
