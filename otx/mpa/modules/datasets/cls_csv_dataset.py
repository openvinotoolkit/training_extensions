# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pandas as pd
from mmcls.core.evaluation import precision_recall_f1, support
from mmcls.datasets.base_dataset import BaseDataset
from mmcls.datasets.builder import DATASETS
from mmcls.models.losses import accuracy


@DATASETS.register_module()
class CSVDatasetCls(BaseDataset):
    def __init__(self, data_file, pipeline=[], classes=None, **kwargs):
        self.data_file = data_file
        self.CLASSES = self.get_classes(classes)
        _pipeline = [dict(type="LoadImageFromFile"), *pipeline]
        super(CSVDatasetCls, self).__init__(classes=classes, pipeline=_pipeline, **kwargs)

    def _read_csvs(self):
        if isinstance(self.data_file, (list, tuple)):
            data_file_list = self.data_file
        else:
            data_file_list = [self.data_file]

        try:
            dataframe = None
            for data_file in data_file_list:
                _df_data = pd.read_csv(data_file)
                dataframe = pd.concat([dataframe, _df_data], ignore_index=True)

            if self.ann_file:
                if isinstance(self.ann_file, (list, tuple)):
                    ann_file_list = self.ann_file
                else:
                    ann_file_list = [self.ann_file]

                df_anno = None
                for ann_file in ann_file_list:
                    _df_anno = pd.read_csv(ann_file)
                    df_anno = pd.concat([df_anno, _df_anno], ignore_index=True)

                dataframe = pd.merge(dataframe, df_anno, left_on="ImageID", right_on="ImageID", how="inner")
        except pd.errors.EmptyDataError:
            raise ValueError("The csv file is empty.")

        return dataframe

    def load_annotations(self):
        dataframe = self._read_csvs()
        data_infos = []
        for _, data in dataframe.iterrows():
            info = {"img_prefix": self.data_prefix}
            info["img_info"] = {"filename": data["ImagePath"]}
            if "Status" in data.index:
                info["gt_label"] = np.array(self.class_to_idx[data["Status"]])
            data_infos.append(info)
        return data_infos

    def evaluate(self, results, metric="accuracy", metric_options=None, logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {"topk": (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ["accuracy", "precision", "recall", "f1_score", "support", "class_accuracy"]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, "dataset testing results should " "be of the same length as gt_labels."

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f"metirc {invalid_metrics} is not supported.")

        topk = metric_options.get("topk", (1, 5))
        thrs = metric_options.get("thrs")
        average_mode = metric_options.get("average_mode", "macro")

        if "accuracy" in metrics:
            acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            if isinstance(topk, tuple):
                eval_results_ = {f"accuracy_top-{k}": a for k, a in zip(topk, acc)}
            else:
                eval_results_ = {"accuracy": acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({f"{key}_thr_{thr:.2f}": value.item() for thr, value in zip(thrs, values)})
            else:
                eval_results.update({k: v.item() for k, v in eval_results_.items()})

        if "support" in metrics:
            support_value = support(results, gt_labels, average_mode=average_mode)
            eval_results["support"] = support_value

        precision_recall_f1_keys = ["precision", "recall", "f1_score"]
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            precision_recall_f1_values = precision_recall_f1(results, gt_labels, average_mode=average_mode, thrs=thrs)
            for key, values in zip(precision_recall_f1_keys, precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({f"{key}_thr_{thr:.2f}": value for thr, value in zip(thrs, values)})
                    else:
                        eval_results[key] = values
        if "class_accuracy" in metrics:
            accuracies = self.class_accuracy(results, gt_labels, self.CLASSES)
            eval_results.update({f"{c} accuracy - ": a for c, a in zip(self.CLASSES, accuracies)})
            eval_results.update({"mean accuracy": np.mean(accuracies)})

        return eval_results

    def class_accuracy(self, res, gt, classes):
        num_cls = len(classes)
        accracies = []
        pred_label = res.argsort(axis=1)[:, -1:][:, ::-1]
        for i in range(num_cls):
            cls_pred = pred_label == i
            cls_pred = cls_pred[gt == i]
            cls_acc = np.sum(cls_pred) / len(cls_pred)
            accracies.append(cls_acc)
        return accracies
