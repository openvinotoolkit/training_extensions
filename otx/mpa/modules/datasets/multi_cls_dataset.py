# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from mmcls.core.evaluation import f1_score, precision, recall
from mmcls.datasets.builder import DATASETS
from mmcls.models.losses import accuracy

from .cls_csv_dataset import CSVDatasetCls


@DATASETS.register_module()
class MultiClsDataset(CSVDatasetCls):
    def __init__(self, tasks=None, **kwargs):
        self.tasks = tasks
        super(MultiClsDataset, self).__init__(**kwargs)

    def load_annotations(self):
        dataframe = self._read_csvs()
        data_infos = []
        for _, data in dataframe.iterrows():
            info = {"img_prefix": self.data_prefix}
            info["img_info"] = {"filename": data["ImagePath"]}
            gt_labels = []
            for task, cls in zip(self.tasks.keys(), self.tasks.values()):
                gt_labels += [cls.index(data[task])]
            info["gt_label"] = np.array(gt_labels, dtype=np.int64)
            data_infos.append(info)
        return data_infos

    def evaluate(self, results, metric="accuracy", metric_options={"topk": (1,)}, logger=None):
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ["accuracy", "precision", "recall", "f1_score", "class_accuracy"]
        eval_results = {}
        if results:
            results = self._refine_results(results)
            for metric in metrics:
                if metric not in allowed_metrics:
                    raise KeyError(f"metric {metric} is not supported.")
                gt_labels = self.get_gt_labels()
                gt_labels = np.transpose(gt_labels)
                for task, gt in zip(self.tasks.keys(), gt_labels):
                    res = results[task]
                    num_imgs = len(res)
                    assert len(gt) == num_imgs
                    if metric == "accuracy":
                        topk = metric_options.get("topk")
                        acc = accuracy(res, gt, topk)
                        eval_result = {f"{task} top-{k}": a.item() for k, a in zip(topk, acc)}
                    elif metric == "precision":
                        precision_value = precision(res, gt)
                        eval_result = {f"{task} precision": precision_value}
                    elif metric == "recall":
                        recall_value = recall(res, gt)
                        eval_result = {f"{task} recall": recall_value}
                    elif metric == "f1_score":
                        f1_score_value = f1_score(res, gt)
                        eval_result = {f"{task} f1_score": f1_score_value}
                    elif metric == "class_accuracy":
                        classes = self.tasks[task]
                        acc = self.class_accuracy(res, gt, classes)
                        eval_result = {f"{task} - {c}": a for c, a in zip(classes, acc)}
                    eval_results.update(eval_result)
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

    def _refine_results(self, results):
        tasks = results[0].keys()
        res_refine = {}
        for task in tasks:
            res_refine[task] = np.concatenate([res[task] for res in results])

        return res_refine
