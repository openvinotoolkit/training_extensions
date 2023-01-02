# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp

import numpy as np
from mmseg.datasets import DATASETS, CustomDataset

from otx.mpa.modules.utils.task_adapt import map_class_names


@DATASETS.register_module()
class SegIncrCityscapesDataset(CustomDataset):
    """Cityscapes dataset for Class Incremental Learning.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.

    Args:
        classes (list): dataset classes
    """

    def __init__(self, split, classes, new_classes, **kwargs):
        super(SegIncrCityscapesDataset, self).__init__(
            img_suffix="_leftImg8bit.png",
            seg_map_suffix="_gtFine_labelTrainIds.png",
            split=split,
            classes=classes,
            **kwargs
        )
        self.classes = classes
        self.new_classes = new_classes
        self.img_indices = dict(old=[], new=[])
        self.statistics()

    def statistics(self):
        gt_seg_maps = self.get_gt_seg_maps(False)
        classes = ["background"] + self.classes

        new_class_indices = map_class_names(self.new_classes, classes)
        for idx in range(len(gt_seg_maps)):
            gt_map = gt_seg_maps[idx]
            gt_map[np.where((gt_map == 255))] = 0
            gt = np.unique(gt_map)

            label_schema = []
            for i in gt:
                label_schema.append(classes[i])
            model2data = map_class_names(classes, label_schema)
            new_class_values = [model2data[idx] for idx in new_class_indices]
            if any(value is not -1 for value in new_class_values):
                self.img_indices["new"].append(idx)
            else:
                self.img_indices["old"].append(idx)

    def evaluate(self, results, metric="mIoU", logger=None, imgfile_prefix=None, efficient_test=False, show_log=False):
        """Evaluation in Cityscapes/default protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file,
                for cityscapes evaluation only. It includes the file path and
                the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output png files. The output files would be
                png images under folder "a/b/prefix/xxx.png", where "xxx" is
                the image name of cityscapes. If not specified, a temp file
                will be created for evaluation.
                Default: None.

        Returns:
            dict[str, float]: Cityscapes/default metrics.
        """

        eval_results = dict()
        metrics = metric.copy() if isinstance(metric, list) else [metric]
        if "cityscapes" in metrics:
            eval_results.update(self._evaluate_cityscapes(results, logger, imgfile_prefix))
            metrics.remove("cityscapes")

        if len(metrics) > 0:
            eval_results.update(
                super(SegIncrCityscapesDataset, self).evaluate(results, metrics, logger, efficient_test, show_log)
            )

        return eval_results
