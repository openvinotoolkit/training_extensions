"""Base MMDataset for Detection Task."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from collections import OrderedDict
from typing import List

import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.builder import DATASETS
from mmrotate.core import eval_rbbox_map, poly2obb_np

from otx.algorithms.detection.adapters.mmdet.datasets.dataset import OTXDetDataset
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.label import Domain, LabelEntity
from otx.api.utils.shape_factory import ShapeFactory


def get_annotation_mmrotate_format(
    dataset_item: DatasetItemEntity,
    labels: List[LabelEntity],
    domain: Domain,
    min_size: int = -1,
    angle_version: str = "oc",
) -> dict:
    """Function to convert OTX annotation to mmrotate format.

    This is used both in the OTXDataset class defined in
    this file as in the custom pipeline element 'LoadAnnotationFromOTXDataset'

    :param dataset_item: DatasetItem for which to get annotations
    :param labels: List of labels that are used in the task
    :param min_size: Minimum size of the bounding box
    :param angle_version: Version of angle to use
    :return dict: annotation information dict in mmdet format
    """
    width, height = dataset_item.width, dataset_item.height

    # load annotations for item
    gt_bboxes = []
    gt_labels = []
    gt_ann_ids = []

    label_idx = {label.id: i for i, label in enumerate(labels)}

    for annotation in dataset_item.get_annotations(labels=labels, include_empty=False, preserve_id=True):
        box = ShapeFactory.shape_as_rectangle(annotation.shape)

        if min(box.width * width, box.height * height) < min_size:
            continue

        class_indices = [
            label_idx[label.id] for label in annotation.get_labels(include_empty=False) if label.domain == domain
        ]

        polygon = ShapeFactory.shape_as_polygon(annotation.shape)
        points = np.array([p for point in polygon.points for p in [point.x * width, point.y * height]])
        points[::2] = np.clip(points[::2], 0, width)
        points[1::2] = np.clip(points[1::2], 0, height)
        points = points.astype(np.uint64)
        obb = poly2obb_np(points, angle_version)
        if obb is not None:
            x, y, w, h, a = obb
            gt_bboxes.append([x, y, w, h, a])
            gt_labels.extend(class_indices)
            item_id = getattr(dataset_item, "id_", None)
            gt_ann_ids.append((item_id, annotation.id_))

    if len(gt_bboxes) > 0:
        ann_info = dict(
            bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 5),
            labels=np.array(gt_labels, dtype=int),
            ann_ids=gt_ann_ids,
        )
    else:
        ann_info = dict(
            bboxes=np.zeros((0, 5), dtype=np.float32),
            labels=np.array([], dtype=int),
            ann_ids=[],
        )
    return ann_info


@DATASETS.register_module()
class OTXRotatedDataset(OTXDetDataset):
    def __init__(self, angle_version: str = "oc", **kwargs):
        """Initialize OTXDataset.

        :param min_size: Minimum size of the bounding box
        :param angle_version: Version of angle to use
        :param kwargs: Additional arguments
        """
        super(OTXRotatedDataset, self).__init__(**kwargs)
        self.angle_version = angle_version

    def get_ann_info(self, idx: int):
        """This method is used for evaluation of predictions.

        The CustomDataset class implements a method
        CustomDataset.evaluate, which uses the class method get_ann_info to retrieve annotations.

        :param idx: index of the dataset item for which to get the annotations
        :return ann_info: dict that contains the coordinates of the bboxes and their corresponding labels
        """
        dataset_item = self.otx_dataset[idx]
        labels = self.labels
        return get_annotation_mmrotate_format(dataset_item, labels, self.domain, angle_version=self.angle_version)

    def evaluate(  # pylint: disable=too-many-branches
        self,
        results,
        metric="mAP",
        logger=None,
        proposal_nums=(100, 300, 1000),
        iou_thr=0.5,
        scale_ranges=None,
    ):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ["mAP"]
        eval_results = OrderedDict()
        for metric in metrics:  # pylint: disable=redefined-argument-from-local
            if metric not in allowed_metrics:
                raise KeyError(f"metric {metric} is not supported")
            annotations = [self.get_ann_info(i) for i in range(len(self))]
            iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
            if metric == "mAP":
                assert isinstance(iou_thrs, list)
                mean_aps = []
                for iou_thr in iou_thrs:  # pylint: disable=redefined-argument-from-local
                    print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                    mean_ap, _ = eval_rbbox_map(
                        results,
                        annotations,
                        scale_ranges=scale_ranges,
                        iou_thr=iou_thr,
                        dataset=self.CLASSES,
                        logger=logger,
                    )
                    mean_aps.append(mean_ap)
                    eval_results[f"AP{int(iou_thr * 100):02d}"] = round(mean_ap, 3)
                eval_results["mAP"] = sum(mean_aps) / len(mean_aps)
        return eval_results
