"""Collection of utils for data in Detection Task."""

# Copyright (C) 2021-2022 Intel Corporation
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

import math
import os.path as osp
from typing import List, Optional

import numpy as np
from mmdet.datasets import CocoDataset

from otx.algorithms.detection.adapters.mmdet.data.dataset import (
    get_annotation_mmdet_format,
)
from otx.algorithms.detection.configs.base.configuration import DetectionConfig
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    DirectoryPathCheck,
    JsonFilePathCheck,
    check_input_parameters_type,
)
from otx.api.utils.shape_factory import ShapeFactory


@check_input_parameters_type()
def find_label_by_name(labels: List[LabelEntity], name: str, domain: Domain):
    """Return label from name."""
    matching_labels = [label for label in labels if label.name == name]
    if len(matching_labels) == 1:
        return matching_labels[0]
    if len(matching_labels) == 0:
        label = LabelEntity(name=name, domain=domain, id=ID(len(labels)))
        labels.append(label)
        return label
    raise ValueError("Found multiple matching labels")


@check_input_parameters_type({"ann_file_path": JsonFilePathCheck, "data_root_dir": DirectoryPathCheck})
def load_dataset_items_coco_format(
    ann_file_path: str,
    data_root_dir: str,
    domain: Domain,
    subset: Subset = Subset.NONE,
    labels_list: Optional[List[LabelEntity]] = None,
    with_mask: bool = False,
):  # pylint: disable=too-many-locals
    """Load dataset from CocoDataset."""
    test_mode = subset in {Subset.VALIDATION, Subset.TESTING}

    coco_dataset = CocoDataset(
        ann_file=ann_file_path,
        data_root=data_root_dir,
        classes=None,
        test_mode=test_mode,
        with_mask=with_mask,
    )
    coco_dataset.test_mode = False
    for label_name in coco_dataset.classes:
        find_label_by_name(labels_list, label_name, domain)

    dataset_items = []
    for item in coco_dataset:

        def create_gt_box(x1, y1, x2, y2, label_name):
            return Annotation(
                Rectangle(x1=x1, y1=y1, x2=x2, y2=y2),
                labels=[ScoredLabel(label=find_label_by_name(labels_list, label_name, domain))],
            )

        def create_gt_polygon(polygon_group, label_name):
            if len(polygon_group) != 1:
                raise RuntimeError(
                    "Complex instance segmentation masks consisting of several polygons are not supported."
                )

            return Annotation(
                Polygon(points=polygon_group[0]),
                labels=[ScoredLabel(label=find_label_by_name(labels_list, label_name, domain))],
            )

        img_height = item["img_info"].get("height")
        img_width = item["img_info"].get("width")
        divisor = np.array(
            [img_width, img_height, img_width, img_height],
            dtype=item["gt_bboxes"].dtype,
        )
        bboxes = item["gt_bboxes"] / divisor
        labels = item["gt_labels"]

        assert len(bboxes) == len(labels)
        if with_mask:
            polygons = item["gt_masks"]
            assert len(bboxes) == len(polygons)
            normalized_polygons = []  # type: List
            for polygon_group in polygons:
                normalized_polygons.append([])
                for polygon in polygon_group:
                    normalized_polygon = [p / divisor[i % 2] for i, p in enumerate(polygon)]
                    points = [
                        Point(normalized_polygon[i], normalized_polygon[i + 1]) for i in range(0, len(polygon), 2)
                    ]
                    normalized_polygons[-1].append(points)

        if item["img_prefix"] is not None:
            filename = osp.join(item["img_prefix"], item["img_info"]["filename"])
        else:
            filename = item["img_info"]["filename"]

        if with_mask:
            shapes = [
                create_gt_polygon(polygon_group, coco_dataset.classes[label_id])
                for polygon_group, label_id in zip(normalized_polygons, labels)
            ]
        else:
            shapes = [
                create_gt_box(x1, y1, x2, y2, coco_dataset.classes[label_id])
                for (x1, y1, x2, y2), label_id in zip(bboxes, labels)
            ]

        dataset_item = DatasetItemEntity(
            media=Image(file_path=filename),
            annotation_scene=AnnotationSceneEntity(annotations=shapes, kind=AnnotationSceneKind.ANNOTATION),
            subset=subset,
        )
        dataset_items.append(dataset_item)

    return dataset_items


@check_input_parameters_type({"dataset": DatasetParamTypeCheck})
def get_sizes_from_dataset_entity(dataset: DatasetEntity, target_wh: List[int]):
    """Function to get sizes of instances in DatasetEntity and to resize it to the target size.

    :param dataset: DatasetEntity in which to get statistics
    :param target_wh: target width and height of the dataset
    :return list: tuples with width and height of each instance
    """
    wh_stats = []
    for item in dataset:
        for ann in item.get_annotations(include_empty=False):
            has_detection_labels = any(
                label.domain == Domain.DETECTION for label in ann.get_labels(include_empty=False)
            )
            if has_detection_labels:
                box = ShapeFactory.shape_as_rectangle(ann.shape)
                width = box.width * target_wh[0]
                height = box.height * target_wh[1]
                wh_stats.append((width, height))
    return wh_stats


@check_input_parameters_type()
def get_anchor_boxes(wh_stats: List[tuple], group_as: List[int]):
    """Get anchor box widths & heights."""
    from sklearn.cluster import KMeans

    kmeans = KMeans(init="k-means++", n_clusters=sum(group_as), random_state=0).fit(wh_stats)
    centers = kmeans.cluster_centers_

    areas = np.sqrt(np.prod(centers, axis=1))
    idx = np.argsort(areas)

    widths = centers[idx, 0]
    heights = centers[idx, 1]

    group_as = np.cumsum(group_as[:-1])
    widths, heights = np.split(widths, group_as), np.split(heights, group_as)
    widths = [width.tolist() for width in widths]
    heights = [height.tolist() for height in heights]
    return widths, heights


@check_input_parameters_type()
def format_list_to_str(value_lists: list):
    """Decrease floating point digits in logs."""
    str_value = ""
    for value_list in value_lists:
        str_value += "[" + ", ".join(f"{value:.2f}" for value in value_list) + "], "
    return f"[{str_value[:-2]}]"


# TODO [Eugene] please add unit test for this function
def adaptive_tile_params(
    tiling_parameters: DetectionConfig.BaseTilingParameters, dataset: DatasetEntity, object_tile_ratio=0.01, rule="avg"
):
    """Config tile parameters.

    Adapt based on annotation statistics.
    i.e. tile size, tile overlap, ratio and max objects per sample

    Args:
        tiling_parameters (BaseTilingParameters): tiling parameters of the model
        dataset (DatasetEntity): training dataset
        object_tile_ratio (float, optional): The desired ratio of object area and tile area. Defaults to 0.01.
        rule (str, optional): min or avg.  In `min` mode, tile size is computed based on the smallest object, and in
                              `avg` mode tile size is computed by averaging all the object areas. Defaults to "avg".

    """
    assert rule in ["min", "avg"], f"Unknown rule: {rule}"

    bboxes = np.zeros((0, 4), dtype=np.float32)
    labels = dataset.get_labels(include_empty=False)
    domain = labels[0].domain
    max_object = 0
    for dataset_item in dataset:
        result = get_annotation_mmdet_format(dataset_item, labels, domain)
        if len(result["bboxes"]):
            bboxes = np.concatenate((bboxes, result["bboxes"]), 0)
            if len(result["bboxes"]) > max_object:
                max_object = len(result["bboxes"])

    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    if rule == "min":
        object_area = np.min(areas)
    elif rule == "avg":
        object_area = np.mean(areas)
    max_area = np.max(areas)

    tile_size = int(math.sqrt(object_area / object_tile_ratio))
    tile_overlap = max_area / (tile_size**2)

    # validate parameters are in range
    tile_size = max(
        tiling_parameters.get_metadata("tile_size")["min_value"],
        min(tiling_parameters.get_metadata("tile_size")["max_value"], tile_size),
    )
    tile_overlap = max(
        tiling_parameters.get_metadata("tile_overlap")["min_value"],
        min(tiling_parameters.get_metadata("tile_overlap")["max_value"], tile_overlap),
    )
    max_object = max(
        tiling_parameters.get_metadata("tile_max_number")["min_value"],
        min(tiling_parameters.get_metadata("tile_max_number")["max_value"], max_object),
    )

    tiling_parameters.tile_size = tile_size
    tiling_parameters.tile_max_number = max_object
    tiling_parameters.tile_overlap = tile_overlap
