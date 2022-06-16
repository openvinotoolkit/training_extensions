# Copyright (C) 2021 Intel Corporation
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

from copy import deepcopy
from typing import Any, Dict, List, Sequence

import numpy as np
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)
from ote_sdk.utils.shape_factory import ShapeFactory

from mmdet.core import PolygonMasks
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.pipelines import Compose


@check_input_parameters_type()
def get_annotation_mmdet_format(
    dataset_item: DatasetItemEntity,
    labels: List[LabelEntity],
    domain: Domain,
    min_size: int = -1,
) -> dict:
    """
    Function to convert a OTE annotation to mmdetection format. This is used both in the OTEDataset class defined in
    this file as in the custom pipeline element 'LoadAnnotationFromOTEDataset'

    :param dataset_item: DatasetItem for which to get annotations
    :param labels: List of labels that are used in the task
    :return dict: annotation information dict in mmdet format
    """
    width, height = dataset_item.width, dataset_item.height

    # load annotations for item
    gt_bboxes = []
    gt_labels = []
    gt_polygons = []

    label_idx = {label.id: i for i, label in enumerate(labels)}

    for annotation in dataset_item.get_annotations(labels=labels, include_empty=False):

        box = ShapeFactory.shape_as_rectangle(annotation.shape)

        if min(box.width * width, box.height * height) < min_size:
            continue

        class_indices = [
            label_idx[label.id]
            for label in annotation.get_labels(include_empty=False)
            if label.domain == domain
        ]

        n = len(class_indices)
        gt_bboxes.extend([[box.x1 * width, box.y1 * height, box.x2 * width, box.y2 * height] for _ in range(n)])
        if domain != Domain.DETECTION:
            polygon = ShapeFactory.shape_as_polygon(annotation.shape)
            polygon = np.array([p for point in polygon.points for p in [point.x * width, point.y * height]])
            gt_polygons.extend([[polygon] for _ in range(n)])
        gt_labels.extend(class_indices)

    if len(gt_bboxes) > 0:
        ann_info = dict(
            bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
            labels=np.array(gt_labels, dtype=int),
            masks=PolygonMasks(
                gt_polygons, height=height, width=width) if gt_polygons else [])
    else:
        ann_info = dict(
            bboxes=np.zeros((0, 4), dtype=np.float32),
            labels=np.array([], dtype=int),
            masks=[])
    return ann_info


@DATASETS.register_module()
class OTEDataset(CustomDataset):
    """
    Wrapper that allows using a OTE dataset to train mmdetection models. This wrapper is not based on the filesystem,
    but instead loads the items here directly from the OTE DatasetEntity object.

    The wrapper overwrites some methods of the CustomDataset class: prepare_train_img, prepare_test_img and prepipeline
    Naming of certain attributes might seem a bit peculiar but this is due to the conventions set in CustomDataset. For
    instance, CustomDatasets expects the dataset items to be stored in the attribute data_infos, which is why it is
    named like that and not dataset_items.

    """

    class _DataInfoProxy:
        """
        This class is intended to be a wrapper to use it in CustomDataset-derived class as `self.data_infos`.
        Instead of using list `data_infos` as in CustomDataset, our implementation of dataset OTEDataset
        uses this proxy class with overriden __len__ and __getitem__; this proxy class
        forwards data access operations to ote_dataset and converts the dataset items to the view
        convenient for mmdetection.
        """
        def __init__(self, ote_dataset, labels):
            self.ote_dataset = ote_dataset
            self.labels = labels
            self.label_idx = {label.id: i for i, label in enumerate(labels)}

        def __len__(self):
            return len(self.ote_dataset)

        def __getitem__(self, index):
            """
            Prepare a dict 'data_info' that is expected by the mmdet pipeline to handle images and annotations
            :return data_info: dictionary that contains the image and image metadata, as well as the labels of
            the objects in the image
            """

            dataset = self.ote_dataset
            item = dataset[index]
            ignored_labels = np.array([self.label_idx[lbs.id] for lbs in item.ignored_labels])

            height, width = item.height, item.width

            data_info = dict(dataset_item=item, width=width, height=height, index=index,
                             ann_info=dict(label_list=self.labels), ignored_labels=ignored_labels)

            return data_info

    @check_input_parameters_type({"ote_dataset": DatasetParamTypeCheck})
    def __init__(
            self,
            ote_dataset: DatasetEntity,
            labels: List[LabelEntity],
            pipeline: Sequence[dict],
            domain: Domain,
            test_mode: bool = False,
    ):
        self.ote_dataset = ote_dataset
        self.labels = labels
        self.CLASSES = list(label.name for label in labels)
        self.domain = domain
        self.test_mode = test_mode

        # Instead of using list data_infos as in CustomDataset, this implementation of dataset
        # uses a proxy class with overriden __len__ and __getitem__; this proxy class
        # forwards data access operations to ote_dataset.
        # Note that list `data_infos` cannot be used here, since OTE dataset class does not have interface to
        # get only annotation of a data item, so we would load the whole data item (including image)
        # even if we need only checking aspect ratio of the image; due to it
        # this implementation of dataset does not uses such tricks as skipping images with wrong aspect ratios or
        # small image size, since otherwise reading the whole dataset during initialization will be required.
        self.data_infos = OTEDataset._DataInfoProxy(ote_dataset, labels)

        self.proposals = None  # Attribute expected by mmdet but not used for OTE datasets

        if not test_mode:
            self._set_group_flag()

        self.pipeline = Compose(pipeline)

    def _set_group_flag(self):
        """Set flag for grouping images.

        Originally, in Custom dataset, images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        This implementation will set group 0 for every image.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def _rand_another(self, idx):
        return np.random.choice(len(self))

    # In contrast with CustomDataset this implementation of dataset
    # does not filter images w.r.t. the min size
    def _filter_imgs(self, min_size=32):
        raise NotImplementedError

    @check_input_parameters_type()
    def prepare_train_img(self, idx: int) -> dict:
        """Get training data and annotations after pipeline.

        :param idx: int, Index of data.
        :return dict: Training data and annotation after pipeline with new keys introduced by pipeline.
        """
        item = deepcopy(self.data_infos[idx])
        self.pre_pipeline(item)
        return self.pipeline(item)

    @check_input_parameters_type()
    def prepare_test_img(self, idx: int) -> dict:
        """Get testing data after pipeline.

        :param idx: int, Index of data.
        :return dict: Testing data after pipeline with new keys introduced by pipeline.
        """
        # FIXME.
        # item = deepcopy(self.data_infos[idx])
        item = self.data_infos[idx]
        self.pre_pipeline(item)
        return self.pipeline(item)

    @staticmethod
    @check_input_parameters_type()
    def pre_pipeline(results: Dict[str, Any]):
        """Prepare results dict for pipeline. Add expected keys to the dict. """
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    @check_input_parameters_type()
    def get_ann_info(self, idx: int):
        """
        This method is used for evaluation of predictions. The CustomDataset class implements a method
        CustomDataset.evaluate, which uses the class method get_ann_info to retrieve annotations.

        :param idx: index of the dataset item for which to get the annotations
        :return ann_info: dict that contains the coordinates of the bboxes and their corresponding labels
        """
        dataset_item = self.ote_dataset[idx]
        labels = self.labels
        return get_annotation_mmdet_format(dataset_item, labels, self.domain)
