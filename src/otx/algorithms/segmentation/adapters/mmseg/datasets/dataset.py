"""Base MMDataset for Segmentation Task."""

# Copyright (C) 2023 Intel Corporation
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

from abc import ABCMeta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.pipelines import Compose

from otx.algorithms.common.utils.data import get_old_new_img_indices
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.utils.segmentation_utils import mask_from_dataset_item


# pylint: disable=invalid-name, too-many-locals, too-many-instance-attributes, super-init-not-called
def get_annotation_mmseg_format(
    dataset_item: DatasetItemEntity,
    labels: List[LabelEntity],
    use_otx_adapter: bool = True,
) -> dict:
    """Function to convert a OTX annotation to mmsegmentation format.

    This is used both in the OTXDataset class defined in this file
    as in the custom pipeline element 'LoadAnnotationFromOTXDataset'

    :param dataset_item: DatasetItem for which to get annotations
    :param labels: List of labels in the project
    :return dict: annotation information dict in mmseg format
    """
    gt_seg_map = mask_from_dataset_item(dataset_item, labels, use_otx_adapter)

    gt_seg_map = gt_seg_map.squeeze(2).astype(np.uint8)
    ann_info = dict(gt_semantic_seg=gt_seg_map)

    return ann_info


@DATASETS.register_module()
class _OTXSegDataset(CustomDataset, metaclass=ABCMeta):
    """Wrapper that allows using a OTX dataset to train mmsegmentation models.

    This wrapper is not based on the filesystem,
    but instead loads the items here directly from the OTX Dataset object.

    The wrapper overwrites some methods of the CustomDataset class: prepare_train_img, prepare_test_img and prepipeline
    Naming of certain attributes might seem a bit peculiar but this is due to the conventions set in CustomDataset. For
    instance, CustomDatasets expects the dataset items to be stored in the attribute data_infos, which is why it is
    named like that and not dataset_items.

    """

    class _DataInfoProxy:
        """This class is intended to be a wrapper to use it in CustomDataset-derived class as `self.data_infos`.

        Instead of using list `data_infos` as in CustomDataset, our implementation of dataset OTXDataset
        uses this proxy class with overriden __len__ and __getitem__; this proxy class
        forwards data access operations to otx_dataset and converts the dataset items to the view
        convenient for mmsegmentation.
        """

        def __init__(
            self,
            otx_dataset,
            labels=None,
            **kwargs,  # pylint: disable=unused-argument
        ):
            self.otx_dataset = otx_dataset
            self.labels = labels
            self.label_idx = {label.id: i for i, label in enumerate(labels)}

        def __len__(self):
            return len(self.otx_dataset)

        def __getitem__(self, index):
            """Prepare a dict 'data_info' that is expected by the mmseg pipeline to handle images and annotations.

            :return data_info: dictionary that contains the image and image metadata, as well as the labels of
            the objects in the image
            """
            dataset = self.otx_dataset
            item = dataset[index]
            ignored_labels = np.array([self.label_idx[lbs.id] + 1 for lbs in item.ignored_labels])

            data_info = dict(
                dataset_item=item,
                width=item.width,
                height=item.height,
                index=index,
                ann_info=dict(labels=self.labels),
                ignored_labels=ignored_labels,
            )

            return data_info

    def __init__(
        self,
        otx_dataset: DatasetEntity,
        pipeline: Sequence[dict],
        classes: Optional[List[str]] = None,
        test_mode: bool = False,
        use_otx_adapter: bool = True,
    ):
        self.otx_dataset = otx_dataset
        self.test_mode = test_mode

        self.ignore_index = 255
        self.reduce_zero_label = False
        self.label_map = None
        self.use_otx_adapter = use_otx_adapter

        dataset_labels = self.otx_dataset.get_labels(include_empty=False)
        self.project_labels = self.filter_labels(dataset_labels, classes)
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(classes, None)

        # Instead of using list data_infos as in CustomDataset, this implementation of dataset
        # uses a proxy class with overriden __len__ and __getitem__; this proxy class
        # forwards data access operations to otx_dataset.
        # Note that list `data_infos` cannot be used here, since OTX dataset class does not have interface to
        # get only annotation of a data item, so we would load the whole data item (including image)
        # even if we need only checking aspect ratio of the image; due to it
        # this implementation of dataset does not uses such tricks as skipping images with wrong aspect ratios or
        # small image size, since otherwise reading the whole dataset during initialization will be required.
        self.data_infos = _OTXSegDataset._DataInfoProxy(self.otx_dataset, self.project_labels)

        self.pipeline = Compose(pipeline)

    @staticmethod
    def filter_labels(all_labels: List[LabelEntity], label_names: List[str]) -> List[LabelEntity]:
        """Filter and collect actual label entities."""
        filtered_labels = []
        for label_name in label_names:
            matches = [label for label in all_labels if label.name == label_name]
            if len(matches) == 0:
                continue

            assert len(matches) == 1

            filtered_labels.append(matches[0])

        return filtered_labels

    def __len__(self):
        """Total number of samples of data."""

        return len(self.data_infos)

    def pre_pipeline(self, results: Dict[str, Any]):
        """Prepare results dict for pipeline."""

        results["seg_fields"] = []

    def prepare_train_img(self, idx: int) -> dict:
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys introduced by pipeline.
        """

        item = self.data_infos[idx]

        self.pre_pipeline(item)
        out = self.pipeline(item)

        return out

    def prepare_test_img(self, idx: int) -> dict:
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by pipeline.
        """

        item = self.data_infos[idx]

        self.pre_pipeline(item)
        out = self.pipeline(item)

        return out

    def get_ann_info(self, idx: int):
        """This method is used for evaluation of predictions.

        The CustomDataset class implements a method
        CustomDataset.evaluate, which uses the class method get_ann_info to retrieve annotations.

        :param idx: index of the dataset item for which to get the annotations
        :return ann_info: dict that contains the coordinates of the bboxes and their corresponding labels
        """

        dataset_item = self.otx_dataset[idx]
        ann_info = get_annotation_mmseg_format(dataset_item, self.project_labels, self.use_otx_adapter)

        return ann_info

    def get_gt_seg_maps(self, efficient_test: bool = False):
        """Get ground truth segmentation maps for evaluation."""

        gt_seg_maps = []
        for item_id in range(len(self)):
            ann_info = self.get_ann_info(item_id)
            gt_seg_maps.append(ann_info["gt_semantic_seg"])
        if efficient_test:
            pass

        return gt_seg_maps


@DATASETS.register_module()
class OTXSegDataset(_OTXSegDataset, metaclass=ABCMeta):
    """Wrapper dataset that allows using a OTX dataset to train models."""

    def __init__(self, **kwargs):
        pipeline = []
        test_mode = kwargs.get("test_mode", False)
        use_otx_adapter = True
        if "dataset" in kwargs:
            dataset = kwargs["dataset"]
            otx_dataset = dataset.otx_dataset
            pipeline = dataset.pipeline
            classes = dataset.labels
            new_classes = dataset.new_classes
        else:
            otx_dataset = kwargs["otx_dataset"]
            pipeline = kwargs["pipeline"]
            classes = kwargs["labels"]
            new_classes = kwargs.get("new_classes", [])

        if test_mode is False:
            self.img_indices = get_old_new_img_indices(classes, new_classes, otx_dataset)

        for pipe in pipeline:
            if pipe["type"] == "LoadImageFromOTXDataset" and "use_otx_adapter" in pipe:
                use_otx_adapter = pipe["use_otx_adapter"]
                break

        if classes:
            classes = [c.name for c in classes]
            classes = ["background"] + classes
        else:
            classes = []
        super().__init__(
            otx_dataset=otx_dataset,
            pipeline=pipeline,
            classes=classes,
            use_otx_adapter=use_otx_adapter,
        )

        self.CLASSES = [label.name for label in self.project_labels]
        if "background" not in self.CLASSES:
            self.CLASSES = ["background"] + self.CLASSES

        if self.label_map is None:
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in classes:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)
