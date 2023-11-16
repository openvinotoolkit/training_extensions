"""Base MMDataset for Detection Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from mmcv.transforms import Compose
from mmdet.datasets import BaseDetDataset
from mmdet.registry import DATASETS

if TYPE_CHECKING:
    from datumaro.components.dataset import Dataset as DatumDataset

from otx.v2.api.entities.label import LabelEntity
from otx.v2.api.entities.utils.data_utils import get_old_new_img_indices


@DATASETS.register_module()
class OTXDetDataset(BaseDetDataset):
    """Wrapper that allows using a OTX dataset to train mmdetection models.

    This wrapper is not based on the filesystem,
    but instead loads the items here directly from the OTX DatasetEntity object.

    The wrapper overwrites some methods of the BaseDetDataset class: prepare_train_img, prepare_test_img and prepipeline
    Naming of certain attributes might seem a bit peculiar but this is due to the conventions set in BaseDetDataset. For
    instance, CustomDatasets expects the dataset items to be stored in the attribute data_infos, which is why it is
    named like that and not dataset_items.

    """

    class _DataInfoProxy:
        """This class is intended to be a wrapper to use it in BaseDetDataset-derived class as `self.data_infos`.

        Instead of using list `data_infos` as in BaseDetDataset, our implementation of dataset OTXDataset
        uses this proxy class with overriden __len__ and __getitem__; this proxy class
        forwards data access operations to otx_dataset and converts the dataset items to the view
        convenient for mmdetection.

        Args:
            otx_dataset (DatasetEntity): DatasetEntity from dataset api
            labels (List[LabelEntity]): List of LabelEntity
        """

        def __init__(self, otx_dataset: DatumDataset, labels: list[LabelEntity]) -> None:
            self.otx_dataset = otx_dataset
            self.labels = labels
            self.label_idx = {label.id: i for i, label in enumerate(labels)}
            self.item_ids = [(item.id, item.subset) for item in self.otx_dataset]

        def __len__(self) -> int:
            return len(self.otx_dataset)

        def __getitem__(self, index: int) -> dict[str, Any]:
            """Prepare a dict 'data_info' that is expected by the mmdet pipeline to handle images and annotations.

            :return data_info: dictionary that contains the image and image metadata, as well as the labels of
            the objects in the image
            """
            dataset = self.otx_dataset
            item = dataset.get(id=self.item_ids[index][0], subset=self.item_ids[index][1])
            ignored_labels = item.attributes.get("ignored_labels", [])

            return {
                "dataset_item": item,
                "index": index,
                "ann_info": {"label_list": self.labels},
                "ignored_labels": ignored_labels,
                "bbox_fields": [],
                "mask_fields": [],
                "seg_fields": [],
            }

    def __init__(
        self,
        otx_dataset: DatumDataset,
        labels: list[LabelEntity],
        pipeline: Sequence[dict],
        test_mode: bool = False,
        max_refetch: int = 1000,
        **kwargs,
    ) -> None:
        """Initialize OTXDetDataset.

        Args:
            otx_dataset (DatasetEntiy): DatasetEntity from dataset api
            labels (List[LabelEntity]): List of LabelEntity
            pipeline (Sequence[dict]): List of data pipeline
            test_mode (bool): Whether current dataset is for test or not
            max_refetch (int): If ``Basedataset.prepare_data`` get a None img.
                The maximum extra number of cycles to get a valid
            kwargs: Additional kwargs
        """
        dataset_cfg = kwargs.copy()
        _ = dataset_cfg.pop("org_type", None)
        new_classes = dataset_cfg.pop("new_classes", [])
        self.otx_dataset = otx_dataset
        self.labels = labels
        self.domain = self.labels[0].domain
        self.test_mode = test_mode
        self.max_refetch = max_refetch

        self._metainfo = {"classes": [label.name for label in labels], "domain": self.domain}

        # Instead of using list data_infos as in BaseDetDataset, this implementation of dataset
        # uses a proxy class with overriden __len__ and __getitem__; this proxy class
        # forwards data access operations to otx_dataset.
        # Note that list `data_infos` cannot be used here, since OTX dataset class does not have interface to
        # get only annotation of a data item, so we would load the whole data item (including image)
        # even if we need only checking aspect ratio of the image; due to it
        # this implementation of dataset does not uses such tricks as skipping images with wrong aspect ratios or
        # small image size, since otherwise reading the whole dataset during initialization will be required.
        self.data_list: Any = OTXDetDataset._DataInfoProxy(otx_dataset, labels)

        self.proposals = None  # Attribute expected by mmdet but not used for OTX datasets

        if not test_mode:
            self.img_indices = get_old_new_img_indices(self.labels, new_classes, self.otx_dataset)

        self.pipeline = Compose(pipeline)
        self.serialize_data = None  # OTX has own data caching mechanism
        self._fully_initialized = True

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.data_list)
