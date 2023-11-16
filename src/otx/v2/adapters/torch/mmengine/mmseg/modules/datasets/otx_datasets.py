"""Base MMDataset for Segmentation Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from mmseg.datasets import BaseCDDataset

from otx.v2.adapters.torch.mmengine.mmseg.registry import DATASETS
from otx.v2.api.entities.label import LabelEntity

if TYPE_CHECKING:
    from datumaro.components.dataset import Dataset as DatumDataset


def check_and_convert_to_tuple(pipeline: list[dict] | None) -> list[dict] | None:
    """Check if the pipeline is None, and if not, convert any lists in the pipeline to tuples.

    Args:
        pipeline (list[dict] | None): The pipeline to check and convert.

    Returns:
        list[dict] | None: The converted pipeline, or None if the input was None.
    """
    if pipeline is None:
        return None
    for step in pipeline:
        for key, value in step.items():
            if isinstance(value, list):
                step[key] = tuple(value)
    return pipeline


@DATASETS.register_module()
class OTXSegDataset(BaseCDDataset):
    """Wrapper that allows using a OTX dataset to train mmsegmentation models.

    This wrapper is not based on the filesystem,
    but instead loads the items here directly from the OTX Dataset object.

    The wrapper overwrites some methods of the CustomDataset class: prepare_train_img, prepare_test_img and prepipeline
    Naming of certain attributes might seem a bit peculiar but this is due to the conventions set in CustomDataset. For
    instance, CustomDatasets expects the dataset items to be stored in the attribute data_infos, which is why it is
    named like that and not dataset_items.

    """

    def __init__(
        self,
        otx_dataset: DatumDataset,
        labels: list[LabelEntity],
        empty_label: list | None = None,
        pipeline: list | None = None,
        **kwargs,
    ) -> None:
        """Dataset class for OTX datasets.

        Args:
            otx_dataset (DatasetEntity): The OTX dataset to use.
            labels (list[LabelEntity]): List of label entities.
            empty_label (list | None, optional): Empty label. Defaults to None.
            pipeline (list | None, optional): Data processing pipeline. Defaults to None.
            **kwargs: Additional keyword arguments.

        """
        self.otx_dataset = otx_dataset
        self.empty_label = empty_label
        self.labels: list[LabelEntity] = labels

        # Add background label if not present
        mmseg_labels = [lbs.name for lbs in labels]
        if "background" not in mmseg_labels:
            mmseg_labels = ["background", *mmseg_labels]
        metainfo = {"classes": mmseg_labels}
        test_mode = kwargs.get("test_mode", False)
        super().__init__(
            metainfo=metainfo,
            pipeline=check_and_convert_to_tuple(pipeline),
            test_mode=test_mode,
            lazy_init=True,
        )
        self.serialize_data = None  # OTX has its own data caching mechanism
        self._fully_initialized = True
        # TODO (Eugene): item_ids is unnecessary, DatumaroDataset should support indexing
        # CVS-124394
        self.item_ids = [(item.id, item.subset) for item in self.otx_dataset]

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            int: number of items in the dataset
        """
        return len(self.otx_dataset)

    def prepare_data(self, idx: int) -> dict:
        # TODO (Eugene): this part could be reused from the base class
        # CVS-124394
        """Get item from dataset."""
        dataset = self.otx_dataset
        item = dataset.get(id=self.item_ids[idx][0], subset=self.item_ids[idx][1])
        ignored_labels = item.attributes.get("ignored_labels", [])

        data_info = {
            "dataset_item": item,
            "index": idx,
            "ignored_labels": ignored_labels,
            "seg_fields": [],
        }
        return self.pipeline(data_info)
