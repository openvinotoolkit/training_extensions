"""OTX adapters.torch.mmengine.mmdet.Dataset API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import Dataset as TorchDataset

from otx.v2.adapters.torch.mmengine.dataset import MMXDataset
from otx.v2.adapters.torch.mmengine.mmdet.modules.datasets import OTXDetDataset
from otx.v2.adapters.torch.mmengine.mmdet.registry import MMDetRegistry
from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.utils.decorators import add_subset_dataloader
from otx.v2.api.utils.type_utils import str_to_task_type

SUBSET_LIST = ["train", "val", "test", "unlabeled"]


def get_default_pipeline(
    subset: str = "train",
    is_semisl: bool = False,
    task: TaskType | str = TaskType.DETECTION,
) -> list:
    """Returns the default pipeline for training a model.

    Args:
        subset (str): Subset of default pipeline
        is_semisl (bool, optional): Whether to use a semi-supervised pipeline. Defaults to False.
        task (TaskType, str): Task type of the dataset. Detection or Instance segmentation.

    Returns:
        list: The default pipeline as a dictionary or list, depending on whether `is_semisl` is True or False.
    """
    message = "SemiSL is not implemented in MMDet task."
    if is_semisl:
        raise NotImplementedError(message)

    if isinstance(task, str):
        task = str_to_task_type(task)

    default_pipeline = {
        TaskType.DETECTION: {
            "train": [
                {
                    'type': 'LoadResizeDataFromOTXDataset',
                    'load_ann_cfg': {'type': 'LoadAnnotationFromOTXDataset', 'with_bbox': True},
                    'resize_cfg': {
                        'type': 'Resize', 'scale': (512, 512), 'keep_ratio': True, 'downscale_only': True,
                    },
                    'enable_memcache': True},
                {'type': 'MinIoURandomCrop', 'min_ious': (0.1, 0.3, 0.5, 0.7, 0.9), 'min_crop_size': 0.3},
                {'type': 'Resize', 'scale': (512, 512), 'keep_ratio': False},
                {'type': 'RandomFlip', 'prob': 0.5},
                {
                    'type': 'PackDetInputs',
                    'meta_keys': [
                        'ori_filename',
                        'flip_direction',
                        'scale_factor',
                        'gt_ann_ids',
                        'flip',
                        'ignored_labels',
                        'ori_shape',
                        'filename',
                        'img_shape',
                        'pad_shape',
                    ],
                },
            ],
            "val": [
                {
                    'type': 'LoadResizeDataFromOTXDataset',
                    'load_ann_cfg': {'type': 'LoadAnnotationFromOTXDataset', 'with_bbox': True},
                    'resize_cfg': {'type': 'Resize', 'scale': (512, 512), 'keep_ratio': False},
                    'enable_memcache': True,
                    'eval_mode': True,
                },
                {
                    'type': 'PackDetInputs',
                    'meta_keys': ['ori_filename', 'scale_factor', 'ori_shape', 'filename', 'img_shape', 'pad_shape'],
                },
            ],
            "test": [
                {
                    'type': 'LoadResizeDataFromOTXDataset',
                    'load_ann_cfg': {'type': 'LoadAnnotationFromOTXDataset', 'with_bbox': True},
                    'resize_cfg': {'type': 'Resize', 'scale': (512, 512), 'keep_ratio': False},
                    'enable_memcache': True,
                    'eval_mode': True,
                },
                {
                    'type': 'PackDetInputs',
                    'meta_keys': ['ori_filename', 'scale_factor', 'ori_shape', 'filename', 'img_shape', 'pad_shape'],
                },
            ],
        },
        TaskType.INSTANCE_SEGMENTATION: {
            "train": [
                {
                    "type": "LoadResizeDataFromOTXDataset",
                    "load_ann_cfg": {
                        "type": "LoadAnnotationFromOTXDataset",
                        "domain": "instance_segmentation",
                        "with_bbox": True,
                        "with_mask": True,
                        "poly2mask": False,
                    },
                    "resize_cfg": {
                        "type": "Resize",
                        "scale": (512, 512),
                        "keep_ratio": False,
                    },
                    "enable_memcache": True,  # Cache after resizing image & annotations
                },
                {"type": "RandomFlip", "prob": 0.5},
                {"type": "Pad", "size_divisor": 32},
                {
                    "type": "PackDetInputs",
                    "meta_keys": [
                        "ori_filename",
                        "flip_direction",
                        "scale_factor",
                        "gt_ann_ids",
                        "flip",
                        "ignored_labels",
                        "ori_shape",
                        "filename",
                        "img_shape",
                        "pad_shape",
                    ],
                },
            ],
            "val": [
                {
                    "type": "LoadResizeDataFromOTXDataset",
                    "load_ann_cfg": {
                        "type": "LoadAnnotationFromOTXDataset",
                        "domain": "instance_segmentation",
                        "with_bbox": True,
                        "with_mask": True,
                        "poly2mask": False,
                    },
                    "resize_cfg": {"type": "Resize", "scale": (512, 512), "keep_ratio": False},
                    "enable_memcache": True,  # Cache after resizing image
                    "eval_mode": True,
                },
                {
                    "type": "PackDetInputs",
                    "meta_keys": ["ori_filename", "scale_factor", "ori_shape", "filename", "img_shape", "pad_shape"],
                },
            ],
            "test": [
                {"type": "LoadImageFromOTXDataset"},
                {"type": "Resize", "scale": (512, 512), "keep_ratio": False},
                {
                    "type": "LoadAnnotationFromOTXDataset",
                    "domain": "instance_segmentation",
                    "with_bbox": True,
                    "with_mask": True,
                    "poly2mask": False,
                },
                {
                    "type": "PackDetInputs",
                    "meta_keys": ["ori_filename", "scale_factor", "ori_shape", "filename", "img_shape", "pad_shape"],
                },
            ],
        },
    }
    return default_pipeline[task][subset]


@add_subset_dataloader(SUBSET_LIST)
class MMDetDataset(MMXDataset):
    """A class representing a dataset for training a model."""

    def __init__(
        self,
        task: TaskType | str | None = None,
        train_type: TrainType | str | None = None,
        train_data_roots: str | None = None,
        train_ann_files: str | None = None,
        val_data_roots: str | None = None,
        val_ann_files: str | None = None,
        test_data_roots: str | None = None,
        test_ann_files: str | None = None,
        unlabeled_data_roots: str | None = None,
        unlabeled_file_list: str | None = None,
        data_format: str | None = None,
    ) -> None:
        r"""MMDet's Dataset class.

        Args:
            task (TaskType | str | None, optional): The task type of the dataset want to load.
                Defaults to None.
            train_type (TrainType | str | None, optional): The train type of the dataset want to load.
                Defaults to None.
            train_data_roots (str | None, optional): The root address of the dataset to be used for training.
                Defaults to None.
            train_ann_files (str | None, optional): Location of the annotation file for the dataset
            val_data_roots (str | None, optional): The root address of the dataset
                to be used for validation. Defaults to None.
            val_ann_files (str | None, optional): Location of the annotation file for the dataset
                to be used for validation. Defaults to None.
            test_data_roots (str | None, optional): The root address of the dataset
                to be used for testing. Defaults to None.
            test_ann_files (str | None, optional): Location of the annotation file for the dataset
                to be used for testing. Defaults to None.
            unlabeled_data_roots (str | None, optional): The root address of the unlabeled dataset
                to be used for training. Defaults to None.
            unlabeled_file_list (str | None, optional): The file where the list of unlabeled images is declared.
                Defaults to None.
            data_format (str | None, optional): The format of the dataset. Defaults to None.
        """
        super().__init__(
            task,
            train_type,
            train_data_roots,
            train_ann_files,
            val_data_roots,
            val_ann_files,
            test_data_roots,
            test_ann_files,
            unlabeled_data_roots,
            unlabeled_file_list,
            data_format,
        )
        self.scope = "mmdet"
        self.dataset_registry = MMDetRegistry().get("dataset")

    def _get_sub_task_dataset(self) -> TorchDataset:
        return OTXDetDataset

    def _build_dataset(
        self,
        subset: str,
        pipeline: list | None = None,
        config: dict | None = None,
    ) -> TorchDataset | None:
        """Builds a TorchDataset object for the given subset using the specified pipeline and configuration.

        Args:
            subset (str): The subset to build the dataset for.
            pipeline (list | None, optional]): The pipeline to use for the dataset.
                Defaults to None.
            config (dict | None, optional): The configuration to use for the dataset.
                Defaults to None.

        Returns:
            TorchDataset | None, optional: The built TorchDataset object, or None if the dataset is empty.
        """
        dataset_config = config.get("dataset", config) if config is not None else {}
        if pipeline is None and "pipeline" not in dataset_config:
            is_semisl = subset == "unlabeled"
            pipeline = get_default_pipeline(subset, is_semisl=is_semisl, task=self.task)
        return super()._build_dataset(subset, pipeline, config)
