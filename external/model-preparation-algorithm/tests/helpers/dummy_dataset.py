# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import logging
import os.path as osp
from collections import namedtuple
from copy import deepcopy
from typing import List

from detection_tasks.extension.datasets.data_utils import load_dataset_items_coco_format
from segmentation_tasks.extension.datasets.mmdataset import load_dataset_items
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label import Domain
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.test_suite.training_tests_common import (
    ROOT_PATH_KEY,
    make_paths_be_abs,
)
from torchreid_tasks.utils import ClassificationDatasetAdapter

logger = logging.getLogger(__name__)


def DATASET_PARAMETERS_FIELDS() -> List[str]:
    return deepcopy(
        [
            "annotations_train",
            "images_train_dir",
            "annotations_val",
            "images_val_dir",
            "annotations_test",
            "images_test_dir",
            "pre_trained_model",
        ]
    )


DatasetParameters = namedtuple("DatasetParameters", DATASET_PARAMETERS_FIELDS())


def get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name):
    if dataset_name not in dataset_definitions:
        raise ValueError(
            f"dataset {dataset_name} is absent in dataset_definitions, "
            f"dataset_definitions.keys={list(dataset_definitions.keys())}"
        )
    cur_dataset_definition = dataset_definitions[dataset_name]
    training_parameters_fields = {
        k: v
        for k, v in cur_dataset_definition.items()
        if k in DATASET_PARAMETERS_FIELDS()
    }
    make_paths_be_abs(training_parameters_fields, dataset_definitions[ROOT_PATH_KEY])

    assert set(DATASET_PARAMETERS_FIELDS()) == set(
        training_parameters_fields.keys()
    ), f"ERROR: dataset definitions for name={dataset_name} does not contain all required fields"
    assert all(
        training_parameters_fields.values()
    ), f"ERROR: dataset definitions for name={dataset_name} contains empty values for some required fields"

    params = DatasetParameters(**training_parameters_fields)
    return params


def create_classification_dataset_and_labels_schema(dataset_params, model_name):
    logger.debug(f"Using for train annotation file {dataset_params.annotations_train}")
    logger.debug(f"Using for val annotation file {dataset_params.annotations_val}")

    dataset = ClassificationDatasetAdapter(
        train_data_root=osp.join(dataset_params.images_train_dir),
        train_ann_file=osp.join(dataset_params.annotations_train),
        val_data_root=osp.join(dataset_params.images_val_dir),
        val_ann_file=osp.join(dataset_params.annotations_val),
        test_data_root=osp.join(dataset_params.images_test_dir),
        test_ann_file=osp.join(dataset_params.annotations_test),
    )

    labels_schema = LabelSchemaEntity.from_labels(dataset.get_labels())
    return dataset, labels_schema


def create_object_detection_dataset_and_labels_schema(dataset_params):
    logger.debug(f"Using for train annotation file {dataset_params.annotations_train}")
    logger.debug(f"Using for val annotation file {dataset_params.annotations_val}")
    labels_list = []
    items = []
    items.extend(
        load_dataset_items_coco_format(
            ann_file_path=dataset_params.annotations_train,
            data_root_dir=dataset_params.images_train_dir,
            domain=Domain.DETECTION,
            subset=Subset.TRAINING,
            labels_list=labels_list,
        )
    )
    items.extend(
        load_dataset_items_coco_format(
            ann_file_path=dataset_params.annotations_val,
            data_root_dir=dataset_params.images_val_dir,
            domain=Domain.DETECTION,
            subset=Subset.VALIDATION,
            labels_list=labels_list,
        )
    )
    items.extend(
        load_dataset_items_coco_format(
            ann_file_path=dataset_params.annotations_test,
            data_root_dir=dataset_params.images_test_dir,
            domain=Domain.DETECTION,
            subset=Subset.TESTING,
            labels_list=labels_list,
        )
    )
    dataset = DatasetEntity(items=items)
    labels_schema = LabelSchemaEntity.from_labels(dataset.get_labels())
    return dataset, labels_schema


def create_segmentation_dataset_and_labels_schema(dataset_params):
    logger.debug(f"Using for train annotation file {dataset_params.annotations_train}")
    logger.debug(f"Using for val annotation file {dataset_params.annotations_val}")
    labels_list = []
    items = load_dataset_items(
        ann_file_path=dataset_params.annotations_train,
        data_root_dir=dataset_params.images_train_dir,
        subset=Subset.TRAINING,
        labels_list=labels_list,
    )
    items.extend(
        load_dataset_items(
            ann_file_path=dataset_params.annotations_val,
            data_root_dir=dataset_params.images_val_dir,
            subset=Subset.VALIDATION,
            labels_list=labels_list,
        )
    )
    items.extend(
        load_dataset_items(
            ann_file_path=dataset_params.annotations_test,
            data_root_dir=dataset_params.images_test_dir,
            subset=Subset.TESTING,
            labels_list=labels_list,
        )
    )
    dataset = DatasetEntity(items=items)
    labels_schema = LabelSchemaEntity.from_labels(labels_list)
    return dataset, labels_schema
