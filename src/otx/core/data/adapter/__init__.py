"""OTX Core Data Adapter."""

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

# pylint: disable=too-many-return-statements, too-many-arguments
import importlib
import os

from otx.algorithms.common.configs.training_base import TrainType
from otx.api.entities.model_template import TaskType

ADAPTERS = {
    TaskType.CLASSIFICATION: {
        "Incremental": {
            "module_name": "classification_dataset_adapter",
            "class": "ClassificationDatasetAdapter",
        },
        "Selfsupervised": {
            "module_name": "classification_dataset_adapter",
            "class": "SelfSLClassificationDatasetAdapter",
        },
    },
    TaskType.DETECTION: {
        "Incremental": {
            "module_name": "detection_dataset_adapter",
            "class": "DetectionDatasetAdapter",
        }
    },
    TaskType.ROTATED_DETECTION: {
        "Incremental": {
            "module_name": "detection_dataset_adapter",
            "class": "DetectionDatasetAdapter",
        }
    },
    TaskType.INSTANCE_SEGMENTATION: {
        "Incremental": {
            "module_name": "detection_dataset_adapter",
            "class": "DetectionDatasetAdapter",
        }
    },
    TaskType.SEGMENTATION: {
        "Incremental": {
            "module_name": "segmentation_dataset_adapter",
            "class": "SegmentationDatasetAdapter",
        },
        "Selfsupervised": {
            "module_name": "segmentation_dataset_adapter",
            "class": "SelfSLSegmentationDatasetAdapter",
        },
    },
    TaskType.ANOMALY_CLASSIFICATION: {
        "Incremental": {
            "module_name": "anomaly_dataset_adapter",
            "class": "AnomalyClassificationDatasetAdapter",
        }
    },
    TaskType.ANOMALY_DETECTION: {
        "Incremental": {
            "module_name": "anomaly_dataset_adapter",
            "class": "AnomalyDetectionDatasetAdapter",
        }
    },
    TaskType.ANOMALY_SEGMENTATION: {
        "Incremental": {
            "module_name": "anomaly_dataset_adapter",
            "class": "AnomalySegmentationDatasetAdapter",
        }
    },
}
if os.getenv("FEATURE_FLAGS_OTX_ACTION_TASKS", "0") == "1":
    ADAPTERS.update(
        {
            TaskType.ACTION_CLASSIFICATION: {
                "Incremental": {
                    "module_name": "action_dataset_adapter",
                    "class": "ActionClassificationDatasetAdapter",
                }
            },
            TaskType.ACTION_DETECTION: {
                "Incremental": {
                    "module_name": "action_dataset_adapter",
                    "class": "ActionDetectionDatasetAdapter",
                }
            },
        }
    )
# TODO: update to real template
if os.getenv("FEATURE_FLAGS_OTX_VISUAL_PROMPTING_TASKS", "0") == "1":
    ADAPTERS.update(
        {
            TaskType.VISUAL_PROMPTING: {
                "Incremental": {
                    "module_name": "visual_prompting_dataset_adapter",
                    "class": "VisualPromptingDatasetAdapter",
                }
            },
        }
    )


def get_dataset_adapter(
    task_type: TaskType,
    train_type: TrainType,
    train_data_roots: str = None,
    train_ann_files: str = None,
    val_data_roots: str = None,
    val_ann_files: str = None,
    test_data_roots: str = None,
    test_ann_files: str = None,
    unlabeled_data_roots: str = None,
    unlabeled_file_list: str = None,
    **kwargs,
):
    """Returns a dataset class by task type.

    Args:
        task_type: A task type such as ANOMALY_CLASSIFICATION, ANOMALY_DETECTION, ANOMALY_SEGMENTATION,
            CLASSIFICATION, INSTANCE_SEGMENTATION, DETECTION, CLASSIFICATION, ROTATED_DETECTION, SEGMENTATION.
        train_type: train type such as Incremental and Selfsupervised.
            Selfsupervised is only supported for SEGMENTATION.
        train_data_roots: the path of data root for training data
        train_ann_files: the path of annotation file for training data
        val_data_roots: the path of data root for validation data
        val_ann_files: the path of annotation file for validation data
        test_data_roots: the path of data root for test data
        test_ann_files: the path of annotation file for test data
        unlabeled_data_roots: the path of data root for unlabeled data
        unlabeled_file_list: the path of unlabeled file list
        kwargs: optional kwargs
    """

    train_type_to_be_called = str(
        train_type if train_type == TrainType.Selfsupervised.value else TrainType.Incremental.value
    )
    module_root = "otx.core.data.adapter."
    module = importlib.import_module(module_root + ADAPTERS[task_type][train_type_to_be_called]["module_name"])
    return getattr(module, ADAPTERS[task_type][train_type_to_be_called]["class"])(
        task_type=task_type,
        train_data_roots=train_data_roots,
        train_ann_files=train_ann_files,
        val_data_roots=val_data_roots,
        val_ann_files=val_ann_files,
        test_data_roots=test_data_roots,
        test_ann_files=test_ann_files,
        unlabeled_data_roots=unlabeled_data_roots,
        unlabeled_file_list=unlabeled_file_list,
        **kwargs,
    )
