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


import logging
import os
from collections import namedtuple
from copy import deepcopy
from typing import List, Type

from ote_anomalib.data.mvtec import OteMvtecDataset
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model_template import TaskType
from ote_sdk.test_suite.training_tests_common import (
    make_paths_be_abs,
    ROOT_PATH_KEY,
)
from ote_sdk.test_suite.training_tests_actions import (create_environment_and_task,
                                                       OTETestTrainingAction,
                                                       BaseOTETestAction,
                                                       OTETestTrainingEvaluationAction,
                                                       OTETestExportAction,
                                                       OTETestExportEvaluationAction,
                                                       OTETestPotAction,
                                                       OTETestPotEvaluationAction,
                                                       OTETestNNCFAction,
                                                       OTETestNNCFEvaluationAction,
                                                       OTETestNNCFExportAction,
                                                       OTETestNNCFExportEvaluationAction,
                                                       OTETestNNCFGraphAction)


logger = logging.getLogger(__name__)


def DATASET_PARAMETERS_FIELDS() -> List[str]:
    return deepcopy(["dataset_path"])


DatasetParameters = namedtuple("DatasetParameters", DATASET_PARAMETERS_FIELDS())  # type: ignore


def _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name):
    if dataset_name not in dataset_definitions:
        raise ValueError(
            f"dataset {dataset_name} is absent in dataset_definitions, "
            f"dataset_definitions.keys={list(dataset_definitions.keys())}"
        )
    cur_dataset_definition = dataset_definitions[dataset_name]
    training_parameters_fields = {k: v for k, v in cur_dataset_definition.items() if k in DATASET_PARAMETERS_FIELDS()}
    print(f"training_parameters_fields: {training_parameters_fields}")
    make_paths_be_abs(training_parameters_fields, dataset_definitions[ROOT_PATH_KEY])
    print(f"training_parameters_fields after make_paths_be_abs: {training_parameters_fields}")

    assert set(DATASET_PARAMETERS_FIELDS()) == set(
        training_parameters_fields.keys()
    ), f"ERROR: dataset definitions for name={dataset_name} does not contain all required fields"
    assert all(
        training_parameters_fields.values()
    ), f"ERROR: dataset definitions for name={dataset_name} contains empty values for some required fields"

    params = DatasetParameters(**training_parameters_fields)
    return params


def _create_anomaly_dataset_and_labels_schema(dataset_params: DatasetParameters, dataset_name: str, task_type: TaskType):
    logger.debug(f'Path to dataset: {dataset_params.dataset_path}')
    category_list = [f.path for f in os.scandir(dataset_params.dataset_path) if f.is_dir()]
    items = []
    if "short" in dataset_name:
        logger.debug(f'Creating short dataset {dataset_name}')
        items.extend(OteMvtecDataset(path=dataset_params.dataset_path, seed=0, task_type=task_type)
                     .generate())
    else:
        for category in category_list:
            logger.debug(f'Creating dataset for {category}')
            items.extend(OteMvtecDataset(path=category, seed=0, task_type=task_type).generate())
    dataset = DatasetEntity(items=items)
    labels = dataset.get_labels()
    labels_schema = LabelSchemaEntity.from_labels(labels)
    return dataset, labels_schema


def get_anomaly_domain_test_action_classes(anomaly_domain_test_train_action: OTETestTrainingAction) -> List[Type[BaseOTETestAction]]:
    return [
        anomaly_domain_test_train_action,
        OTETestTrainingEvaluationAction,
        OTETestExportAction,
        OTETestExportEvaluationAction,
        OTETestPotAction,
        OTETestPotEvaluationAction,
        OTETestNNCFAction,
        OTETestNNCFEvaluationAction,
        OTETestNNCFExportAction,
        OTETestNNCFExportEvaluationAction,
        OTETestNNCFGraphAction,
    ]


