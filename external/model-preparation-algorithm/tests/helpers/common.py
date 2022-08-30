# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import time
from typing import List, Type

from mpa_tasks.apis import BaseTask
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.metrics import Performance
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.test_suite.training_tests_actions import (
    BaseOTETestAction,
    OTETestExportAction,
    OTETestExportEvaluationAction,
    OTETestPotAction,
    OTETestPotEvaluationAction,
    OTETestTrainingAction,
    OTETestTrainingEvaluationAction,
)


def get_test_action_classes() -> List[Type[BaseOTETestAction]]:
    return [
        OTETestTrainingAction,
        OTETestTrainingEvaluationAction,
        OTETestExportAction,
        OTETestExportEvaluationAction,
        OTETestPotAction,
        OTETestPotEvaluationAction,
    ]


def eval(task: BaseTask, model: ModelEntity, dataset: DatasetEntity) -> Performance:
    start_time = time.time()
    result_dataset = task.infer(dataset.with_empty_annotations())
    end_time = time.time()
    print(f"{len(dataset)} analysed in {end_time - start_time} seconds")
    result_set = ResultSetEntity(
        model=model, ground_truth_dataset=dataset, prediction_dataset=result_dataset
    )
    task.evaluate(result_set)
    assert result_set.performance is not None
    return result_set.performance
