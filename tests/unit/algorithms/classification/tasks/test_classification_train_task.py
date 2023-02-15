# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from otx.algorithms.classification.tasks import ClassificationTrainTask
from otx.algorithms.common.tasks import BaseTask
from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.metrics import NullPerformance
from otx.api.entities.model import ModelConfiguration, ModelEntity
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import (
    DEFAULT_CLS_TEMPLATE,
    init_environment,
    setup_configurable_parameters,
)


@pytest.fixture
def otx_model():
    model_configuration = ModelConfiguration(
        configurable_parameters=ConfigurableParameters(header="header", description="description"),
        label_schema=LabelSchemaEntity(),
    )
    return ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)


class TestOTXClsTaskTrain:
    @pytest.fixture(autouse=True)
    def setup(self, otx_model, tmp_dir_path) -> None:
        hyper_parameters, model_template = setup_configurable_parameters(DEFAULT_CLS_TEMPLATE)
        self.task_env, self.dataset = init_environment(params=hyper_parameters, model_template=model_template)
        self.model = otx_model
        self.cls_train_task = ClassificationTrainTask(self.task_env, output_path=str(tmp_dir_path))

    @e2e_pytest_unit
    def test_save_model(self, mocker):
        mocker.patch("torch.load", return_value="")
        self.cls_train_task.save_model(self.model)

        assert self.model.get_data("weights.pth")
        assert self.model.get_data("label_schema.json")

    @e2e_pytest_unit
    def test_train(self, mocker):
        from otx.algorithms.common.adapters.mmcv.hooks import OTXLoggerHook

        # generate some dummy learning curves
        mock_lcurve_val = OTXLoggerHook.Curve()
        mock_lcurve_val.x = [0, 1]
        mock_lcurve_val.y = [0.1, 0.2]
        # patch training process
        mocker.patch.object(BaseTask, "_run_task", return_value={"final_ckpt": ""})
        self.cls_train_task._learning_curves = {"val/accuracy_top-1": mock_lcurve_val}
        mocker.patch.object(ClassificationTrainTask, "save_model")
        self.cls_train_task.train(self.dataset, self.model)

        assert self.model.performance != NullPerformance()
        assert self.model.performance.score.value == 0.2

    @e2e_pytest_unit
    def test_cancel_training(self):
        self.cls_train_task.cancel_training()

        assert self.cls_train_task._should_stop is True
