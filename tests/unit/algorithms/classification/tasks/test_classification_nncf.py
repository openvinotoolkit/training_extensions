# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from mmcv.utils import Config

from otx.algorithms.classification.tasks import ClassificationNNCFTask
from otx.algorithms.common.tasks import BaseTask
from otx.algorithms.common.tasks.nncf_base import NNCFBaseTask
from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.configuration.helper import create
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.metrics import NullPerformance
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.entities.model_template import parse_model_template
from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import (
    DEFAULT_CLS_TEMPLATE,
    init_environment,
)


@pytest.fixture
def otx_model():
    model_configuration = ModelConfiguration(
        configurable_parameters=ConfigurableParameters(header="header", description="description"),
        label_schema=LabelSchemaEntity(),
    )
    return ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)


class TestOTXClsTaskNNCF:
    @pytest.fixture(autouse=True)
    def setup(self, otx_model, tmp_dir_path) -> None:
        model_template = parse_model_template(DEFAULT_CLS_TEMPLATE)
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.learning_parameters.num_iters = 10
        self.task_env, self.dataset = init_environment(params=hyper_parameters, model_template=model_template)
        self.model = otx_model
        self.cls_nncf_task = ClassificationNNCFTask(self.task_env, output_path=str(tmp_dir_path))

    @e2e_pytest_unit
    def test_save_model(self, mocker):
        mocker.patch("torch.load", return_value="")
        self.cls_nncf_task._recipe_cfg = Config({"model": {}})
        self.cls_nncf_task.save_model(self.model)

        assert self.model.get_data("weights.pth")
        assert self.model.get_data("label_schema.json")

    @e2e_pytest_unit
    def test_optimize(self, mocker):
        from otx.algorithms.common.adapters.mmcv.hooks import OTXLoggerHook

        # generate some dummy learning curves
        mock_lcurve_val = OTXLoggerHook.Curve()
        mock_lcurve_val.x = [0, 1]
        mock_lcurve_val.y = [0.1, 0.2]
        # patch training process
        mocker.patch.object(BaseTask, "_run_task", return_value={"final_ckpt": ""})
        self.cls_nncf_task._learning_curves = {"val/accuracy_top-1": mock_lcurve_val}
        mocker.patch.object(ClassificationNNCFTask, "save_model")
        self.cls_nncf_task.optimize(OptimizationType.NNCF, self.dataset, self.model)

        assert self.model.performance != NullPerformance()
        assert self.model.performance.score.value == 0.2

    @e2e_pytest_unit
    def test_initialize(self, mocker):
        """Test initialize method in OTXDetTaskNNCF."""
        options = {}
        self.cls_nncf_task._initialize(options)

        assert "model_builder" in options
        assert NNCFBaseTask.model_builder == options["model_builder"].func
