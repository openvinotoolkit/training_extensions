import pytest
import numpy as np
from pathlib import Path
from otx.algorithms.classification.tasks import ClassificationTrainTask
from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.entities.model_template import parse_model_template
from otx.api.entities.train_parameters import TrainParameters
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)
from otx.algorithms.common.tasks import BaseTask
from otx.api.configuration.helper import create
from tests.unit.algorithms.classification.test_helper import init_environment, DEFAULT_CLS_TEMPLATE

from otx.api.entities.metrics import NullPerformance


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
        model_template = parse_model_template(DEFAULT_CLS_TEMPLATE)
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.learning_parameters.num_iters = 10
        self.task_env, self.dataset, _ = init_environment(params=hyper_parameters, model_template=model_template)
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
        def progress_callback(progress: float, score = None):
            training_progress_curve.append(progress)

        from otx.algorithms.common.adapters.mmcv.hooks import OTXLoggerHook
        training_progress_curve = []

        train_parameters = TrainParameters()
        train_parameters.update_progress = progress_callback

        self.cls_train_task.train(self.dataset, self.model, train_parameters)

        assert self.model.performance != NullPerformance()
        assert 0. < self.model.performance.score.value <= 1.
        assert len(training_progress_curve) > 0
        assert np.all(training_progress_curve[1:] >= training_progress_curve[:-1])

    @e2e_pytest_unit
    def test_cancel_training(self):
        self.cls_train_task.cancel_training()

        assert self.cls_train_task._should_stop is True
