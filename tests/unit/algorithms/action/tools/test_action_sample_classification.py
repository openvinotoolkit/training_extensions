"""Unit Test for otx.algorithms.action.tools.sample_classification."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.utils import Config

from otx.algorithms.action.tools.sample_classification import (
    load_test_dataset,
    main,
    parse_args,
)
from otx.algorithms.common.configs.training_base import TrainType
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import Domain
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model_template import TaskType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.action.test_helpers import (
    generate_action_cls_otx_dataset,
    generate_labels,
)


@e2e_pytest_unit
def test_parse_args(mocker) -> None:
    """Test parse_args function."""

    class MockArgParser:
        def __init__(self, description):
            self.description = description

        def add_argument(self, name, *args, **kwargs):
            setattr(self, name.split("--")[-1], True)

        def parse_args(self):
            return self

    mocker.patch("otx.algorithms.action.tools.sample_classification.argparse.ArgumentParser", side_effect=MockArgParser)
    parser = parse_args()
    assert parser.template_file_path is not None
    assert parser.export is not None


@e2e_pytest_unit
def test_load_test_dataset() -> None:
    """Test laod_test_dataset function."""

    class MockTemplate:
        task_type = TaskType.ACTION_CLASSIFICATION
        hyper_parameters = Config(
            {"parameter_overrides": {"algo_backend": {"train_type": {"default_value": TrainType.Incremental.value}}}}
        )

    dataset, label_schema = load_test_dataset(MockTemplate())
    isinstance(dataset, DatasetEntity)
    isinstance(label_schema, LabelSchemaEntity)


@e2e_pytest_unit
def test_main(mocker) -> None:
    """Test main function."""

    class MockArgs:
        template_file_path = "dummy_path"
        export = False

    class MockTaskEnvironment:
        def __init__(self, *args, **kwargs):
            self.model = None

        def get_model_configuration(self):
            return True

    class MockTestCls:
        def __init__(self, task_environment):
            pass

        def train(self, dataset, output_model):
            pass

        def infer(self, dataset, params):
            return dataset

        def evaluate(self, resultset):
            resultset.performance = 1.0

        def export(self, export_type, model, dump_features):
            return model

        def optimize(self, optimization_type, dataset, modle, params):
            pass

    mocker.patch(
        "otx.algorithms.action.tools.sample_classification.parse_model_template",
        return_value=Config(
            {
                "hyper_parameters": {"data": "dummy_data"},
                "entrypoints": {"base": "dummy_base", "openvino": "dummy_base"},
            }
        ),
    )
    mocker.patch(
        "otx.algorithms.action.tools.sample_classification.load_test_dataset",
        return_value=(
            generate_action_cls_otx_dataset(3, 3, generate_labels(3, Domain.ACTION_CLASSIFICATION)),
            "dummy_label_schema",
        ),
    )
    mocker.patch(
        "otx.algorithms.action.tools.sample_classification.create",
        return_value=Config({"learning_parameters": {"num_iters": 4}}),
    )
    mocker.patch("otx.algorithms.action.tools.sample_classification.TaskEnvironment", side_effect=MockTaskEnvironment)
    mocker.patch("otx.algorithms.action.tools.sample_classification.get_task_class", return_value=MockTestCls)
    main(MockArgs())

    MockArgs.export = True
    main(MockArgs())
