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
from collections import OrderedDict
from copy import deepcopy
from pprint import pformat
from typing import Any, Callable, Dict, List, Optional, Type

import pytest

from otx.api.configuration.helper import create as otx_sdk_configuration_helper_create
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import TaskType, parse_model_template
from otx.api.entities.subset import Subset
from otx.api.entities.train_parameters import TrainParameters
from tests.regression.anomaly.anomaly_common import (
    _create_anomaly_dataset_and_labels_schema,
    _get_dataset_params_from_dataset_definitions,
    get_anomaly_domain_test_action_classes,
    get_dummy_compressed_model,
)
from tests.test_suite.e2e_test_system import DataCollector, e2e_pytest_performance
from tests.test_suite.training_test_case import (
    OTXTestCaseInterface,
    generate_otx_integration_test_case_class,
)
from tests.test_suite.training_tests_actions import (
    OTXTestTrainingAction,
    create_environment_and_task,
)
from tests.test_suite.training_tests_common import (
    KEEP_CONFIG_FIELD_VALUE,
    REALLIFE_USECASE_CONSTANT,
    ROOT_PATH_KEY,
    make_path_be_abs,
    performance_to_score_name_value,
)
from tests.test_suite.training_tests_helper import (
    DefaultOTXTestCreationParametersInterface,
    OTXTestHelper,
    OTXTrainingTestInterface,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def otx_test_domain_fx():
    return "custom-anomaly-segmentation"


class AnomalySegmentationTrainingTestParameters(DefaultOTXTestCreationParametersInterface):
    def test_case_class(self) -> Type[OTXTestCaseInterface]:
        return generate_otx_integration_test_case_class(
            get_anomaly_domain_test_action_classes(AnomalySegmentationTestTrainingAction)
        )

    def test_bunches(self) -> List[Dict[str, Any]]:
        # Extend with other datasets
        test_bunches = [
            dict(
                model_name=[
                    "otx_anomaly_segmentation_padim",
                    "otx_anomaly_segmentation_stfpm",
                ],
                dataset_name="mvtec_short_bottle",
                usecase="precommit",
            ),
            dict(
                model_name=[
                    "otx_anomaly_segmentation_padim",
                    "otx_anomaly_segmentation_stfpm",
                ],
                dataset_name="mvtec_short_bottle",
                patience=KEEP_CONFIG_FIELD_VALUE,
                batch_size=KEEP_CONFIG_FIELD_VALUE,
                usecase=REALLIFE_USECASE_CONSTANT,
            ),
            dict(
                model_name=[
                    "otx_anomaly_segmentation_padim",
                    "otx_anomaly_segmentation_stfpm",
                ],
                dataset_name="mvtec_short_cable",
                patience=KEEP_CONFIG_FIELD_VALUE,
                batch_size=KEEP_CONFIG_FIELD_VALUE,
                usecase=REALLIFE_USECASE_CONSTANT,
            ),
            dict(
                model_name=[
                    "otx_anomaly_segmentation_padim",
                    "otx_anomaly_segmentation_stfpm",
                ],
                dataset_name="mvtec_short_capsule",
                patience=KEEP_CONFIG_FIELD_VALUE,
                batch_size=KEEP_CONFIG_FIELD_VALUE,
                usecase=REALLIFE_USECASE_CONSTANT,
            ),
            dict(
                model_name=[
                    "otx_anomaly_segmentation_padim",
                    "otx_anomaly_segmentation_stfpm",
                ],
                dataset_name="mvtec_short_carpet",
                patience=KEEP_CONFIG_FIELD_VALUE,
                batch_size=KEEP_CONFIG_FIELD_VALUE,
                usecase=REALLIFE_USECASE_CONSTANT,
            ),
            dict(
                model_name=[
                    "otx_anomaly_segmentation_padim",
                    "otx_anomaly_segmentation_stfpm",
                ],
                dataset_name="mvtec_short_grid",
                patience=KEEP_CONFIG_FIELD_VALUE,
                batch_size=KEEP_CONFIG_FIELD_VALUE,
                usecase=REALLIFE_USECASE_CONSTANT,
            ),
            dict(
                model_name=[
                    "otx_anomaly_segmentation_padim",
                    "otx_anomaly_segmentation_stfpm",
                ],
                dataset_name="mvtec_short_hazelnut",
                patience=KEEP_CONFIG_FIELD_VALUE,
                batch_size=KEEP_CONFIG_FIELD_VALUE,
                usecase=REALLIFE_USECASE_CONSTANT,
            ),
            dict(
                model_name=[
                    "otx_anomaly_segmentation_padim",
                    "otx_anomaly_segmentation_stfpm",
                ],
                dataset_name="mvtec_short_leather",
                patience=KEEP_CONFIG_FIELD_VALUE,
                batch_size=KEEP_CONFIG_FIELD_VALUE,
                usecase=REALLIFE_USECASE_CONSTANT,
            ),
            dict(
                model_name=[
                    "otx_anomaly_segmentation_padim",
                    "otx_anomaly_segmentation_stfpm",
                ],
                dataset_name="mvtec_short_metal_nut",
                patience=KEEP_CONFIG_FIELD_VALUE,
                batch_size=KEEP_CONFIG_FIELD_VALUE,
                usecase=REALLIFE_USECASE_CONSTANT,
            ),
            dict(
                model_name=[
                    "otx_anomaly_segmentation_padim",
                    "otx_anomaly_segmentation_stfpm",
                ],
                dataset_name="mvtec_short_pill",
                patience=KEEP_CONFIG_FIELD_VALUE,
                batch_size=KEEP_CONFIG_FIELD_VALUE,
                usecase=REALLIFE_USECASE_CONSTANT,
            ),
            dict(
                model_name=[
                    "otx_anomaly_segmentation_padim",
                    "otx_anomaly_segmentation_stfpm",
                ],
                dataset_name="mvtec_short_screw",
                patience=KEEP_CONFIG_FIELD_VALUE,
                batch_size=KEEP_CONFIG_FIELD_VALUE,
                usecase=REALLIFE_USECASE_CONSTANT,
            ),
            dict(
                model_name=[
                    "otx_anomaly_segmentation_padim",
                    "otx_anomaly_segmentation_stfpm",
                ],
                dataset_name="mvtec_short_tile",
                patience=KEEP_CONFIG_FIELD_VALUE,
                batch_size=KEEP_CONFIG_FIELD_VALUE,
                usecase=REALLIFE_USECASE_CONSTANT,
            ),
            dict(
                model_name=[
                    "otx_anomaly_segmentation_padim",
                    "otx_anomaly_segmentation_stfpm",
                ],
                dataset_name="mvtec_short_toothbrush",
                patience=KEEP_CONFIG_FIELD_VALUE,
                batch_size=KEEP_CONFIG_FIELD_VALUE,
                usecase=REALLIFE_USECASE_CONSTANT,
            ),
            dict(
                model_name=[
                    "otx_anomaly_segmentation_padim",
                    "otx_anomaly_segmentation_stfpm",
                ],
                dataset_name="mvtec_short_transistor",
                patience=KEEP_CONFIG_FIELD_VALUE,
                batch_size=KEEP_CONFIG_FIELD_VALUE,
                usecase=REALLIFE_USECASE_CONSTANT,
            ),
            dict(
                model_name=[
                    "otx_anomaly_segmentation_padim",
                    "otx_anomaly_segmentation_stfpm",
                ],
                dataset_name="mvtec_short_wood",
                patience=KEEP_CONFIG_FIELD_VALUE,
                batch_size=KEEP_CONFIG_FIELD_VALUE,
                usecase=REALLIFE_USECASE_CONSTANT,
            ),
            dict(
                model_name=[
                    "otx_anomaly_segmentation_padim",
                    "otx_anomaly_segmentation_stfpm",
                ],
                dataset_name="mvtec_short_zipper",
                patience=KEEP_CONFIG_FIELD_VALUE,
                batch_size=KEEP_CONFIG_FIELD_VALUE,
                usecase=REALLIFE_USECASE_CONSTANT,
            ),
        ]
        return deepcopy(test_bunches)

    def short_test_parameters_names_for_generating_id(self) -> OrderedDict:
        DEFAULT_SHORT_TEST_PARAMETERS_NAMES_FOR_GENERATING_ID = OrderedDict(
            [
                ("test_stage", "ACTION"),
                ("model_name", "model"),
                ("dataset_name", "dataset"),
                ("patience", "patience"),
                ("batch_size", "batch"),
                ("usecase", "usecase"),
            ]
        )
        return deepcopy(DEFAULT_SHORT_TEST_PARAMETERS_NAMES_FOR_GENERATING_ID)

    def test_parameters_defining_test_case_behavior(self) -> List[str]:
        DEFAULT_TEST_PARAMETERS_DEFINING_IMPL_BEHAVIOR = [
            "model_name",
            "dataset_name",
            "patience",
            "batch_size",
        ]
        return deepcopy(DEFAULT_TEST_PARAMETERS_DEFINING_IMPL_BEHAVIOR)

    def default_test_parameters(self) -> Dict[str, Any]:
        DEFAULT_TEST_PARAMETERS = {
            "patience": 1,
            "batch_size": 2,
        }
        return deepcopy(DEFAULT_TEST_PARAMETERS)


class AnomalySegmentationTestTrainingAction(OTXTestTrainingAction):
    _name = "training"

    def __init__(
        self,
        dataset: DatasetEntity,
        labels_schema: LabelSchemaEntity(),
        template_path: str,
        patience: str,
        batch_size: str,
    ):
        self.dataset = dataset
        self.labels_schema = labels_schema
        self.template_path = template_path
        self.num_training_iters = patience
        self.batch_size = batch_size

    def _get_training_performance_as_score_name_value(self):
        training_performance = getattr(self.output_model, "performance", None)
        if training_performance is None:
            raise RuntimeError("Cannot get training performance")
        return performance_to_score_name_value(training_performance)

    def _run_otx_training(self, data_collector: DataCollector):
        logger.debug(f"self.template_path = {self.template_path}")

        print(f"train dataset: {len(self.dataset.get_subset(Subset.TRAINING))} items")
        print(f"validation dataset: " f"{len(self.dataset.get_subset(Subset.VALIDATION))} items")

        logger.debug("Load model template")
        self.model_template = parse_model_template(self.template_path)

        logger.debug("Set hyperparameters")
        params = otx_sdk_configuration_helper_create(self.model_template.hyper_parameters.data)
        if hasattr(params, "model") and hasattr(params.learning_parameters, "early_stopping"):
            if self.num_training_iters != KEEP_CONFIG_FIELD_VALUE:
                params.learning_parameters.early_stopping.patience = int(self.num_training_iters)
                logger.debug(
                    f"Set params.learning_parameters.early_stopping.patience="
                    f"{params.learning_parameters.early_stopping.patience}"
                )
            else:
                logger.debug(
                    f"Keep params.learning_parameters.early_stopping.patience="
                    f"{params.learning_parameters.early_stopping.patience}"
                )
        if self.batch_size != KEEP_CONFIG_FIELD_VALUE:
            params.learning_parameters.train_batch_size = int(self.batch_size)
            logger.debug(
                f"Set params.learning_parameters.train_batch_size=" f"{params.learning_parameters.train_batch_size}"
            )
        else:
            logger.debug(
                f"Keep params.learning_parameters.train_batch_size=" f"{params.learning_parameters.train_batch_size}"
            )

        logger.debug("Setup environment")
        self.environment, self.task = create_environment_and_task(params, self.labels_schema, self.model_template)

        logger.debug("Train model")
        self.output_model = ModelEntity(
            self.dataset,
            self.environment.get_model_configuration(),
        )

        self.copy_hyperparams = deepcopy(self.task.task_environment.get_hyper_parameters())

        try:
            self.task.train(self.dataset, self.output_model, TrainParameters)
        except Exception as ex:
            raise RuntimeError("Training failed") from ex

        score_name, score_value = self._get_training_performance_as_score_name_value()
        logger.info(f"performance={self.output_model.performance}")
        data_collector.log_final_metric("metric_name", self.name + "/" + score_name)
        data_collector.log_final_metric("metric_value", score_value)

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._run_otx_training(data_collector)
        results = {
            "model_template": self.model_template,
            "task": self.task,
            "dataset": self.dataset,
            "environment": self.environment,
            "output_model": self.output_model,
        }
        return results


class TestOTXReallifeAnomalySegmentation(OTXTrainingTestInterface):
    """
    The main class of running test in this file.
    """

    PERFORMANCE_RESULTS = None  # it is required for e2e system
    helper = OTXTestHelper(AnomalySegmentationTrainingTestParameters())

    @classmethod
    def get_list_of_tests(cls, usecase: Optional[str] = None):
        """
        This method should be a classmethod. It is called before fixture initialization, during
        tests discovering.
        """
        return cls.helper.get_list_of_tests(usecase)

    @pytest.fixture
    def params_factories_for_test_actions_fx(
        self, current_test_parameters_fx, dataset_definitions_fx, otx_current_reference_dir_fx, template_paths_fx
    ) -> Dict[str, Callable[[], Dict]]:
        logger.debug("params_factories_for_test_actions_fx: begin")

        test_parameters = deepcopy(current_test_parameters_fx)
        dataset_definitions = deepcopy(dataset_definitions_fx)
        template_paths = deepcopy(template_paths_fx)

        def _training_params_factory() -> Dict:
            if dataset_definitions is None:
                pytest.skip('The parameter "--dataset-definitions" is not set')
            model_name = test_parameters["model_name"]
            dataset_name = test_parameters["dataset_name"]
            patience = test_parameters["patience"]
            batch_size = test_parameters["batch_size"]
            dataset_params = _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name)

            if model_name not in template_paths:
                raise ValueError(
                    f"Model {model_name} is absent in template_paths, "
                    f"template_paths.keys={list(template_paths.keys())}"
                )
            template_path = make_path_be_abs(template_paths[model_name], template_paths[ROOT_PATH_KEY])
            logger.debug("training params factory: Before creating dataset and labels_schema")
            dataset, labels_schema = _create_anomaly_dataset_and_labels_schema(
                dataset_params, dataset_name, TaskType.ANOMALY_SEGMENTATION
            )
            logger.debug("training params factory: After creating dataset and labels_schema")
            return {
                "dataset": dataset,
                "labels_schema": labels_schema,
                "template_path": template_path,
                "patience": patience,
                "batch_size": batch_size,
            }

        def _nncf_graph_params_factory() -> Dict[str, Callable[[], Dict]]:
            if dataset_definitions is None:
                pytest.skip('The parameter "--dataset-definitions" is not set')

            model_name = test_parameters["model_name"]
            dataset_name = test_parameters["dataset_name"]

            dataset_params = _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name)

            if model_name not in template_paths:
                raise ValueError(
                    f"Model {model_name} is absent in template_paths, "
                    f"template_paths.keys={list(template_paths.keys())}"
                )
            template_path = make_path_be_abs(template_paths[model_name], template_paths[ROOT_PATH_KEY])

            logger.debug("training params factory: Before creating dataset and labels_schema")
            dataset, labels_schema = _create_anomaly_dataset_and_labels_schema(
                dataset_params, dataset_name, TaskType.ANOMALY_SEGMENTATION
            )
            logger.debug("training params factory: After creating dataset and labels_schema")

            return {
                "dataset": dataset,
                "labels_schema": labels_schema,
                "template_path": template_path,
                "reference_dir": otx_current_reference_dir_fx,
                "fn_get_compressed_model": get_dummy_compressed_model,
            }

        def _reference_params_factory() -> Dict:
            return {"reference_dir": otx_current_reference_dir_fx}

        params_factories_for_test_actions = {
            "training": _training_params_factory,
            "nncf_graph": _nncf_graph_params_factory,
            "nncf_validate_fq": _reference_params_factory,
            "pot_validate_fq": _reference_params_factory,
        }
        logger.debug("params_factories_for_test_actions_fx: end")
        return params_factories_for_test_actions

    @pytest.fixture
    def test_case_fx(self, current_test_parameters_fx, params_factories_for_test_actions_fx):
        """
        This fixture returns the test case class OTXIntegrationTestCase that should be used for the current test.
        Note that the cache from the test helper allows to store the instance of the class
        between the tests.
        If the main parameters used for this test are the same as the main parameters used for the previous test,
        the instance of the test case class will be kept and re-used. It is helpful for tests that can
        re-use the result of operations (model training, model optimization, etc) made for the previous tests,
        if these operations are time-consuming.
        If the main parameters used for this test differs w.r.t. the previous test, a new instance of
        test case class will be created.
        """
        test_case = type(self).helper.get_test_case(current_test_parameters_fx, params_factories_for_test_actions_fx)
        return test_case

    @pytest.fixture
    def data_collector_fx(self, request) -> DataCollector:
        setup = deepcopy(request.node.callspec.params)
        setup["environment_name"] = os.environ.get("TT_ENVIRONMENT_NAME", "no-env")
        setup["test_type"] = os.environ.get("TT_TEST_TYPE", "no-test-type")  # TODO: get from e2e test type
        setup["scenario"] = "api"  # TODO(lbeynens): get from a fixture!
        setup["test"] = request.node.name
        setup["subject"] = "custom-anomaly-segmentation"
        setup["project"] = "otx"
        if "test_parameters" in setup:
            assert isinstance(setup["test_parameters"], dict)
            if "dataset_name" not in setup:
                setup["dataset_name"] = setup["test_parameters"].get("dataset_name")
            if "model_name" not in setup:
                setup["model_name"] = setup["test_parameters"].get("model_name")
            if "test_stage" not in setup:
                setup["test_stage"] = setup["test_parameters"].get("test_stage")
            if "usecase" not in setup:
                setup["usecase"] = setup["test_parameters"].get("usecase")
        logger.info(f"creating DataCollector: setup=\n{pformat(setup, width=140)}")
        data_collector = DataCollector(name="TestOTXIntegration", setup=setup)
        with data_collector:
            logger.info("data_collector is created")
            yield data_collector
        logger.info("data_collector is released")

    @e2e_pytest_performance
    def test(self, test_parameters, test_case_fx, data_collector_fx, cur_test_expected_metrics_callback_fx):
        test_case_fx.run_stage(test_parameters["test_stage"], data_collector_fx, cur_test_expected_metrics_callback_fx)
