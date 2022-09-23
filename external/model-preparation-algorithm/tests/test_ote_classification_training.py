# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import logging
import os
import os.path as osp
from copy import deepcopy
from pprint import pformat
from typing import Any, Callable, Dict, List, Optional, Type

import pytest
from ote_sdk.test_suite.e2e_test_system import DataCollector, e2e_pytest_performance
from ote_sdk.test_suite.training_test_case import (
    OTETestCaseInterface,
    generate_ote_integration_test_case_class,
)
from ote_sdk.test_suite.training_tests_common import (
    KEEP_CONFIG_FIELD_VALUE,
    REALLIFE_USECASE_CONSTANT,
    ROOT_PATH_KEY,
    make_path_be_abs,
)
from ote_sdk.test_suite.training_tests_helper import (
    DefaultOTETestCreationParametersInterface,
    OTETestHelper,
    OTETrainingTestInterface,
)

from tests.mpa_common import (
    _create_classification_dataset_and_labels_schema,
    _get_dataset_params_from_dataset_definitions,
    get_test_action_classes,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def ote_test_domain_fx():
    return "custom-classification-cls-incr"


class ClassificationClsIncrTrainingTestParameters(DefaultOTETestCreationParametersInterface):
    def test_case_class(self) -> Type[OTETestCaseInterface]:
        return generate_ote_integration_test_case_class(get_test_action_classes())

    def test_bunches(self) -> List[Dict[str, Any]]:
        test_bunches = [
            dict(
                model_name=[
                    "Custom_Image_Classification_EfficinetNet-B0",
                    "Custom_Image_Classification_EfficientNet-V2-S",
                    "Custom_Image_Classification_MobileNet-V3-large-1x",
                ],
                dataset_name=["cifar10_cls_incr"],
                usecase="precommit",
            ),
            dict(
                model_name=[
                    "Custom_Image_Classification_EfficinetNet-B0",
                    "Custom_Image_Classification_EfficientNet-V2-S",
                    "Custom_Image_Classification_MobileNet-V3-large-1x",
                ],
                dataset_name=["cifar10_cls_incr"],
                num_training_iters=KEEP_CONFIG_FIELD_VALUE,
                batch_size=KEEP_CONFIG_FIELD_VALUE,
                usecase=REALLIFE_USECASE_CONSTANT,
            ),
        ]

        return deepcopy(test_bunches)

    def default_test_parameters(self) -> Dict[str, Any]:
        DEFAULT_TEST_PARAMETERS = {
            "num_training_iters": 2,
            "batch_size": 16,
        }
        return deepcopy(DEFAULT_TEST_PARAMETERS)


class TestOTEReallifeMPACls(OTETrainingTestInterface):
    """
    The main class of running test in this file.
    """

    PERFORMANCE_RESULTS = None  # it is required for e2e system
    helper = OTETestHelper(ClassificationClsIncrTrainingTestParameters())

    @classmethod
    def get_list_of_tests(cls, usecase: Optional[str] = None):
        """
        This method should be a classmethod. It is called before fixture initialization, during
        tests discovering.
        """
        return cls.helper.get_list_of_tests(usecase)

    @pytest.fixture
    def params_factories_for_test_actions_fx(
        self,
        current_test_parameters_fx,
        dataset_definitions_fx,
        template_paths_fx,
        ote_current_reference_dir_fx,
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
            num_training_iters = test_parameters["num_training_iters"]
            batch_size = test_parameters["batch_size"]

            dataset_params = _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name)

            if model_name not in template_paths:
                raise ValueError(
                    f"Model {model_name} is absent in template_paths, "
                    f"template_paths.keys={list(template_paths.keys())}"
                )
            template_path = make_path_be_abs(template_paths[model_name], template_paths[ROOT_PATH_KEY])

            logger.debug("training params factory: Before creating dataset and labels_schema")
            dataset, labels_schema = _create_classification_dataset_and_labels_schema(dataset_params, model_name)
            ckpt_path = None
            if hasattr(dataset_params, "pre_trained_model"):
                ckpt_path = osp.join(
                    osp.join(dataset_params.pre_trained_model, model_name),
                    "weights.pth",
                )
            logger.info(f"Pretrained path : {ckpt_path}")
            logger.debug("training params factory: After creating dataset and labels_schema")

            return {
                "dataset": dataset,
                "labels_schema": labels_schema,
                "template_path": template_path,
                "num_training_iters": num_training_iters,
                "batch_size": batch_size,
                "checkpoint": ckpt_path,
            }

        params_factories_for_test_actions = {
            "training": _training_params_factory,
        }
        logger.debug("params_factories_for_test_actions_fx: end")
        return params_factories_for_test_actions

    @pytest.fixture
    def test_case_fx(self, current_test_parameters_fx, params_factories_for_test_actions_fx):
        """
        This fixture returns the test case class OTEIntegrationTestCase that should be used for the current test.
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

    # TODO: move to common fixtures
    @pytest.fixture
    def data_collector_fx(self, request) -> DataCollector:
        setup = deepcopy(request.node.callspec.params)
        setup["environment_name"] = os.environ.get("TT_ENVIRONMENT_NAME", "no-env")
        setup["test_type"] = os.environ.get("TT_TEST_TYPE", "no-test-type")  # TODO: get from e2e test type
        setup["scenario"] = "api"  # TODO(lbeynens): get from a fixture!
        setup["test"] = request.node.name
        setup["subject"] = "custom-classification-cls-incr"
        setup["project"] = "ote"
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
        data_collector = DataCollector(name="TestOTEIntegration", setup=setup)
        with data_collector:
            logger.info("data_collector is created")
            yield data_collector
        logger.info("data_collector is released")

    @e2e_pytest_performance
    def test(
        self,
        test_parameters,
        test_case_fx,
        data_collector_fx,
        cur_test_expected_metrics_callback_fx,
    ):
        test_case_fx.run_stage(
            test_parameters["test_stage"],
            data_collector_fx,
            cur_test_expected_metrics_callback_fx,
        )
