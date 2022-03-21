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

import glob
import logging
import os
import os.path as osp
from collections import (namedtuple,
                        OrderedDict)
from copy import deepcopy
from pprint import pformat
from typing import Any, Callable, Dict, List, Optional, Type

import pytest
import yaml
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.train_parameters import TrainParameters

from ote_anomalib.data.mvtec import OteMvtecDataset
from ote_anomalib.logging import get_logger

from ote_sdk.configuration.helper import create as ote_sdk_configuration_helper_create
from ote_sdk.test_suite.training_test_case import (OTETestCaseInterface,
                                                   generate_ote_integration_test_case_class)
from ote_sdk.test_suite.e2e_test_system import DataCollector, e2e_pytest_performance
from ote_sdk.test_suite.training_tests_common import (make_path_be_abs,
                                                      make_paths_be_abs,
                                                      performance_to_score_name_value,
                                                      KEEP_CONFIG_FIELD_VALUE,
                                                      REALLIFE_USECASE_CONSTANT,
                                                      ROOT_PATH_KEY)
from ote_sdk.test_suite.training_tests_helper import (OTETestHelper,
                                                      DefaultOTETestCreationParametersInterface,
                                                      OTETrainingTestInterface)
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

@pytest.fixture
def ote_test_domain_fx():
    return 'anomaly-segmentation'

@pytest.fixture(scope='session')
def ote_templates_root_dir_fx():
    import os.path as osp
    import logging
    logger = logging.getLogger(__name__)
    root = osp.dirname(osp.dirname(osp.realpath(__file__)))
    root = f'{root}/anomaly_segmentation/configs/'
    logger.debug(f'overloaded ote_templates_root_dir_fx: return {root}')
    return root

def DATASET_PARAMETERS_FIELDS() -> List[str]:
    return deepcopy(['dataset_path',
                     ])

DatasetParameters = namedtuple('DatasetParameters', DATASET_PARAMETERS_FIELDS())


def _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name):
    if dataset_name not in dataset_definitions:
        raise ValueError(f'dataset {dataset_name} is absent in dataset_definitions, '
                         f'dataset_definitions.keys={list(dataset_definitions.keys())}')
    cur_dataset_definition = dataset_definitions[dataset_name]
    training_parameters_fields = {k: v for k, v in cur_dataset_definition.items()
                                  if k in DATASET_PARAMETERS_FIELDS()}
    make_paths_be_abs(training_parameters_fields, dataset_definitions[ROOT_PATH_KEY])

    assert set(DATASET_PARAMETERS_FIELDS()) == set(training_parameters_fields.keys()), \
            f'ERROR: dataset definitions for name={dataset_name} does not contain all required fields'
    assert all(training_parameters_fields.values()), \
            f'ERROR: dataset definitions for name={dataset_name} contains empty values for some required fields'

    params = DatasetParameters(**training_parameters_fields)
    return params


def _create_anomaly_segmentation_dataset_and_labels_schema(dataset_params, dataset_name):
    logger.debug(f'Path to dataset: {dataset_params.dataset_path}')
    category_list = [f.path for f in os.scandir(dataset_params.dataset_path) if f.is_dir()]
    items = []
    if "short" in dataset_name:
        logger.debug(f'Creating short dataset {dataset_name}')
        items.extend(OteMvtecDataset(path=dataset_params.dataset_path, seed=0).generate())
    else:
        for category in category_list:
            logger.debug(f'Creating dataset for {category}')
            items.extend(OteMvtecDataset(path=category, seed=0).generate())
    dataset = DatasetEntity(items=items)
    labels = dataset.get_labels()
    labels_schema = LabelSchemaEntity.from_labels(labels)
    return dataset, labels_schema

def get_anomaly_segmentation_test_action_classes() -> List[Type[BaseOTETestAction]]:
    return [
        AnomalySegmentationTestTrainingAction,
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

class AnomalySegmentationTrainingTestParameters(DefaultOTETestCreationParametersInterface):
    def test_case_class(self) -> Type[OTETestCaseInterface]:
        return generate_ote_integration_test_case_class(
            get_anomaly_segmentation_test_action_classes()
        )

    def test_bunches(self) -> List[Dict[str, Any]]:
        # Extend with other datasets
        test_bunches = [
                dict(
                    model_name=[
                       'ote_anomaly_segmentation_padim',
                       'ote_anomaly_segmentation_stfpm',
                    ],
                    dataset_name='mvtec_short',
                    usecase='precommit',
                ),
                dict(
                    model_name=[
                       'ote_anomaly_segmentation_padim',
                       'ote_anomaly_segmentation_stfpm',
                    ],
                    dataset_name='mvtec',
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

# TODO:sstrehlk: This function copies with minor change OTETestTrainingAction
#             from ote_sdk.test_suite.
#             Try to avoid copying of code.
class AnomalySegmentationTestTrainingAction(OTETestTrainingAction):
    _name = "training"

    def __init__(
        self, dataset, labels_schema, template_path, patience, batch_size
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

    def _run_ote_training(self, data_collector):
        logger.debug(f"self.template_path = {self.template_path}")

        print(f"train dataset: {len(self.dataset.get_subset(Subset.TRAINING))} items")
        print(
            f"validation dataset: "
            f"{len(self.dataset.get_subset(Subset.VALIDATION))} items"
        )

        logger.debug("Load model template")
        self.model_template = parse_model_template(self.template_path)

        logger.debug("Set hyperparameters")
        params = ote_sdk_configuration_helper_create(
            self.model_template.hyper_parameters.data
        )
        if hasattr(params, 'model'):
            if self.num_training_iters != KEEP_CONFIG_FIELD_VALUE:
                params.model.early_stopping.patience = int(self.num_training_iters)
                logger.debug(
                    f"Set params.model.early_stopping.patience="
                    f"{params.model.early_stopping.patience}"
                )
            else:
                logger.debug(
                    f"Keep params.model.early_stopping.patience="
                    f"{params.model.early_stopping.patience}"
                )
        if self.batch_size != KEEP_CONFIG_FIELD_VALUE:
            params.dataset.train_batch_size = int(self.batch_size)
            logger.debug(
                f"Set params.dataset.train_batch_size="
                f"{params.dataset.train_batch_size}"
            )
        else:
            logger.debug(
                f"Keep params.dataset.train_batch_size="
                f"{params.dataset.train_batch_size}"
            )

        logger.debug("Setup environment")
        self.environment, self.task = create_environment_and_task(
            params, self.labels_schema, self.model_template
        )

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
        self._run_ote_training(data_collector)
        results = {
            "model_template": self.model_template,
            "task": self.task,
            "dataset": self.dataset,
            "environment": self.environment,
            "output_model": self.output_model,
        }
        return results


class TestOTEReallifeAnomalySegmentation(OTETrainingTestInterface):
    """
    The main class of running test in this file.
    """
    PERFORMANCE_RESULTS = None # it is required for e2e system
    helper = OTETestHelper(AnomalySegmentationTrainingTestParameters())

    @classmethod
    def get_list_of_tests(cls, usecase: Optional[str] = None):
        """
        This method should be a classmethod. It is called before fixture initialization, during
        tests discovering.
        """
        return cls.helper.get_list_of_tests(usecase)

    @pytest.fixture
    def params_factories_for_test_actions_fx(self, current_test_parameters_fx,
                                             dataset_definitions_fx,ote_current_reference_dir_fx,
                                             template_paths_fx) -> Dict[str,Callable[[], Dict]]:
        logger.debug('params_factories_for_test_actions_fx: begin')

        test_parameters = deepcopy(current_test_parameters_fx)
        dataset_definitions = deepcopy(dataset_definitions_fx)
        template_paths = deepcopy(template_paths_fx)
        def _training_params_factory() -> Dict:
            if dataset_definitions is None:
                pytest.skip('The parameter "--dataset-definitions" is not set')
            model_name = test_parameters['model_name']
            dataset_name = test_parameters['dataset_name']
            patience = test_parameters['patience']
            batch_size = test_parameters['batch_size']
            dataset_params = _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name)

            if model_name not in template_paths:
                raise ValueError(f'Model {model_name} is absent in template_paths, '
                                 f'template_paths.keys={list(template_paths.keys())}')
            template_path = make_path_be_abs(template_paths[model_name], template_paths[ROOT_PATH_KEY])
            logger.debug('training params factory: Before creating dataset and labels_schema')
            dataset, labels_schema = _create_anomaly_segmentation_dataset_and_labels_schema(dataset_params, dataset_name)
            logger.debug('training params factory: After creating dataset and labels_schema')
            return {
                'dataset': dataset,
                'labels_schema': labels_schema,
                'template_path': template_path,
                'patience': patience,
                'batch_size': batch_size,
            }

        def _nncf_graph_params_factory() -> Dict:
            if dataset_definitions is None:
                pytest.skip('The parameter "--dataset-definitions" is not set')

            model_name = test_parameters['model_name']
            dataset_name = test_parameters['dataset_name']

            dataset_params = _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name)

            if model_name not in template_paths:
                raise ValueError(f'Model {model_name} is absent in template_paths, '
                                 f'template_paths.keys={list(template_paths.keys())}')
            template_path = make_path_be_abs(template_paths[model_name], template_paths[ROOT_PATH_KEY])

            logger.debug('training params factory: Before creating dataset and labels_schema')
            dataset, labels_schema = _create_anomaly_segmentation_dataset_and_labels_schema(dataset_params, dataset_name)
            logger.debug('training params factory: After creating dataset and labels_schema')

            return {
                'dataset': dataset,
                'labels_schema': labels_schema,
                'template_path': template_path,
                'reference_dir': ote_current_reference_dir_fx,
                'fn_get_compressed_model': None #NNCF not yet implemented in Anomaly
            }

        params_factories_for_test_actions = {
            'training': _training_params_factory,
            'nncf_graph': _nncf_graph_params_factory,
        }
        logger.debug('params_factories_for_test_actions_fx: end')
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
        test_case = type(self).helper.get_test_case(current_test_parameters_fx,
                                                    params_factories_for_test_actions_fx)
        return test_case

    # TODO(lbeynens): move to common fixtures
    @pytest.fixture
    def data_collector_fx(self, request) -> DataCollector:
        setup = deepcopy(request.node.callspec.params)
        setup['environment_name'] = os.environ.get('TT_ENVIRONMENT_NAME', 'no-env')
        setup['test_type'] = os.environ.get('TT_TEST_TYPE', 'no-test-type') # TODO: get from e2e test type
        setup['scenario'] = 'api' # TODO(lbeynens): get from a fixture!
        setup['test'] = request.node.name
        setup['subject'] = 'custom-anomaly-segmentation'
        setup['project'] = 'ote'
        if 'test_parameters' in setup:
            assert isinstance(setup['test_parameters'], dict)
            if 'dataset_name' not in setup:
                setup['dataset_name'] = setup['test_parameters'].get('dataset_name')
            if 'model_name' not in setup:
                setup['model_name'] = setup['test_parameters'].get('model_name')
            if 'test_stage' not in setup:
                setup['test_stage'] = setup['test_parameters'].get('test_stage')
            if 'usecase' not in setup:
                setup['usecase'] = setup['test_parameters'].get('usecase')
        logger.info(f'creating DataCollector: setup=\n{pformat(setup, width=140)}')
        data_collector = DataCollector(name='TestOTEIntegration',
                                       setup=setup)
        with data_collector:
            logger.info('data_collector is created')
            yield data_collector
        logger.info('data_collector is released')

    @e2e_pytest_performance
    def test(self,
             test_parameters,
             test_case_fx, data_collector_fx,
             cur_test_expected_metrics_callback_fx):
        if "nncf" in test_parameters["test_stage"]:
            pytest.xfail("NNCF not yet supported for Anomaly Segmentation")
        test_case_fx.run_stage(test_parameters['test_stage'], data_collector_fx,
                               cur_test_expected_metrics_callback_fx)
