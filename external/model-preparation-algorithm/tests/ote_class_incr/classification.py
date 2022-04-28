# Copyright (C) 2021 Intel Corporation
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
import os.path as osp
from collections import (namedtuple,
                        OrderedDict)
from copy import deepcopy
from pprint import pformat
from typing import Any, Callable, Dict, List, Optional, Type


import pytest
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.configuration.helper import create as ote_sdk_configuration_helper_create

from ote_sdk.test_suite.training_test_case import (OTETestCaseInterface,
                                                   generate_ote_integration_test_case_class)
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_performance
from ote_sdk.test_suite.training_tests_common import (make_path_be_abs,
                                                      make_paths_be_abs,
                                                      DEFAULT_FIELD_VALUE_FOR_USING_IN_TEST,
                                                      KEEP_CONFIG_FIELD_VALUE,
                                                      REALLIFE_USECASE_CONSTANT,
                                                      ROOT_PATH_KEY)
from ote_sdk.test_suite.training_tests_helper import (OTETestHelper,
                                                      DefaultOTETestCreationParametersInterface,
                                                      OTETrainingTestInterface)
from ote_sdk.test_suite.training_tests_actions import (OTETestTrainingAction,
                                                       BaseOTETestAction,
                                                       OTETestTrainingEvaluationAction,
                                                       OTETestExportAction,
                                                       OTETestExportEvaluationAction,
                                                       OTETestPotAction,
                                                       OTETestPotEvaluationAction)

from torchreid_tasks.utils import ClassificationDatasetAdapter

logger = logging.getLogger(__name__)

def DATASET_PARAMETERS_FIELDS() -> List[str]:
    return deepcopy(['annotations_train',
                     'images_train_dir',
                     'annotations_val',
                     'images_val_dir',
                     'annotations_test',
                     'images_test_dir',
                     'pre_trained_model'
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

def _create_classification_dataset_and_labels_schema(dataset_params, model_name):
    logger.debug(f'Using for train annotation file {dataset_params.annotations_train}')
    logger.debug(f'Using for val annotation file {dataset_params.annotations_val}')

    dataset = ClassificationDatasetAdapter(
        train_data_root=osp.join(dataset_params.images_train_dir),
        train_ann_file=osp.join(dataset_params.annotations_train),
        val_data_root=osp.join(dataset_params.images_val_dir),
        val_ann_file=osp.join(dataset_params.annotations_val),
        test_data_root=osp.join(dataset_params.images_test_dir),
        test_ann_file=osp.join(dataset_params.annotations_test))
    
    labels_schema = LabelSchemaEntity.from_labels(dataset.get_labels())
    return dataset, labels_schema

def get_image_classification_test_action_classes() -> List[Type[BaseOTETestAction]]:
    return [
        OTETestTrainingAction,
        OTETestTrainingEvaluationAction,
        OTETestExportAction,
        OTETestExportEvaluationAction,
        OTETestPotAction,
        OTETestPotEvaluationAction,
    ]

class ClassificationTrainingTestParameters(DefaultOTETestCreationParametersInterface):
    def test_case_class(self) -> Type[OTETestCaseInterface]:
        return generate_ote_integration_test_case_class(
            get_image_classification_test_action_classes()
        )

    def test_bunches(self) -> List[Dict[str, Any]]:
        test_bunches = [
                dict(
                    model_name=[
                       'ClassIncremental_Image_Classification_EfficinetNet-B0',
                    ],
                    dataset_name=['cifar10_airplane_automobile_bird_cat_deer_frog'],
                    usecase='precommit',
                ),
                dict(
                    model_name=[
                       'ClassIncremental_Image_Classification_EfficinetNet-B0',
                    ],
                    dataset_name=['cifar10_airplane_automobile_bird_cat_deer_frog'],
                    num_training_iters=KEEP_CONFIG_FIELD_VALUE, 
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
                ("num_training_iters", "num_iters"),
                ("batch_size", "batch"),
                ("usecase", "usecase"),
            ]
        )
        return deepcopy(DEFAULT_SHORT_TEST_PARAMETERS_NAMES_FOR_GENERATING_ID)

    def test_parameters_defining_test_case_behavior(self) -> List[str]:
        DEFAULT_TEST_PARAMETERS_DEFINING_IMPL_BEHAVIOR = [
            "model_name",
            "dataset_name",
            "num_training_iters",
            "batch_size",
        ] # this needs to distinguish the test case -> transition in helper.cache -> transition in new stages(_OTEIntegrationTestCase)
        return deepcopy(DEFAULT_TEST_PARAMETERS_DEFINING_IMPL_BEHAVIOR)

    def default_test_parameters(self) -> Dict[str, Any]:
        DEFAULT_TEST_PARAMETERS = {
            "num_training_iters": 2,
            "batch_size": 16,
        } # the mandatory params for running test
        return deepcopy(DEFAULT_TEST_PARAMETERS)

class TestOTEReallifeClassification(OTETrainingTestInterface):
    """
    The main class of running test in this file.
    """
    PERFORMANCE_RESULTS = None # it is required for e2e system
    helper = OTETestHelper(ClassificationTrainingTestParameters())

    @classmethod
    def get_list_of_tests(cls, usecase: Optional[str] = None):
        """
        This method should be a classmethod. It is called before fixture initialization, during
        tests discovering.
        """
        return cls.helper.get_list_of_tests(usecase)

    @pytest.fixture
    def params_factories_for_test_actions_fx(self, current_test_parameters_fx,
                                             dataset_definitions_fx, template_paths_fx,
                                             ote_current_reference_dir_fx) -> Dict[str,Callable[[], Dict]]:
        logger.debug('params_factories_for_test_actions_fx: begin')

        test_parameters = deepcopy(current_test_parameters_fx)
        dataset_definitions = deepcopy(dataset_definitions_fx)
        template_paths = deepcopy(template_paths_fx)
        def _training_params_factory() -> Dict:
            if dataset_definitions is None:
                pytest.skip('The parameter "--dataset-definitions" is not set')

            model_name = test_parameters['model_name']
            dataset_name = test_parameters['dataset_name']
            num_training_iters = test_parameters['num_training_iters']
            batch_size = test_parameters['batch_size']

            dataset_params = _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name)

            if model_name not in template_paths:
                raise ValueError(f'Model {model_name} is absent in template_paths, '
                                 f'template_paths.keys={list(template_paths.keys())}')
            template_path = make_path_be_abs(template_paths[model_name], template_paths[ROOT_PATH_KEY])

            logger.debug('training params factory: Before creating dataset and labels_schema')
            dataset, labels_schema = _create_classification_dataset_and_labels_schema(dataset_params, model_name)
            ckpt_path = None
            if hasattr(dataset_params, 'pre_trained_model'):
                ckpt_path = osp.join(osp.join(dataset_params.pre_trained_model, model_name),"weights.pth")
            logger.info(f"Pretrained path : {ckpt_path}")
            logger.debug('training params factory: After creating dataset and labels_schema')

            return {
                'dataset': dataset,
                'labels_schema': labels_schema,
                'template_path': template_path,
                'num_training_iters': num_training_iters,
                'batch_size': batch_size,
                'checkpoint': ckpt_path
            }
        params_factories_for_test_actions = {
            'training': _training_params_factory, #_name of OTETestTrainingAction is 'training' -> rest of action classes doesn't need param(it will get it from the previous action classes)
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
                                                    params_factories_for_test_actions_fx) # this needs when test case is changed - give test cases as params to _OTEIntegrationTestCase -> each action will be initialized
        return test_case #_OTEIntegrationTestCase object (constructed stages including action classes initialized by params_factories)

    @e2e_pytest_performance
    def test(self,
             test_parameters,
             test_case_fx, data_collector_fx,
             cur_test_expected_metrics_callback_fx):
        test_case_fx.run_stage(test_parameters['test_stage'], data_collector_fx,
                               cur_test_expected_metrics_callback_fx)
