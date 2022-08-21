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
from collections import namedtuple
from copy import deepcopy
from pprint import pformat
from typing import Any, Callable, Dict, List, Optional, Type

import pytest
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label import Domain
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.subset import Subset

from detection_tasks.extension.datasets.data_utils import load_dataset_items_coco_format

from ote_sdk.test_suite.e2e_test_system import DataCollector, e2e_pytest_performance
from ote_sdk.test_suite.training_tests_common import (make_path_be_abs,
                                                      make_paths_be_abs,
                                                      KEEP_CONFIG_FIELD_VALUE,
                                                      REALLIFE_USECASE_CONSTANT,
                                                      ROOT_PATH_KEY)
from ote_sdk.test_suite.training_tests_helper import (OTETestHelper,
                                                      DefaultOTETestCreationParametersInterface,
                                                      OTETrainingTestInterface)


logger = logging.getLogger(__name__)

def DATASET_PARAMETERS_FIELDS() -> List[str]:
    return deepcopy(['annotations_train',
                     'images_train_dir',
                     'annotations_val',
                     'images_val_dir',
                     'annotations_test',
                     'images_test_dir',
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


def _create_object_detection_dataset_and_labels_schema(dataset_params, domain: Domain):
    logger.debug(f'Using for train annotation file {dataset_params.annotations_train}')
    logger.debug(f'Using for val annotation file {dataset_params.annotations_val}')
    labels_list = []
    items = load_dataset_items_coco_format(
        ann_file_path=dataset_params.annotations_train,
        data_root_dir=dataset_params.images_train_dir,
        domain=domain,
        subset=Subset.TRAINING,
        labels_list=labels_list)
    items.extend(load_dataset_items_coco_format(
        ann_file_path=dataset_params.annotations_val,
        data_root_dir=dataset_params.images_val_dir,
        domain=domain,
        subset=Subset.VALIDATION,
        labels_list=labels_list))
    items.extend(load_dataset_items_coco_format(
        ann_file_path=dataset_params.annotations_test,
        data_root_dir=dataset_params.images_test_dir,
        domain=domain,
        subset=Subset.TESTING,
        labels_list=labels_list))
    dataset = DatasetEntity(items=items)
    labels_schema = LabelSchemaEntity.from_labels(labels_list)
    return dataset, labels_schema


class ObjectDetectionTrainingTestParameters(DefaultOTETestCreationParametersInterface):
    def test_bunches(self) -> List[Dict[str, Any]]:
        test_bunches = [
                dict(
                    model_name=[
                       'Custom_Object_Detection_Gen3_ATSS',
                       'Custom_Object_Detection_Gen3_SSD',
                    ],
                    dataset_name='dataset1_tiled_shortened_500_A',
                    usecase='precommit',
                ),
                dict(
                    model_name=[
                       'Custom_Object_Detection_Gen3_ATSS',
                       'Custom_Object_Detection_Gen3_SSD',
                    ],
                    dataset_name=[
                        'bbcd',
                        'weed-coco',
                        'pcd',
                        'aerial',
                        'dice',
                        'fish',
                        'vitens',
                        'diopsis',
                    ],
                    num_training_iters=KEEP_CONFIG_FIELD_VALUE,
                    batch_size=KEEP_CONFIG_FIELD_VALUE,
                    usecase=REALLIFE_USECASE_CONSTANT,
                ),

        ]
        return deepcopy(test_bunches)


class InstanceSegmentationTrainingTestParameters(DefaultOTETestCreationParametersInterface):
    def test_bunches(self) -> List[Dict[str, Any]]:
        test_bunches = [
                dict(
                    model_name=[
                        'Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50',
                        'Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B',
                    ],
                    dataset_name='aeromonas_short',
                    usecase='precommit',
                ),
                dict(
                    model_name=[
                        'Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50',
                        'Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B',
                    ],
                    dataset_name='aeromonas',
                    num_training_iters=KEEP_CONFIG_FIELD_VALUE,
                    batch_size=KEEP_CONFIG_FIELD_VALUE,
                    usecase=REALLIFE_USECASE_CONSTANT,
                ),

        ]
        return deepcopy(test_bunches)


def get_dummy_compressed_model(task):
    """
    Return compressed model without initialization
    """
    # pylint:disable=protected-access
    from mmdet.integration.nncf import wrap_nncf_model
    from mmdet.apis.fake_input import get_fake_input

    # Disable quantaizers initialization
    for compression in task._config.nncf_config['compression']:
        if compression["algorithm"] == "quantization":
            compression["initializer"] = {
                "batchnorm_adaptation": {
                    "num_bn_adaptation_samples": 0
                }
            }

    _, compressed_model = wrap_nncf_model(task._model,
                                          task._config,
                                          get_fake_input_func=get_fake_input)
    return compressed_model


@pytest.fixture
def params_factories_for_test_actions_fx(current_test_parameters_fx,
                                         dataset_definitions_fx, template_paths_fx,
                                         ote_current_reference_dir_fx) -> Dict[str, Callable[[], Dict]]:
    logger.debug('params_factories_for_test_actions_fx: begin')

    test_parameters = deepcopy(current_test_parameters_fx)
    dataset_definitions = deepcopy(dataset_definitions_fx)
    template_paths = deepcopy(template_paths_fx)

    def _training_params_factory() -> Dict:
        if dataset_definitions is None:
            pytest.skip('The parameter "--dataset-definitions" is not set')

        model_name = test_parameters['model_name']
        if "Custom_Object_Detection" in model_name:
            domain = Domain.DETECTION
        elif "Custom_Counting_Instance_Segmentation" in model_name:
            domain = Domain.INSTANCE_SEGMENTATION
        else:
            domain = None
        dataset_name = test_parameters['dataset_name']
        num_training_iters = test_parameters['num_training_iters']
        batch_size = test_parameters['batch_size']

        dataset_params = _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name)

        if model_name not in template_paths:
            raise ValueError(f'Model {model_name} is absent in template_paths, '
                             f'template_paths.keys={list(template_paths.keys())}')
        template_path = make_path_be_abs(template_paths[model_name], template_paths[ROOT_PATH_KEY])

        logger.debug('training params factory: Before creating dataset and labels_schema')
        dataset, labels_schema = _create_object_detection_dataset_and_labels_schema(
            dataset_params, domain)
        logger.debug('training params factory: After creating dataset and labels_schema')

        return {
            'dataset': dataset,
            'labels_schema': labels_schema,
            'template_path': template_path,
            'num_training_iters': num_training_iters,
            'batch_size': batch_size,
        }

    def _nncf_graph_params_factory() -> Dict:
        if dataset_definitions is None:
            pytest.skip('The parameter "--dataset-definitions" is not set')

        model_name = test_parameters['model_name']
        if "Custom_Object_Detection" in model_name:
            domain = Domain.DETECTION
        elif "Custom_Counting_Instance_Segmentation" in model_name:
            domain = Domain.INSTANCE_SEGMENTATION
        else:
            domain = None
        dataset_name = test_parameters['dataset_name']

        dataset_params = _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name)

        if model_name not in template_paths:
            raise ValueError(f'Model {model_name} is absent in template_paths, '
                             f'template_paths.keys={list(template_paths.keys())}')
        template_path = make_path_be_abs(template_paths[model_name], template_paths[ROOT_PATH_KEY])

        logger.debug('training params factory: Before creating dataset and labels_schema')
        dataset, labels_schema = _create_object_detection_dataset_and_labels_schema(
            dataset_params, domain)
        logger.debug('training params factory: After creating dataset and labels_schema')

        return {
            'dataset': dataset,
            'labels_schema': labels_schema,
            'template_path': template_path,
            'reference_dir': ote_current_reference_dir_fx,
            'fn_get_compressed_model': get_dummy_compressed_model,
        }

    params_factories_for_test_actions = {
        'training': _training_params_factory,
        'nncf_graph': _nncf_graph_params_factory,
    }
    logger.debug('params_factories_for_test_actions_fx: end')
    return params_factories_for_test_actions


# TODO(lbeynens): move to common fixtures
@pytest.fixture
def data_collector_fx(request) -> DataCollector:
    setup = deepcopy(request.node.callspec.params)
    setup['environment_name'] = os.environ.get('TT_ENVIRONMENT_NAME', 'no-env')
    setup['test_type'] = os.environ.get('TT_TEST_TYPE', 'no-test-type') # TODO: get from e2e test type
    setup['scenario'] = 'api' # TODO(lbeynens): get from a fixture!
    setup['test'] = request.node.name
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
    model_name = setup['test_parameters'].get('model_name')
    if "Custom_Object_Detection" in model_name:
        subject = 'custom-object-detection'
    elif "Custom_Counting_Instance_Segmentation" in model_name:
        subject = 'custom-counting-instance-seg'
    else:
        subject = None
    setup['subject'] = subject
    logger.info(f'creating DataCollector: setup=\n{pformat(setup, width=140)}')
    data_collector = DataCollector(name='TestOTEIntegration',
                                   setup=setup)
    with data_collector:
        logger.info('data_collector is created')
        yield data_collector
    logger.info('data_collector is released')


class TestOTEReallifeObjectDetection(OTETrainingTestInterface):
    """
    The main class of running test in this file.
    """
    PERFORMANCE_RESULTS = None # it is required for e2e system
    helper = OTETestHelper(ObjectDetectionTrainingTestParameters())

    @classmethod
    def get_list_of_tests(cls, usecase: Optional[str] = None):
        """
        This method should be a classmethod. It is called before fixture initialization, during
        tests discovering.
        """
        return cls.helper.get_list_of_tests(usecase)

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

    @e2e_pytest_performance
    def test(self,
             test_parameters,
             test_case_fx, data_collector_fx,
             cur_test_expected_metrics_callback_fx):
        test_case_fx.run_stage(test_parameters['test_stage'], data_collector_fx,
                               cur_test_expected_metrics_callback_fx)


class TestInstanceSegmentation(OTETrainingTestInterface):
    """
    The main class of running test in this file.
    """
    PERFORMANCE_RESULTS = None # it is required for e2e system
    helper = OTETestHelper(InstanceSegmentationTrainingTestParameters())

    @classmethod
    def get_list_of_tests(cls, usecase: Optional[str] = None):
        """
        This method should be a classmethod. It is called before fixture initialization, during
        tests discovering.
        """
        return cls.helper.get_list_of_tests(usecase)

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

    @e2e_pytest_performance
    def test(self,
             test_parameters,
             test_case_fx, data_collector_fx,
             cur_test_expected_metrics_callback_fx):
        test_case_fx.run_stage(test_parameters['test_stage'], data_collector_fx,
                               cur_test_expected_metrics_callback_fx)
