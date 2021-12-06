# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
The file contains fixtures that may be used in algo backend's
reallife training tests.

Note that the fixtures ote_templates_root_dir_fx and ote_test_domain_fx
MUST be overriden in algo backend's conftest.py file.
"""

# pylint: disable=redefined-outer-name

import glob
import logging
import os
import os.path as osp
from copy import deepcopy
from pprint import pformat
from typing import Callable, Dict, Optional

import pytest
import yaml

from .e2e_test_system import DataCollector
from .training_tests_common import REALLIFE_USECASE_CONSTANT

logger = logging.getLogger(__name__)

#########################################################################################
# Fixtures that should be overriden in algo backends


@pytest.fixture(scope="session")
def ote_templates_root_dir_fx():
    """
    The fixture returns an absolute path to the folder where (in the subfolders)
    the reallife training tests will look OTE template files (the files 'template.yaml').

    The fixture MUST be overriden in algo backend's conftest.py file.
    """
    raise NotImplementedError(
        "The fixture ote_templates_root_dir_fx should be overriden in algo backend"
    )


@pytest.fixture
def ote_test_domain_fx():
    """
    The fixture returns an string that will be used as the 'subject' field in the
    e2e test system dashboard.
    At the moment it is supposed that the fixture should returns something like
    'custom-object-detection'.

    The fixture MUST be overriden in algo backend's conftest.py file.
    """
    raise NotImplementedError(
        "The fixture ote_test_domain_fx should be overriden in algo backend"
    )


@pytest.fixture
def ote_test_scenario_fx():
    """
    The fixture returns an string that will be used as the 'scenario' field in the
    e2e test system dashboard.
    At the moment it is supposed that the fixture should returns something like
    'api' or 'integration' or 'reallife'.

    The fixture may be overriden in algo backend's conftest.py file.
    """
    return "api"


#
#########################################################################################

# pylint: disable=invalid-name


def ROOT_PATH_KEY():
    """
    Constant for storying in dict-s with paths the root path
    that will be used for resolving relative paths.
    """
    return "_root_path"


@pytest.fixture
def dataset_definitions_fx(request):
    """
    Return dataset definitions read from a YAML file passed as the parameter --dataset-definitions.
    Note that the dataset definitions should store the following structure:
    {
        <dataset_name>: {
            'annotations_train': <annotation_file_path1>
            'images_train_dir': <images_folder_path1>
            'annotations_val': <annotation_file_path2>
            'images_val_dir': <images_folder_path2>
            'annotations_test': <annotation_file_path3>
            'images_test_dir':  <images_folder_path3>
        }
    }
    """
    path = request.config.getoption("--dataset-definitions")
    if path is None:
        logger.warning(
            f"The command line parameter '--dataset-definitions' is not set"
            f"whereas it is required for the test {request.node.originalname or request.node.name}"
            f" -- ALL THE TESTS THAT REQUIRE THIS PARAMETER ARE SKIPPED"
        )
        return None
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data[ROOT_PATH_KEY()] = osp.dirname(path)
    return data


@pytest.fixture(scope="session")
def template_paths_fx(ote_templates_root_dir_fx):
    """
    Return mapping model names to template paths, received from globbing the
    folder pointed by the fixture ote_templates_root_dir_fx.
    Note that the function searches files with name `template.yaml`, and for each such file
    the model name is the name of the parent folder of the file.
    """
    root = ote_templates_root_dir_fx
    assert osp.isabs(
        root
    ), f"Error: ote_templates_root_dir_fx is not an absolute path: {root}"
    glb = glob.glob(f"{root}/**/template.yaml", recursive=True)
    data = {}
    for p in glb:
        assert osp.isabs(p), f"Error: not absolute path {p}"
        name = osp.basename(osp.dirname(p))
        if name in data:
            raise RuntimeError(
                f"Duplication of names in {root} folder: {data[name]} and {p}"
            )
        data[name] = p
    data[ROOT_PATH_KEY()] = ""
    return data


@pytest.fixture
def expected_metrics_all_tests_fx(request):
    # pylint: disable=line-too-long
    """
    Return expected metrics for reallife tests read from a YAML file passed as the parameter --expected-metrics-file.
    Note that the structure of expected metrics should be a dict that maps tests to the expected metric numbers.
    The keys of the dict are the parameters' part of the test id-s -- see the function
    OTETestHelper._generate_test_id, also see the fixture current_test_parameters_string_fx below.

    The value for each key is a structure that stores a requirement on some metric.
    The requirement can be either a target value (probably, with max size of quality drop)
    or the reference to another stage of the same model (also probably with max size of quality drop).
    See details in the description of the fixture cur_test_expected_metrics_callback_fx below.
    E.g.
    ```
    'ACTION-training_evaluation,model-gen3_mobilenetV2_ATSS,dataset-bbcd,num_iters-KEEP_CONFIG_FIELD_VALUE,batch-KEEP_CONFIG_FIELD_VALUE,usecase-reallife':
        'metrics.accuracy.f-measure':
            'target_value': 0.81
            'max_diff': 0.005
    'ACTION-export_evaluation,model-gen3_mobilenetV2_ATSS,dataset-bbcd,num_iters-KEEP_CONFIG_FIELD_VALUE,batch-KEEP_CONFIG_FIELD_VALUE,usecase-reallife':
        'metrics.accuracy.f-measure':
            'base': 'training_evaluation.metrics.accuracy.f-measure'
            'max_diff': 0.01
    ```
    """
    path = request.config.getoption("--expected-metrics-file")
    if path is None:
        logger.warning(
            "The command line parameter '--expected-metrics-file' is not set"
            "whereas it is required to compare with target metrics"
            " -- ALL THE COMPARISON WITH TARGET METRICS IN TESTS WILL BE FAILED"
        )
        return None
    with open(path, encoding="utf-8") as f:
        expected_metrics_all_tests = yaml.safe_load(f)
    assert isinstance(
        expected_metrics_all_tests, dict
    ), f"Wrong metrics file {path}: {expected_metrics_all_tests}"
    return expected_metrics_all_tests


@pytest.fixture
def current_test_parameters_fx(request):
    """
    This fixture returns the test parameter `test_parameters` of the current test.
    """
    cur_test_params = deepcopy(request.node.callspec.params)
    assert "test_parameters" in cur_test_params, (
        f"The test {request.node.name} should be parametrized "
        f"by parameter 'test_parameters'"
    )
    return cur_test_params["test_parameters"]


@pytest.fixture
def current_test_parameters_string_fx(request):
    """
    This fixture returns the part of the test id between square brackets
    (i.e. the part of id that corresponds to the test parameters)
    """
    node_name = request.node.name
    assert "[" in node_name, f"Wrong format of node name {node_name}"
    assert node_name.endswith("]"), f"Wrong format of node name {node_name}"
    index = node_name.find("[")
    return node_name[index + 1 : -1]


# TODO(lbeynens): replace 'callback' with 'factory'
@pytest.fixture
def cur_test_expected_metrics_callback_fx(
    expected_metrics_all_tests_fx,
    current_test_parameters_string_fx,
    current_test_parameters_fx,
) -> Optional[Callable[[], Dict]]:
    """
    This fixture returns
    * either a callback -- a function without parameters that returns
      expected metrics for the current test,
    * or None if the test validation should be skipped.

    The expected metrics for a test is a dict with the structure that stores the
    requirements on metrics on the current test. In this dict
    * each key is a dot-separated metric "address" in the structure received as the result of the test
    * each value is a structure describing a requirement for this metric
    e.g.
    ```
    {
      'metrics.accuracy.f-measure': {
              'target_value': 0.81,
              'max_diff': 0.005
          }
    }
    ```

    Note that the fixture returns a callback instead of returning the expected metrics structure
    themselves, to avoid attempts to read expected metrics for the stages that do not make validation
    at all -- now the callback is called if and only if validation is made for the stage.
    (E.g. the stage 'export' does not make validation, but the stage 'export_evaluation' does.)

    Also note that if the callback is called, but the expected metrics for the current test
    are not found in the structure with expected metrics for all tests, then the callback
    raises exception ValueError to fail the test.

    And also note that each requirement for each metric is a dict with the following structure:
    * The dict points a target value of the metric.
      The target_value may be pointed
      ** either by key 'target_value' (in this case the value is float),
      ** or by the key 'base', in this case the value is a dot-separated address to another value in the
         storage of previous stages' results, e.g.
             'base': 'training_evaluation.metrics.accuracy.f-measure'

    * The dict points a range of acceptable values for the metric.
      The range for the metric values may be pointed
      ** either by key 'max_diff' (with float value),
         in this case the acceptable range will be
         [target_value - max_diff, target_value + max_diff]
         (inclusively).

      ** or the range may be pointed by keys 'max_diff_if_less_threshold' and/or 'max_diff_if_greater_threshold'
         (with float values), in this case the acceptable range is
         `[target_value - max_diff_if_less_threshold, target_value + max_diff_if_greater_threshold]`
         (also inclusively).
         This allows to point non-symmetric ranges w.r.t. the target_value.
         One of 'max_diff_if_less_threshold' or 'max_diff_if_greater_threshold' may be absent, in this case
         it is set to `+infinity`, so the range will be half-bounded.
         E.g. if `max_diff_if_greater_threshold` is absent, the range will be
         [target_value - max_diff_if_less_threshold, +infinity]
    """
    if REALLIFE_USECASE_CONSTANT() != current_test_parameters_fx["usecase"]:
        return None

    # make a copy to avoid later changes in the structs
    expected_metrics_all_tests = deepcopy(expected_metrics_all_tests_fx)
    current_test_parameters_string = deepcopy(current_test_parameters_string_fx)

    def _get_expected_metrics_callback():
        if expected_metrics_all_tests is None:
            raise ValueError(
                f"The dict with expected metrics cannot be read, although it is required "
                f"for validation in the test '{current_test_parameters_string}'"
            )
        if current_test_parameters_string not in expected_metrics_all_tests:
            raise ValueError(
                f"The parameters id string {current_test_parameters_string} is not inside "
                f"the dict with expected metrics -- cannot make validation, so test is failed"
            )
        expected_metrics = expected_metrics_all_tests[current_test_parameters_string]
        if not isinstance(expected_metrics, dict):
            raise ValueError(
                f"The expected metric for parameters id string {current_test_parameters_string} "
                f"should be a dict, whereas it is: {pformat(expected_metrics)}"
            )
        return expected_metrics

    return _get_expected_metrics_callback


@pytest.fixture
def data_collector_fx(
    request, ote_test_scenario_fx, ote_test_domain_fx
) -> DataCollector:
    """
    The fixture returns the DataCollector instance that may be used to pass
    the values (metrics, intermediate results, etc) to the e2e test system dashboard.
    Please, see the interface of DataCollector class in the function
    e2e_test_system._create_class_DataCollector
    (the function creates a stub class with the proper interface if e2e test system is not installed).

    Note that the fixture contains both setup and teardown parts using yield from fixture.
    Each test uses its own instance of DataCollector class, so each test will create its own row in the
    dashboard of e2e test system.
    """
    setup = deepcopy(request.node.callspec.params)
    setup["environment_name"] = os.environ.get("TT_ENVIRONMENT_NAME", "no-env")
    setup["test_type"] = os.environ.get(
        "TT_TEST_TYPE", "no-test-type"
    )  # TODO: get from e2e test type
    setup["scenario"] = ote_test_scenario_fx
    setup["test"] = request.node.name
    setup["subject"] = ote_test_domain_fx
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
