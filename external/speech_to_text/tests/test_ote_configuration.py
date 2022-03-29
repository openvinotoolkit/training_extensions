# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os.path as osp
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_api
from ote_sdk.configuration.helper import convert, create
from speech_to_text.ote import OTESpeechToTextTaskParameters


@e2e_pytest_api
def test_configuration_yaml():
    configuration = OTESpeechToTextTaskParameters()
    configuration_yaml_str = convert(configuration, str)
    configuration_yaml_converted = create(configuration_yaml_str)
    configuration_yaml_loaded = create(osp.join('speech_to_text', 'ote', 'configuration.yaml'))
    assert configuration_yaml_converted == configuration_yaml_loaded
