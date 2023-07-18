"""Tests for rotated object detection with OTX CLI"""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import copy
import os

import pytest
import torch

from otx.api.entities.model_template import parse_model_template
from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import generate_model_template_testing


otx_dir = os.getcwd()

MULTI_GPU_UNAVAILABLE = torch.cuda.device_count() <= 1

templates = Registry("src/otx/algorithms/detection").filter(task_type="ROTATED_DETECTION").templates
templates_ids = [template.model_template_id for template in templates]


TestRotatedDetectionModelTemplates = generate_model_template_testing(templates)
