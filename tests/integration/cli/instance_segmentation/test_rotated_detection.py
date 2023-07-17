"""Tests for Class-Incremental Learning for object detection with OTX CLI"""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import copy
import os

import pytest
import torch

from otx.api.entities.model_template import parse_model_template, ModelCategory, ModelStatus
from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_component


otx_dir = os.getcwd()

MULTI_GPU_UNAVAILABLE = torch.cuda.device_count() <= 1

templates = Registry("src/otx/algorithms/detection").filter(task_type="ROTATED_DETECTION").templates
templates_ids = [template.model_template_id for template in templates]


class TestRotatedDetectionModelTemplates:
    @e2e_pytest_component
    def test_model_category(self):
        stat = {
            ModelCategory.SPEED: 0,
            ModelCategory.BALANCE: 0,
            ModelCategory.ACCURACY: 0,
            ModelCategory.OTHER: 0,
        }
        for template in templates:
            stat[template.model_category] += 1
        assert stat[ModelCategory.SPEED] == 1
        assert stat[ModelCategory.BALANCE] <= 1
        assert stat[ModelCategory.ACCURACY] == 1

    @e2e_pytest_component
    def test_model_status(self):
        for template in templates:
            if template.model_status == ModelStatus.DEPRECATED:
                assert template.model_category == ModelCategory.OTHER

    @e2e_pytest_component
    def test_default_for_task(self):
        num_default_model = 0
        for template in templates:
            if template.is_default_for_task:
                num_default_model += 1
                assert template.model_category != ModelCategory.OTHER
                assert template.model_status == ModelStatus.ACTIVE
        assert num_default_model == 1
