"""Test MMSegmentationTask."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import torch
import pytest
from mmcv import ConfigDict

from otx.api.entities.label import LabelEntity, Domain
from otx.api.entities.id import ID
from otx.algorithms.segmentation.adapters.mmseg.task import MMSegmentationTask
from otx.algorithms.segmentation.adapters.mmseg.models.heads import otx_head_factory
from otx.api.configuration.helper import create
from otx.api.entities.model_template import (
    parse_model_template,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.segmentation.test_helpers import (
    DEFAULT_SEG_TEMPLATE_DIR,
    init_environment,
)


class TestMMSegmentationTask:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        model_template = parse_model_template(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        task_env = init_environment(hyper_parameters, model_template)
        self.mmseg_task = MMSegmentationTask(task_env)
        self.mmseg_task._init_task()

    @e2e_pytest_unit
    def test_configure(self):
        cfg = self.mmseg_task.configure()
        assert "work_dir" in cfg
        assert "resume" in cfg

    @e2e_pytest_unit
    def test_build_model(self, mocker):
        mocker.patch("otx.algorithms.segmentation.adapters.mmseg.utils.builder.build_segmentor")
        self.mmseg_task._recipe_cfg.model.decode_head.base_type = "FCNHead"
        self.mmseg_task._recipe_cfg.model.decode_head.type = otx_head_factory
        self.mmseg_task._recipe_cfg.model.task_adapt = ConfigDict(
            op="REPLACE",
            type="mpa",
            final=["background", "target"],
            src_classes=[],
            dst_classes=["background", "target"],
        )
        model = self.mmseg_task.build_model(self.mmseg_task._recipe_cfg, fp16=True)
        assert isinstance(model, torch.nn.Module)
        assert model.fp16_enabled

    @e2e_pytest_unit
    def test_update_override_configurations(self):
        cfg = ConfigDict(fake_key="fake_value")
        self.mmseg_task.update_override_configurations(cfg)
        assert "fake_key" in self.mmseg_task.override_configs
        assert self.mmseg_task.override_configs == dict(fake_key="fake_value")

    @e2e_pytest_unit
    def test_save_model(self, otx_model, mocker):
        mocker_load = mocker.patch("torch.load", return_value="foo")
        mocker_save = mocker.patch("torch.save")
        self.mmseg_task.save_model(otx_model)

        mocker_load.assert_called_once()
        mocker_save.assert_called_once()

    @e2e_pytest_unit
    def test_label_order(self, mocker):
        mocker.patch("otx.algorithms.segmentation.task.os")
        mocker.patch("otx.algorithms.segmentation.task.TRAIN_TYPE_DIR_PATH")
        mock_environemnt = mocker.MagicMock()

        fake_label = []
        for i in range(20):
            fake_label.append(LabelEntity(name=f"class_{i}", domain=Domain.SEGMENTATION, id=ID(str(i))))
        mock_environemnt.get_labels.return_value = fake_label
        del mock_environemnt.get_hyper_parameters.return_value.learning_parameters.input_size  # To avoid mocking error
        task = MMSegmentationTask(mock_environemnt)

        for i, label_entity in task._label_dictionary.items():
            assert label_entity.name == f"class_{i-1}"
