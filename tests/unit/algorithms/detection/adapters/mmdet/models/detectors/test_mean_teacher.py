from typing import Dict
import pytest
import torch
import numpy as np
from mmdet.core.mask.structures import BitmapMasks

from otx.algorithms.detection.adapters.mmdet.models.detectors.mean_teacher import (
    MeanTeacher,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestMeanTeacher:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        mocker.patch("otx.algorithms.detection.adapters.mmdet.models.detectors.mean_teacher.build_detector")
        mocker.patch.object(MeanTeacher, "_register_state_dict_hook")
        mocker.patch.object(MeanTeacher, "_register_load_state_dict_pre_hook")
        self.mt = MeanTeacher("CustomMaskRCNN")

    @e2e_pytest_unit
    def test_forward_train(self, mocker, monkeypatch):
        def mock_forward_train(*args, **kwargs):
            return {"loss_bbox": 1.0, "loss_cls": 1.0, "loss_mask": 1.0}

        def mock_generate_pseudo_labels(*args, **kwargs):
            return (gt_bboxes, gt_labels, gt_masks, 0.0)

        img = torch.rand(4, 3, 300, 300)
        img_metas = dict()
        gt_bboxes = torch.rand(4, 4)
        gt_labels = torch.randint(20, (4, 1))
        gt_masks = torch.rand(4, 3, 300, 300)
        monkeypatch.setattr(self.mt.model_s, "forward_train", mock_forward_train)
        # mocker.patch(mt.model_s, "forward_train", losses)
        loss = self.mt.forward_train(img, img_metas, gt_bboxes, gt_labels)
        gt_loss = mock_forward_train()
        assert loss == gt_loss
        self.mt.enable_unlabeled_loss(True)
        monkeypatch.setattr(MeanTeacher, "generate_pseudo_labels", mock_generate_pseudo_labels)
        mocker.patch.object(MeanTeacher, "forward_teacher")
        kwargs = {"extra_0": {"img0": img, "img": img, "img_metas": img_metas}}
        loss = self.mt.forward_train(img, img_metas, gt_bboxes, gt_labels, **kwargs)
        gt_loss.update(
            {
                "ps_ratio": torch.tensor([0.0]),
                "loss_bbox_ul": 1.0,
                "loss_cls_ul": 1.0,
                "loss_mask_ul": 1.0,
            }
        )
        assert loss == gt_loss

    @e2e_pytest_unit
    def test_generate_pseudo_labels(self, mocker, monkeypatch):
        gt_bboxes = np.random.rand(1, 1, 5)
        gt_masks = np.random.rand(1, 1, 300, 300) > 0.5
        teacher_output = [([gt_bboxes, gt_masks])]
        img_metas = [{"img_shape": (300, 300, 3)}]
        monkeypatch.setattr(self.mt.model_t, "with_mask", True)
        out = self.mt.generate_pseudo_labels(teacher_output, img_metas, **{"device": "cpu"})
        assert len(out) == 4
        assert isinstance(out[2][-1], BitmapMasks)
        teacher_output = [gt_bboxes]
        monkeypatch.setattr(self.mt.model_t, "with_mask", False)
        out = self.mt.generate_pseudo_labels(teacher_output, img_metas, **{"device": "cpu"})
        assert len(out) == 4
        assert len(out[2]) == 0
