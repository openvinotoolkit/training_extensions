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
        self.mt_is = MeanTeacher("CustomMaskRCNN")
        self.mt_det = MeanTeacher("CustomATSS", unlabeled_loss_weights={"cls": 1, "bbox": 1, "obj": 1})
        self.img = torch.rand(4, 3, 300, 300)
        self.img_metas = [dict(ori_shape=(300, 300), scale_factor=1.0)] * 4
        self.gt_bboxes = torch.rand(4, 4)
        self.gt_labels = torch.randint(20, (4, 1))
        self.gt_masks = torch.rand(4, 3, 300, 300)

    @e2e_pytest_unit
    def test_forward_train_segmentation(self, mocker, monkeypatch):
        def mock_forward_train(*args, **kwargs):
            return {"loss_bbox": 1.0, "loss_cls": 1.0, "loss_mask": 1.0}

        def mock_generate_pseudo_labels(*args, **kwargs):
            return (self.gt_bboxes, self.gt_labels, self.gt_masks, 0.0)

        monkeypatch.setattr(self.mt_is.model_s, "forward_train", mock_forward_train)
        loss = self.mt_is.forward_train(
            self.img, self.img_metas, self.img, self.gt_bboxes, self.gt_labels, self.gt_masks
        )
        gt_loss = mock_forward_train()
        assert loss == gt_loss
        self.mt_is.enable_unlabeled_loss(True)
        monkeypatch.setattr(MeanTeacher, "generate_pseudo_labels", mock_generate_pseudo_labels)
        mocker.patch.object(MeanTeacher, "forward_teacher")
        kwargs = {"extra_0": {"img0": self.img, "img": self.img, "img_metas": self.img_metas}}
        loss_mask = self.mt_is.forward_train(
            self.img, self.img_metas, None, self.gt_bboxes, self.gt_labels, self.gt_masks, **kwargs
        )
        gt_loss.update(
            {
                "ps_ratio": torch.tensor([0.0]),
                "loss_bbox_ul": 1.0,
                "loss_cls_ul": 1.0,
                "loss_mask_ul": 1.0,
            }
        )
        assert loss_mask == gt_loss

    @e2e_pytest_unit
    def test_forward_train_detection(self, mocker, monkeypatch):
        def mock_forward_train(*args, **kwargs):
            return {"loss_bbox": 1.0, "loss_cls": 1.0, "loss_obj": 1.0}

        def mock_generate_pseudo_labels(*args, **kwargs):
            return (self.gt_bboxes, self.gt_labels, [], 0.0)

        monkeypatch.setattr(self.mt_det.model_s, "forward_train", mock_forward_train)
        monkeypatch.setattr(self.mt_is.model_s, "with_mask", False)
        loss = self.mt_det.forward_train(self.img, self.img_metas, self.img, self.gt_bboxes, self.gt_labels)
        gt_loss = mock_forward_train()
        assert loss == gt_loss
        self.mt_det.enable_unlabeled_loss(True)
        monkeypatch.setattr(MeanTeacher, "generate_pseudo_labels", mock_generate_pseudo_labels)
        mocker.patch.object(MeanTeacher, "forward_teacher")
        kwargs = {"extra_0": {"img0": self.img, "img": self.img, "img_metas": self.img_metas}}
        loss_det = self.mt_det.forward_train(
            self.img, self.img_metas, self.img, self.gt_bboxes, self.gt_labels, **kwargs
        )
        gt_loss.update(
            {
                "ps_ratio": torch.tensor([0.0]),
                "loss_bbox_ul": 1.0,
                "loss_cls_ul": 1.0,
                "loss_obj_ul": 1.0,
            }
        )
        assert loss_det == gt_loss

    @e2e_pytest_unit
    def test_generate_pseudo_labels(self, mocker, monkeypatch):
        gt_bboxes = np.random.rand(1, 1, 5)
        gt_masks = np.random.rand(1, 1, 300, 300) > 0.5
        teacher_output = [([gt_bboxes, gt_masks])]
        img_metas = [{"img_shape": (300, 300, 3)}]
        monkeypatch.setattr(self.mt_is.model_t, "with_mask", True)
        out = self.mt_is.generate_pseudo_labels(teacher_output, img_metas, **{"device": "cpu"})
        assert len(out) == 4
        assert isinstance(out[2][-1], BitmapMasks)
        teacher_output = [gt_bboxes]
        monkeypatch.setattr(self.mt_is.model_t, "with_mask", False)
        out = self.mt_is.generate_pseudo_labels(teacher_output, img_metas, **{"device": "cpu"})
        assert len(out) == 4
        assert len(out[2]) == 0

    @e2e_pytest_unit
    def test_forward_teacher(self, mocker, monkeypatch):
        def mock_simple_test_bboxes(*args, **kwargs):
            return [self.gt_bboxes], [self.gt_labels]

        monkeypatch.setattr(self.mt_is.model_t.roi_head, "simple_test_bboxes", mock_simple_test_bboxes)
        mocker.patch("otx.algorithms.detection.adapters.mmdet.models.detectors.mean_teacher.bbox2result")
        mocker.patch("otx.algorithms.detection.adapters.mmdet.models.detectors.mean_teacher.bbox2roi")
        teacher_output = self.mt_is.forward_teacher(self.img, self.img_metas)
        assert teacher_output is not None
        assert isinstance(teacher_output, list)
