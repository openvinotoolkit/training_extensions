import pytest
import torch
from mmcls.models.builder import build_head

from otx.algorithms.classification.adapters.mmcls.models.heads.semisl_cls_head import (
    SemiLinearClsHead,
    SemiNonLinearClsHead,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestSemiSLClsHead:
    @pytest.fixture(autouse=True)
    def setUp(self):
        """Semi-SL for Classification Head Settings."""
        self.in_channels = 1280
        self.num_classes = 10
        self.head_cfg = dict(
            type="SemiLinearClsHead",
            in_channels=self.in_channels,
            num_classes=self.num_classes,
        )

    @e2e_pytest_unit
    def test_build_semisl_cls_head(self):
        """Verifies that SemiSLClsHead builds."""
        head = build_head(self.head_cfg)
        assert isinstance(head, SemiLinearClsHead)

        head_cfg = dict(
            type="SemiNonLinearClsHead",
            in_channels=self.in_channels,
            num_classes=self.num_classes,
        )
        head = build_head(head_cfg)
        assert isinstance(head, SemiNonLinearClsHead)

    @e2e_pytest_unit
    def test_build_semisl_cls_head_type_error(self):
        """Verifies that SemiSLClsHead parameters check with TypeError."""
        with pytest.raises(TypeError):
            self.head_cfg["num_classes"] = [1]
            build_head(self.head_cfg)
        with pytest.raises(TypeError):
            self.head_cfg["in_channels"] = [1]
            build_head(self.head_cfg)
        with pytest.raises(TypeError):
            self.head_cfg["loss"] = [1]
            build_head(self.head_cfg)
        with pytest.raises(TypeError):
            self.head_cfg["topk"] = [1]
            build_head(self.head_cfg)
        with pytest.raises(TypeError):
            self.head_cfg["unlabeled_coef"] = [1]
            build_head(self.head_cfg)
        with pytest.raises(TypeError):
            self.head_cfg["min_threshold"] = [1]
            build_head(self.head_cfg)

    @e2e_pytest_unit
    def test_build_semisl_cls_head_value_error(self):
        """Verifies that SemiSLClsHead parameters check with ValueError."""
        with pytest.raises(ValueError):
            self.head_cfg["num_classes"] = 0
            build_head(self.head_cfg)
        with pytest.raises(ValueError):
            self.head_cfg["num_classes"] = -1
            build_head(self.head_cfg)
        with pytest.raises(ValueError):
            self.head_cfg["in_channels"] = 0
            build_head(self.head_cfg)
        with pytest.raises(ValueError):
            self.head_cfg["in_channels"] = -1
            build_head(self.head_cfg)

    @e2e_pytest_unit
    def test_forward(self, mocker):
        """Verifies that SemiSLClsHead forward function works."""
        head = build_head(self.head_cfg)
        labeled_batch_size = 16
        unlabeled_batch_size = 64

        dummy_gt = torch.randint(self.num_classes, (labeled_batch_size,))
        labeled = torch.rand(labeled_batch_size, self.in_channels)
        unlabeled_weak = torch.rand(unlabeled_batch_size, self.in_channels)
        unlabeled_strong = torch.rand(unlabeled_batch_size, self.in_channels)

        mocker.patch("torch.cuda.is_available", return_value=False)

        dummy_data = {
            "labeled": labeled,
            "unlabeled_weak": unlabeled_weak,
            "unlabeled_strong": unlabeled_strong,
        }
        head.classwise_acc = head.classwise_acc.cpu()
        loss = head.forward_train(dummy_data, dummy_gt)

        assert isinstance(loss, dict)
        assert len(loss) == 3
        assert isinstance(loss["accuracy"], dict)
        assert len(loss["accuracy"]) == 2

        # No Unlabeled Data
        dummy_feature = torch.rand(labeled_batch_size, self.in_channels)
        loss = head.forward_train(dummy_feature, dummy_gt)

        assert isinstance(loss, dict)
        assert len(loss) == 3
        assert isinstance(loss["accuracy"], dict)
        assert len(loss["accuracy"]) == 2

    @e2e_pytest_unit
    def test_simple_test(self):
        """Verifies that SemiSLClsHead simple_test function works."""
        head = build_head(self.head_cfg)
        dummy_feature = torch.rand(3, self.in_channels)
        features = head.simple_test(dummy_feature)
        assert len(features) == 3
        assert len(features[0]) == self.num_classes
