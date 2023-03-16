import pytest
import torch

from otx.algorithms.detection.adapters.mmdet.hooks.det_saliency_map_hook import (
    DetSaliencyMapHook,
)
from otx.algorithms.detection.adapters.mmdet.models.heads.custom_atss_head import (
    CustomATSSHead,
)
from otx.algorithms.detection.adapters.mmdet.models.heads.custom_ssd_head import (
    CustomSSDHead,
)
from otx.algorithms.detection.adapters.mmdet.models.heads.custom_vfnet_head import (
    CustomVFNetHead,
)
from otx.algorithms.detection.adapters.mmdet.models.heads.custom_yolox_head import (
    CustomYOLOXHead,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestDetSaliencyMapHook:
    """Test class for DetSaliencyMapHook."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        class _MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.with_neck = True
                self.neck = torch.nn.Module()
                self.neck.forward = self.forward
                self.bbox_head = torch.nn.Module()
                self.bbox_head.cls_out_channels = 3

            def forward(self, x):
                return x

        self.module = _MockModule()
        self.hook = DetSaliencyMapHook(self.module)

    @e2e_pytest_unit
    def test_func(self, mocker) -> None:
        """Test func function."""

        mocker.patch.object(
            DetSaliencyMapHook, "_get_cls_scores_from_feature_map", return_value=[torch.randn(1, 3, 14, 14)]
        )
        assert self.hook.func(torch.randn(1, 3, 14, 14)) is not None

    @e2e_pytest_unit
    def test_get_cls_scores_from_feature_map(self) -> None:
        """Test _get_cls_scores_from_feature_map function."""

        self.module.bbox_head = CustomATSSHead(num_classes=3, in_channels=64)
        self.hook = DetSaliencyMapHook(self.module)
        assert self.hook._get_cls_scores_from_feature_map(torch.Tensor(1, 3, 64, 32, 32)) is not None
        self.module.bbox_head = CustomYOLOXHead(num_classes=3, in_channels=64)
        self.hook = DetSaliencyMapHook(self.module)
        assert self.hook._get_cls_scores_from_feature_map(torch.Tensor(1, 3, 64, 32, 32)) is not None
        self.module.bbox_head = CustomVFNetHead(num_classes=3, in_channels=64)
        self.module.bbox_head.anchor_generator.num_base_anchors = 1
        self.hook = DetSaliencyMapHook(self.module)
        assert self.hook._get_cls_scores_from_feature_map(torch.Tensor(1, 3, 64, 32, 32)) is not None
        self.module.bbox_head = CustomSSDHead(
            anchor_generator=dict(
                type="SSDAnchorGenerator",
                basesize_ratio_range=(0.15, 0.9),
                strides=(16, 32, 48),
                ratios=[[0.5], [0.1], [0.3]],
            ),
            act_cfg={},
        )
        self.hook = DetSaliencyMapHook(self.module)
        assert self.hook._get_cls_scores_from_feature_map(torch.Tensor(1, 3, 512, 32, 32)) is not None
        self.module.bbox_head = torch.nn.Module()
        self.module.bbox_head.cls_out_channels = 3
        self.hook = DetSaliencyMapHook(self.module)
        with pytest.raises(NotImplementedError):
            self.hook._get_cls_scores_from_feature_map(torch.Tensor(1, 3, 512, 32, 32))
