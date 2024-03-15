import pytest
import torch

from otx.algorithms.segmentation.adapters.mmseg.models.heads.proto_head import ProtoNet
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestProtoNet:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.proto_net = ProtoNet(
            gamma=0.99, num_prototype=4, in_proto_channels=512, in_channels=512, channels=512, num_classes=4
        )

    def test_prototype_learning(self):
        dummy_input = torch.rand(32768, 512)
        dummy_out_seg = torch.rand(8, 4, 64, 64)
        dummy_masks = torch.rand(32768, 4, 4)
        dummy_gt_seg = torch.randint(low=0, high=5, size=(32768,))
        proto_logits, proto_target = self.proto_net.prototype_learning(
            dummy_input, dummy_out_seg, dummy_gt_seg, dummy_masks
        )
        assert proto_logits is not None
        assert proto_target is not None

    def test_forward(self):
        dummy_input = torch.rand(8, 512, 64, 64)
        dummy_gt_seg = torch.randint(low=0, high=5, size=(8, 1, 512, 512))
        proto_out = self.proto_net(inputs=dummy_input, gt_semantic_seg=dummy_gt_seg)
        assert isinstance(proto_out, dict)
        assert proto_out["out_seg"] is not None
