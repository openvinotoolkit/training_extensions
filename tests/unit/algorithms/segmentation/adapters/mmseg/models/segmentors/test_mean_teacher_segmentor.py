import pytest
import torch

from otx.algorithms.segmentation.adapters.mmseg import MeanTeacherSegmentor
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestMeanTeacherSegmentor:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        mocker.patch(
            "otx.algorithms.segmentation.adapters.mmseg.models.segmentors.mean_teacher_segmentor.build_segmentor"
        )
        self.mean_teacher = MeanTeacherSegmentor(None, 100, test_cfg=dict(), decode_head={"align_corners": False})
        self.mean_teacher.proto_net = mocker.MagicMock()
        self.mean_teacher.use_prototype_head = True
        self.input = torch.rand(4, 3, 512, 512)
        self.gt_seg = torch.randint(low=0, high=5, size=(4, 1, 512, 512))

    @e2e_pytest_unit
    def test_decode_proto_network(self, mocker):
        mocker_update_loss = mocker.patch.object(self.mean_teacher, "_update_summary_loss")
        self.mean_teacher.decode_proto_network(self.input, self.gt_seg)
        mocker_update_loss.assert_called_once()
        # dummy input
        self.mean_teacher.decode_proto_network(self.input, self.gt_seg, self.input, self.gt_seg)

    @e2e_pytest_unit
    def test_generate_pseudo_labels(self, mocker):
        mocker.patch(
            "otx.algorithms.segmentation.adapters.mmseg.models.segmentors.mean_teacher_segmentor.resize",
            return_value=self.input,
        )
        pl_from_teacher, reweight_unsup = self.mean_teacher.generate_pseudo_labels(
            ul_w_img=self.input, ul_img_metas=dict()
        )
        assert isinstance(pl_from_teacher, torch.Tensor)
        assert pl_from_teacher.shape == (4, 1, 512, 512)
        assert round(reweight_unsup.item(), 2) == 1.25

    @e2e_pytest_unit
    def test_forward_train(self, mocker):
        loss = self.mean_teacher(self.input, img_metas=dict(), gt_semantic_seg=self.gt_seg)
        assert loss is not None
        self.mean_teacher.semisl_start_iter = -1
        mocker.patch.object(self.mean_teacher, "decode_proto_network")
        mocker.patch.object(self.mean_teacher, "generate_pseudo_labels", return_value=(self.gt_seg, 1.0))
        ul_kwargs = dict(extra_0=dict(img=self.input, ul_w_img=self.input, img_metas=dict()))
        loss = self.mean_teacher(self.input, img_metas=dict(), gt_semantic_seg=self.gt_seg, **ul_kwargs)
        assert loss is not None
        assert loss["sum_loss"] == 0.0
