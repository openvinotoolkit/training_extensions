import pytest
import torch
from otx.algo.segmentation.segmentors.mean_teacher import MeanTeacher
from torch import nn


class TestMeanTeacher:
    @pytest.fixture()
    def model(self):
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        return MeanTeacher(model)

    @pytest.fixture()
    def inputs(self):
        return torch.randn(4, 10)

    @pytest.fixture()
    def unlabeled_weak_images(self):
        return torch.randn(4, 10)

    @pytest.fixture()
    def unlabeled_strong_images(self):
        return torch.randn(4, 10)

    @pytest.fixture()
    def img_metas(self):
        return [{"img_shape": (300, 300, 3)}] * 4

    @pytest.fixture()
    def unlabeled_img_metas(self):
        return [{"img_shape": (300, 300, 3)}] * 4

    @pytest.fixture()
    def masks(self):
        return torch.randn(4, 300, 300)

    def test_forward_labeled_images(self, model, inputs, img_metas, masks):
        output = model.forward(inputs, img_metas, masks, mode="tensor")
        assert output.shape == (4, 2)

    def test_forward_unlabeled_images(self, model, unlabeled_weak_images, unlabeled_strong_images, unlabeled_img_metas):
        output = model.forward(unlabeled_weak_images, unlabeled_strong_images, unlabeled_img_metas, mode="loss")
        assert isinstance(output, dict)

    def test_generate_pseudo_labels(self, model, unlabeled_weak_images, unlabeled_img_metas):
        pl_from_teacher, reweight_unsup = model.generate_pseudo_labels(
            unlabeled_weak_images,
            unlabeled_img_metas,
            percent_unreliable=20,
        )
        assert pl_from_teacher.shape == (4, 1, 300, 300)
        assert isinstance(reweight_unsup, float)
