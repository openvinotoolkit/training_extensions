import pytest
import torch
from otx.algo.classification.torchvision_model import OTXTVModel, TVModelWithLossComputation
from otx.core.data.entity.base import ImageInfo, OTXBatchLossEntity
from otx.core.data.entity.classification import (
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
)


@pytest.fixture()
def fxt_tv_model():
    return OTXTVModel(backbone="resnet50", num_classes=10)


@pytest.fixture()
def fxt_inputs():
    return MulticlassClsBatchDataEntity(
        batch_size=16,
        images=torch.randn(16, 3, 224, 224),
        imgs_info=[ImageInfo(img_idx=i, img_shape=(224, 224), ori_shape=(224, 224)) for i in range(16)],
        labels=[torch.randint(0, 10, (16,))],
    )


class TestOTXTVModel:
    def test_create_model(self, fxt_tv_model):
        assert isinstance(fxt_tv_model._create_model(), TVModelWithLossComputation)

    def test_customize_inputs(self, fxt_tv_model, fxt_inputs):
        outputs = fxt_tv_model._customize_inputs(fxt_inputs)
        assert "images" in outputs
        assert "labels" in outputs
        assert "mode" in outputs

    def test_customize_outputs(self, fxt_tv_model, fxt_inputs):
        outputs = torch.randn(16, 10)
        fxt_tv_model.training = True
        preds = fxt_tv_model._customize_outputs(outputs, fxt_inputs)
        assert isinstance(preds, OTXBatchLossEntity)

        fxt_tv_model.training = False
        preds = fxt_tv_model._customize_outputs(outputs, fxt_inputs)
        assert isinstance(preds, MulticlassClsBatchPredEntity)

    def test_export_parameters(self, fxt_tv_model):
        params = fxt_tv_model._export_parameters
        assert isinstance(params, dict)
        assert "input_size" in params
        assert "resize_mode" in params
        assert "pad_value" in params
        assert "swap_rgb" in params
        assert "via_onnx" in params
        assert "onnx_export_configuration" in params
        assert "mean" in params
        assert "std" in params

    def test_forward_explain_image_classifier(self, fxt_tv_model):
        images = torch.randn(16, 3, 224, 224)
        fxt_tv_model._explain_mode = True
        fxt_tv_model._reset_model_forward()
        outputs = fxt_tv_model._forward_explain_image_classifier(fxt_tv_model.model, images)
        assert "logits" in outputs
        assert "feature_vector" in outputs
        assert "saliency_map" in outputs

    def test_head_forward_fn(self, fxt_tv_model):
        x = torch.randn(16, 2048)
        output = fxt_tv_model.head_forward_fn(x)
        assert output.shape == (16, 10)

    def test_freeze_backbone(self):
        freezed_model = OTXTVModel(backbone="resnet50", num_classes=10, freeze_backbone=True)
        for param in freezed_model.model.backbone.parameters():
            assert not param.requires_grad
