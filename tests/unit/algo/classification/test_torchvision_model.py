import pytest
import torch
from otx.algo.classification.torchvision_model import OTXTVModel, TVModelWithLossComputation
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import MulticlassClsBatchPredEntity


@pytest.fixture()
def fxt_tv_model():
    return OTXTVModel(backbone="mobilenet_v3_small", num_classes=10)


class TestOTXTVModel:
    def test_create_model(self, fxt_tv_model):
        assert isinstance(fxt_tv_model.model, TVModelWithLossComputation)

    def test_customize_inputs(self, fxt_tv_model, fxt_multiclass_cls_batch_data_entity):
        outputs = fxt_tv_model._customize_inputs(fxt_multiclass_cls_batch_data_entity)
        assert "images" in outputs
        assert "labels" in outputs
        assert "mode" in outputs

    def test_customize_outputs(self, fxt_tv_model, fxt_multiclass_cls_batch_data_entity):
        outputs = torch.randn(2, 10)
        fxt_tv_model.training = True
        preds = fxt_tv_model._customize_outputs(outputs, fxt_multiclass_cls_batch_data_entity)
        assert isinstance(preds, OTXBatchLossEntity)

        fxt_tv_model.training = False
        preds = fxt_tv_model._customize_outputs(outputs, fxt_multiclass_cls_batch_data_entity)
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

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_predict_step(self, fxt_tv_model: OTXTVModel, fxt_multiclass_cls_batch_data_entity, explain_mode):
        fxt_tv_model.eval()
        fxt_tv_model.explain_mode = explain_mode
        outputs = fxt_tv_model.predict_step(batch=fxt_multiclass_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, MulticlassClsBatchPredEntity)
        assert outputs.has_xai_outputs == explain_mode

    def test_freeze_backbone(self):
        freezed_model = OTXTVModel(backbone="resnet50", num_classes=10, freeze_backbone=True)
        for param in freezed_model.model.backbone.parameters():
            assert not param.requires_grad
