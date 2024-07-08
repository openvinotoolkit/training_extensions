# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ViT model implementation."""
from __future__ import annotations

import types
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Generic

import numpy as np
import torch
from torch import Tensor, nn

from otx.algo.classification.backbones.vision_transformer import VIT_ARCH_TYPE, TimmVisionTransformer, VisionTransformer
from otx.algo.classification.classifier import ImageClassifier, SemiSLClassifier
from otx.algo.classification.heads import (
    HierarchicalLinearClsHead,
    MultiLabelLinearClsHead,
    OTXSemiSLVisionTransformerClsHead,
    VisionTransformerClsHead,
)
from otx.algo.classification.losses import AsymmetricAngularLossWithIgnore
from otx.algo.classification.utils import get_classification_layers
from otx.algo.classification.utils.embed import resize_pos_embed
from otx.algo.explain.explain_algo import ViTReciproCAM, feature_vector_fn
from otx.algo.utils.mmengine_utils import load_checkpoint_to_model, load_from_http
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.data.entity.base import OTXBatchLossEntity, T_OTXBatchDataEntity, T_OTXBatchPredEntity
from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsBatchPredEntity,
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
    MultilabelClsBatchDataEntity,
    MultilabelClsBatchPredEntity,
)
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics import MetricInput
from otx.core.metrics.accuracy import HLabelClsMetricCallble, MultiClassClsMetricCallable, MultiLabelClsMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.classification import (
    OTXHlabelClsModel,
    OTXMulticlassClsModel,
    OTXMultilabelClsModel,
)
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import HLabelInfo, LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


# pretrained_root = "https://download.openmmlab.com/mmclassification/v0/"
# pretrained_urls = {
#     "deit-tiny": pretrained_root + "deit/deit-tiny_pt-4xb256_in1k_20220218-13b382a0.pth",
# }
pretrained_urls = {
    "deit-tiny": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/deit_tiny_patch16_a2_0-324fe5ea.pth",
}


class ForwardExplainMixInForViT(Generic[T_OTXBatchPredEntity, T_OTXBatchDataEntity]):
    """ViT model which can attach a XAI (Explainable AI) branch."""

    explain_mode: bool
    num_classes: int
    model: ImageClassifier

    @torch.no_grad()
    def head_forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Performs model's neck and head forward."""
        if not hasattr(self.model.backbone, "layers"):
            raise ValueError
        if not hasattr(self.model.backbone, "final_norm"):
            raise ValueError

        # Part of the last transformer_encoder block (except first LayerNorm)
        target_layer = self.model.backbone.layers[-1]
        x = x + target_layer.attn(x)
        x = target_layer.ffn(target_layer.norm2(x), identity=x)

        # Final LayerNorm and neck
        if self.model.backbone.final_norm:
            x = self.model.backbone.norm1(x)
        if self.model.neck is not None:
            x = self.model.neck(x)

        # Head
        cls_token = x[:, 0]
        layer_output = [None, cls_token]
        logit = self.model.head.forward(layer_output)
        if isinstance(logit, list):
            logit = torch.from_numpy(np.array(logit))
        return logit

    @staticmethod
    def _forward_explain_image_classifier(
        self: ImageClassifier,
        images: torch.Tensor,
        mode: str = "tensor",
        **kwargs,  # noqa: ARG004
    ) -> dict[str, torch.Tensor]:
        """Forward func of the ImageClassifier instance, which located in is in OTXModel().model."""
        backbone = self.backbone

        ### Start of backbone forward
        batch_size = images.shape[0]
        x, patch_resolution = backbone.patch_embed(images)

        if backbone.cls_token is not None:
            cls_token = backbone.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + resize_pos_embed(
            backbone.pos_embed,
            backbone.patch_resolution,
            patch_resolution,
            mode=backbone.interpolate_mode,
            num_extra_tokens=backbone.num_extra_tokens,
        )
        x = backbone.drop_after_pos(x)

        x = backbone.pre_norm(x)

        outs = []
        layernorm_feat = None
        for i, layer in enumerate(backbone.layers):
            if i == len(backbone.layers) - 1:
                layernorm_feat = layer.norm1(x)

            x = layer(x)

            if i == len(backbone.layers) - 1 and backbone.final_norm:
                x = backbone.ln1(x)

            if i in backbone.out_indices:
                outs.append(backbone._format_output(x, patch_resolution))  # noqa: SLF001

        x = tuple(outs)
        ### End of backbone forward

        saliency_map = self.explain_fn(layernorm_feat)

        if self.neck is not None:
            x = self.neck(x)

        feature_vector = x[-1]

        if mode in ("tensor", "explain"):
            logits = self.head(x)
        elif mode == "predict":
            logits = self.head.predict(x)
        else:
            msg = f'Invalid mode "{mode}".'
            raise RuntimeError(msg)

        # H-Label Classification Case
        pred_results = self.head._get_predictions(logits)  # noqa: SLF001
        if isinstance(pred_results, dict):
            scores = pred_results["scores"]
            labels = pred_results["labels"]
        else:
            scores = pred_results.unbind(0)
            labels = logits.argmax(-1, keepdim=True).unbind(0)

        return {
            "logits": logits,
            "feature_vector": feature_vector,
            "saliency_map": saliency_map,
            "scores": scores,
            "labels": labels,
        }

    def get_explain_fn(self) -> Callable:
        """Returns explain function."""
        explainer = ViTReciproCAM(
            self.head_forward_fn,
            num_classes=self.num_classes,
        )
        return explainer.func

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for DeitTinyForMultilabelCls."""
        return {"model_type": "transformer"}

    @property
    def has_gap(self) -> bool:
        """Defines if GAP is used right after backbone.

        Note:
            Can be redefined at the model's level.
        """
        return True

    def _register(self) -> None:
        if getattr(self, "_registered", False):
            return
        self.model.feature_vector_fn = feature_vector_fn
        self.model.explain_fn = self.get_explain_fn()
        self._registered = True

    def forward_explain(
        self,
        inputs: T_OTXBatchDataEntity,
    ) -> T_OTXBatchPredEntity:
        """Model forward function."""
        self._register()
        orig_model_forward = self.model.forward

        try:
            self.model.forward = types.MethodType(self._forward_explain_image_classifier, self.model)  # type: ignore[method-assign, assignment]

            forward_func: Callable[[T_OTXBatchDataEntity], T_OTXBatchPredEntity] | None = getattr(self, "forward", None)

            if forward_func is None:
                msg = (
                    "This instance has no forward function. "
                    "Did you attach this mixin into a class derived from OTXModel?"
                )
                raise RuntimeError(msg)

            return forward_func(inputs)
        finally:
            self.model.forward = orig_model_forward  # type: ignore[method-assign, assignment]

    def forward_for_tracing(self, image: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            self._register()
            forward_explain = types.MethodType(self._forward_explain_image_classifier, self.model)
            return forward_explain(images=image, mode="tensor")

        return self.model(images=image, mode="tensor")


class VisionTransformerForMulticlassCls(ForwardExplainMixInForViT, OTXMulticlassClsModel):
    """DeitTiny Model for multi-class classification task."""

    model: ImageClassifier

    def __init__(
        self,
        label_info: LabelInfoTypes,
        arch: VIT_ARCH_TYPE = "deit-tiny",
        pretrained: bool = True,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        self.arch = arch
        self.pretrained = pretrained
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, "multiclass", add_prefix)

    def _create_model(self) -> nn.Module:
        # Get classification_layers for class-incr learning
        sample_model_dict = self._build_model(num_classes=5).state_dict()
        incremental_model_dict = self._build_model(num_classes=6).state_dict()
        self.classification_layers = get_classification_layers(
            sample_model_dict,
            incremental_model_dict,
            prefix="model.",
        )

        model = self._build_model(num_classes=self.num_classes)
        model.init_weights()
        if self.pretrained and self.arch in pretrained_urls:
            print(f"init weight - {pretrained_urls[self.arch]}")
            checkpoint = load_from_http(pretrained_urls[self.arch], map_location="cpu")
            load_checkpoint_to_model(model.backbone, checkpoint)
        return model

    def _build_model(self, num_classes: int) -> nn.Module:
        init_cfg = [
            {"std": 0.2, "layer": "Linear", "type": "TruncNormal"},
            {"bias": 0.0, "val": 1.0, "layer": "LayerNorm", "type": "Constant"},
        ]
        return ImageClassifier(
            # backbone=VisionTransformer(arch=self.arch, img_size=224, patch_size=16),
            backbone=TimmVisionTransformer(img_size=224, patch_size=16, embed_dim=192),
            neck=None,
            head=VisionTransformerClsHead(
                num_classes=num_classes,
                in_channels=192,
                topk=(1, 5) if num_classes >= 5 else (1,),
                loss=nn.CrossEntropyLoss(reduction="none"),
            ),
            init_cfg=init_cfg,
        )

    def _customize_inputs(self, inputs: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        if self.training:
            mode = "loss"
        elif self.explain_mode:
            mode = "explain"
        else:
            mode = "predict"

        return {
            "images": inputs.stacked_images,
            "labels": torch.cat(inputs.labels, dim=0),
            "imgs_info": inputs.imgs_info,
            "mode": mode,
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            return OTXBatchLossEntity(loss=outputs)

        if self.explain_mode:
            return MulticlassClsBatchPredEntity(
                batch_size=inputs.batch_size,
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=outputs["scores"],
                labels=outputs["labels"],
                saliency_map=outputs["saliency_map"],
                feature_vector=outputs["feature_vector"],
            )

        # To list, batch-wise
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs["logits"]
        scores = torch.unbind(logits, 0)
        preds = logits.argmax(-1, keepdim=True).unbind(0)

        return MulticlassClsBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=preds,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, 224, 224),
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=True,  # NOTE: This should be done via onnx
            onnx_export_configuration=None,
            output_names=["logits", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )


class VisionTransformerForMulticlassClsSemiSL(VisionTransformerForMulticlassCls):
    """VisionTransformer model for multiclass classification with semi-supervised learning.

    This class extends the `VisionTransformerForMulticlassCls` class and adds support for semi-supervised learning.
    It overrides the `_build_model` and `_customize_inputs` methods to incorporate the semi-supervised learning.

    Args:
        VisionTransformerForMulticlassCls (class): The base class for VisionTransformer multiclass classification.
    """

    def _build_model(self, num_classes: int) -> nn.Module:
        init_cfg = [
            {"std": 0.2, "layer": "Linear", "type": "TruncNormal"},
            {"bias": 0.0, "val": 1.0, "layer": "LayerNorm", "type": "Constant"},
        ]
        return SemiSLClassifier(
            backbone=VisionTransformer(arch=self.arch, img_size=224, patch_size=16),
            neck=None,
            head=OTXSemiSLVisionTransformerClsHead(
                num_classes=num_classes,
                in_channels=192,
                loss=nn.CrossEntropyLoss(reduction="none"),
            ),
            init_cfg=init_cfg,
        )

    def _customize_inputs(self, inputs: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        """Customizes the input data for the model based on the current mode.

        Args:
            inputs (MulticlassClsBatchDataEntity): The input batch of data.

        Returns:
            dict[str, Any]: The customized input data.
        """
        if self.training:
            mode = "loss"
        elif self.explain_mode:
            mode = "explain"
        else:
            mode = "predict"

        if isinstance(inputs, dict):
            # When used with an unlabeled dataset, it comes in as a dict.
            images = {key: inputs[key].images for key in inputs}
            labels = {key: torch.cat(inputs[key].labels, dim=0) for key in inputs}
            imgs_info = {key: inputs[key].imgs_info for key in inputs}
            return {
                "images": images,
                "labels": labels,
                "imgs_info": imgs_info,
                "mode": mode,
            }
        return {
            "images": inputs.images,
            "labels": torch.cat(inputs.labels, dim=0),
            "imgs_info": inputs.imgs_info,
            "mode": mode,
        }

    def training_step(self, batch: MulticlassClsBatchDataEntity, batch_idx: int) -> Tensor:
        """Performs a single training step on a batch of data.

        Args:
            batch (MulticlassClsBatchDataEntity): The input batch of data.
            batch_idx (int): The index of the current batch.

        Returns:
            Tensor: The computed loss for the training step.
        """
        loss = super().training_step(batch, batch_idx)
        # Collect metrics related to Semi-SL Training.
        self.log(
            "train/unlabeled_coef",
            self.model.head.unlabeled_coef,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "train/num_pseudo_label",
            self.model.head.num_pseudo_label,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss


class VisionTransformerForMultilabelCls(ForwardExplainMixInForViT, OTXMultilabelClsModel):
    """DeitTiny Model for multi-class classification task."""

    model: ImageClassifier

    def __init__(
        self,
        label_info: LabelInfoTypes,
        arch: VIT_ARCH_TYPE = "deit-tiny",
        pretrained: bool = True,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiLabelClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        self.arch = arch
        self.pretrained = pretrained

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, "multiclass", add_prefix)

    def _create_model(self) -> nn.Module:
        # Get classification_layers for class-incr learning
        sample_model_dict = self._build_model(num_classes=5).state_dict()
        incremental_model_dict = self._build_model(num_classes=6).state_dict()
        self.classification_layers = get_classification_layers(
            sample_model_dict,
            incremental_model_dict,
            prefix="model.",
        )

        model = self._build_model(num_classes=self.num_classes)
        model.init_weights()
        if self.pretrained and self.arch in pretrained_urls:
            print(f"init weight - {pretrained_urls[self.arch]}")
            checkpoint = load_from_http(pretrained_urls[self.arch], map_location="cpu")
            load_checkpoint_to_model(model, checkpoint)
        return model

    def _build_model(self, num_classes: int) -> nn.Module:
        init_cfg = [
            {"std": 0.2, "layer": "Linear", "type": "TruncNormal"},
            {"bias": 0.0, "val": 1.0, "layer": "LayerNorm", "type": "Constant"},
        ]
        return ImageClassifier(
            backbone=VisionTransformer(arch=self.arch, img_size=224, patch_size=16),
            neck=None,
            head=MultiLabelLinearClsHead(
                num_classes=num_classes,
                in_channels=192,
                loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
            ),
            init_cfg=init_cfg,
        )

    def _customize_inputs(self, inputs: MultilabelClsBatchDataEntity) -> dict[str, Any]:
        if self.training:
            mode = "loss"
        elif self.explain_mode:
            mode = "explain"
        else:
            mode = "predict"

        return {
            "images": inputs.stacked_images,
            "labels": torch.stack(inputs.labels),
            "imgs_info": inputs.imgs_info,
            "mode": mode,
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: MultilabelClsBatchDataEntity,
    ) -> MultilabelClsBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            return OTXBatchLossEntity(loss=outputs)

        if self.explain_mode:
            return MultilabelClsBatchPredEntity(
                batch_size=inputs.batch_size,
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=outputs["scores"],
                labels=outputs["labels"],
                saliency_map=outputs["saliency_map"],
                feature_vector=outputs["feature_vector"],
            )

        # To list, batch-wise
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs["logits"]
        scores = torch.unbind(logits, 0)
        preds = logits.argmax(-1, keepdim=True).unbind(0)

        return MultilabelClsBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=preds,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, 224, 224),
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=True,  # NOTE: This should be done via onnx
            onnx_export_configuration=None,
            output_names=["logits", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )


class VisionTransformerForHLabelCls(ForwardExplainMixInForViT, OTXHlabelClsModel):
    """DeitTiny Model for hierarchical label classification task."""

    model: ImageClassifier
    label_info: HLabelInfo

    def __init__(
        self,
        label_info: HLabelInfo,
        arch: VIT_ARCH_TYPE = "deit-tiny",
        pretrained: bool = True,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = HLabelClsMetricCallble,
        torch_compile: bool = False,
    ) -> None:
        self.arch = arch
        self.pretrained = pretrained

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, "multiclass", add_prefix)

    def _create_model(self) -> nn.Module:
        # Get classification_layers for class-incr learning
        sample_config = deepcopy(self.label_info.as_head_config_dict())
        sample_config["num_classes"] = 5
        sample_model_dict = self._build_model(head_config=sample_config).state_dict()
        sample_config["num_classes"] = 6
        incremental_model_dict = self._build_model(head_config=sample_config).state_dict()
        self.classification_layers = get_classification_layers(
            sample_model_dict,
            incremental_model_dict,
            prefix="model.",
        )

        model = self._build_model(head_config=self.label_info.as_head_config_dict())
        model.init_weights()
        if self.pretrained and self.arch in pretrained_urls:
            print(f"init weight - {pretrained_urls[self.arch]}")
            checkpoint = load_from_http(pretrained_urls[self.arch], map_location="cpu")
            load_checkpoint_to_model(model, checkpoint)
        return model

    def _build_model(self, head_config: dict) -> nn.Module:
        if not isinstance(self.label_info, HLabelInfo):
            raise TypeError(self.label_info)
        init_cfg = [
            {"std": 0.2, "layer": "Linear", "type": "TruncNormal"},
            {"bias": 0.0, "val": 1.0, "layer": "LayerNorm", "type": "Constant"},
        ]
        return ImageClassifier(
            backbone=VisionTransformer(arch=self.arch, img_size=224, patch_size=16),
            neck=None,
            head=HierarchicalLinearClsHead(
                in_channels=192,
                multiclass_loss=nn.CrossEntropyLoss(),
                multilabel_loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
                **head_config,
            ),
            init_cfg=init_cfg,
        )

    def _customize_inputs(self, inputs: HlabelClsBatchDataEntity) -> dict[str, Any]:
        if self.training:
            mode = "loss"
        elif self.explain_mode:
            mode = "explain"
        else:
            mode = "predict"

        return {
            "images": inputs.stacked_images,
            "labels": torch.stack(inputs.labels),
            "imgs_info": inputs.imgs_info,
            "mode": mode,
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: HlabelClsBatchDataEntity,
    ) -> HlabelClsBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            return OTXBatchLossEntity(loss=outputs)

        if isinstance(outputs, dict):
            scores = outputs["scores"]
            labels = outputs["labels"]
        else:
            scores = outputs
            labels = outputs.argmax(-1, keepdim=True)

        if self.explain_mode:
            return HlabelClsBatchPredEntity(
                batch_size=inputs.batch_size,
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                labels=labels,
                saliency_map=outputs["saliency_map"],
                feature_vector=outputs["feature_vector"],
            )

        return HlabelClsBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: HlabelClsBatchPredEntity,
        inputs: HlabelClsBatchDataEntity,
    ) -> MetricInput:
        hlabel_info: HLabelInfo = self.label_info  # type: ignore[assignment]

        _labels = torch.stack(preds.labels) if isinstance(preds.labels, list) else preds.labels
        _scores = torch.stack(preds.scores) if isinstance(preds.scores, list) else preds.scores
        if hlabel_info.num_multilabel_classes > 0:
            preds_multiclass = _labels[:, : hlabel_info.num_multiclass_heads]
            preds_multilabel = _scores[:, hlabel_info.num_multiclass_heads :]
            pred_result = torch.cat([preds_multiclass, preds_multilabel], dim=1)
        else:
            pred_result = _labels
        return {
            "preds": pred_result,
            "target": torch.stack(inputs.labels),
        }

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, 224, 224),
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=True,  # NOTE: This should be done via onnx
            onnx_export_configuration=None,
            output_names=["logits", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )
