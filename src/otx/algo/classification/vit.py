# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ViT model implementation."""

from __future__ import annotations

import types
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn
from torch.hub import download_url_to_file

from otx.algo.classification.backbones.vision_transformer import VIT_ARCH_TYPE, VisionTransformer
from otx.algo.classification.classifier import HLabelClassifier, ImageClassifier, SemiSLClassifier
from otx.algo.classification.heads import (
    HierarchicalCBAMClsHead,
    MultiLabelLinearClsHead,
    SemiSLVisionTransformerClsHead,
    VisionTransformerClsHead,
)
from otx.algo.classification.losses import AsymmetricAngularLossWithIgnore
from otx.algo.classification.utils import get_classification_layers
from otx.algo.explain.explain_algo import ViTReciproCAM, feature_vector_fn
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.data.entity.base import T_OTXBatchDataEntity, T_OTXBatchPredEntity
from otx.core.metrics.accuracy import HLabelClsMetricCallable, MultiClassClsMetricCallable, MultiLabelClsMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.classification import (
    OTXHlabelClsModel,
    OTXMulticlassClsModel,
    OTXMultilabelClsModel,
)
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import HLabelInfo, LabelInfoTypes
from otx.core.types.task import OTXTrainType

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable

augreg_url = "https://storage.googleapis.com/vit_models/augreg/"
dinov2_url = "https://dl.fbaipublicfiles.com/dinov2/"
pretrained_urls = {
    "vit-tiny": augreg_url
    + "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    "vit-small": augreg_url
    + "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    "vit-base": augreg_url
    + "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "vit-large": augreg_url
    + "L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "dinov2-small": dinov2_url + "dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    "dinov2-base": dinov2_url + "dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
    "dinov2-large": dinov2_url + "dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
    "dinov2-giant": dinov2_url + "dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth",
}


class ForwardExplainMixInForViT(Generic[T_OTXBatchPredEntity, T_OTXBatchDataEntity]):
    """ViT model which can attach a XAI (Explainable AI) branch."""

    explain_mode: bool
    num_classes: int
    model: ImageClassifier

    @torch.no_grad()
    def head_forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Performs model's neck and head forward."""
        if not hasattr(self.model.backbone, "blocks"):
            raise ValueError

        # Part of the last transformer_encoder block (except first LayerNorm)
        target_layer = self.model.backbone.blocks[-1]
        x = x + target_layer.attn(x)
        x = target_layer.mlp(target_layer.norm2(x))

        # Final LayerNorm and neck
        x = self.model.backbone.norm(x)
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

        feat = backbone.forward(images, out_type="raw")[-1]
        x = (feat[:, 0],)

        saliency_map = self.explain_fn(feat)

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

        outputs = {
            "logits": logits,
            "feature_vector": feature_vector,
            "saliency_map": saliency_map,
        }

        if not torch.jit.is_tracing():
            outputs["scores"] = scores
            outputs["labels"] = labels

        return outputs

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
        arch: VIT_ARCH_TYPE = "vit-tiny",
        lora: bool = False,
        pretrained: bool = True,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
        input_size: tuple[int, int] = (224, 224),
        train_type: Literal[OTXTrainType.SUPERVISED, OTXTrainType.SEMI_SUPERVISED] = OTXTrainType.SUPERVISED,
    ) -> None:
        self.arch = arch
        self.lora = lora
        self.pretrained = pretrained

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            input_size=input_size,
            train_type=train_type,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        for key in list(state_dict.keys()):
            new_key = key.replace("patch_embed.projection", "patch_embed.proj")
            new_key = new_key.replace("backbone.ln1", "backbone.norm")
            new_key = new_key.replace("ffn.layers.0.0", "mlp.fc1")
            new_key = new_key.replace("ffn.layers.1", "mlp.fc2")
            new_key = new_key.replace("layers", "blocks")
            new_key = new_key.replace("ln", "norm")
            if new_key != key:
                state_dict[new_key] = state_dict.pop(key)
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
            parts = urlparse(pretrained_urls[self.arch])
            filename = Path(parts.path).name

            cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
            cache_file = cache_dir / filename
            if not Path.exists(cache_file):
                download_url_to_file(pretrained_urls[self.arch], cache_file, "", progress=True)
            model.backbone.load_pretrained(checkpoint_path=cache_file)

        return model

    def _build_model(self, num_classes: int) -> nn.Module:
        init_cfg = [
            {"std": 0.2, "layer": "Linear", "type": "TruncNormal"},
            {"bias": 0.0, "val": 1.0, "layer": "LayerNorm", "type": "Constant"},
        ]
        vit_backbone = VisionTransformer(arch=self.arch, img_size=self.input_size, lora=self.lora)
        if self.train_type == OTXTrainType.SEMI_SUPERVISED:
            return SemiSLClassifier(
                backbone=vit_backbone,
                neck=None,
                head=SemiSLVisionTransformerClsHead(
                    num_classes=num_classes,
                    in_channels=vit_backbone.embed_dim,
                ),
                loss=nn.CrossEntropyLoss(reduction="none"),
            )

        return ImageClassifier(
            backbone=vit_backbone,
            neck=None,
            head=VisionTransformerClsHead(
                num_classes=num_classes,
                in_channels=vit_backbone.embed_dim,
            ),
            loss=nn.CrossEntropyLoss(),
            init_cfg=init_cfg,
        )


class VisionTransformerForMultilabelCls(ForwardExplainMixInForViT, OTXMultilabelClsModel):
    """DeitTiny Model for multi-class classification task."""

    model: ImageClassifier

    def __init__(
        self,
        label_info: LabelInfoTypes,
        arch: VIT_ARCH_TYPE = "vit-tiny",
        lora: bool = False,
        pretrained: bool = True,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiLabelClsMetricCallable,
        torch_compile: bool = False,
        input_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.arch = arch
        self.lora = lora
        self.pretrained = pretrained

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            input_size=input_size,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        for key in list(state_dict.keys()):
            new_key = key.replace("patch_embed.projection", "patch_embed.proj")
            new_key = new_key.replace("backbone.ln1", "backbone.norm")
            new_key = new_key.replace("ffn.layers.0.0", "mlp.fc1")
            new_key = new_key.replace("ffn.layers.1", "mlp.fc2")
            new_key = new_key.replace("layers", "blocks")
            new_key = new_key.replace("ln", "norm")
            if new_key != key:
                state_dict[new_key] = state_dict.pop(key)
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
            parts = urlparse(pretrained_urls[self.arch])
            filename = Path(parts.path).name

            cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
            cache_file = cache_dir / filename
            if not Path.exists(cache_file):
                download_url_to_file(pretrained_urls[self.arch], cache_file, "", progress=True)
            model.backbone.load_pretrained(checkpoint_path=cache_file)
        return model

    def _build_model(self, num_classes: int) -> nn.Module:
        vit_backbone = VisionTransformer(arch=self.arch, img_size=self.input_size, lora=self.lora)
        return ImageClassifier(
            backbone=vit_backbone,
            neck=None,
            head=MultiLabelLinearClsHead(
                num_classes=num_classes,
                in_channels=vit_backbone.embed_dim,
            ),
            loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
        )


class VisionTransformerForHLabelCls(ForwardExplainMixInForViT, OTXHlabelClsModel):
    """DeitTiny Model for hierarchical label classification task."""

    model: ImageClassifier
    label_info: HLabelInfo

    def __init__(
        self,
        label_info: HLabelInfo,
        arch: VIT_ARCH_TYPE = "vit-tiny",
        lora: bool = False,
        pretrained: bool = True,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = HLabelClsMetricCallable,
        torch_compile: bool = False,
        input_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.arch = arch
        self.lora = lora
        self.pretrained = pretrained

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            input_size=input_size,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        for key in list(state_dict.keys()):
            new_key = key.replace("patch_embed.projection", "patch_embed.proj")
            new_key = new_key.replace("backbone.ln1", "backbone.norm")
            new_key = new_key.replace("ffn.layers.0.0", "mlp.fc1")
            new_key = new_key.replace("ffn.layers.1", "mlp.fc2")
            new_key = new_key.replace("layers", "blocks")
            new_key = new_key.replace("ln", "norm")
            if new_key != key:
                state_dict[new_key] = state_dict.pop(key)
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
            parts = urlparse(pretrained_urls[self.arch])
            filename = Path(parts.path).name

            cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
            cache_file = cache_dir / filename
            if not Path.exists(cache_file):
                download_url_to_file(pretrained_urls[self.arch], cache_file, "", progress=True)
            model.backbone.load_pretrained(checkpoint_path=cache_file)
        return model

    def _build_model(self, head_config: dict) -> nn.Module:
        if not isinstance(self.label_info, HLabelInfo):
            raise TypeError(self.label_info)
        init_cfg = [
            {"std": 0.2, "layer": "Linear", "type": "TruncNormal"},
            {"bias": 0.0, "val": 1.0, "layer": "LayerNorm", "type": "Constant"},
        ]
        vit_backbone = VisionTransformer(arch=self.arch, img_size=self.input_size, lora=self.lora)
        return HLabelClassifier(
            backbone=vit_backbone,
            neck=None,
            head=HierarchicalCBAMClsHead(
                in_channels=vit_backbone.embed_dim,
                step_size=1,
                **head_config,
            ),
            multiclass_loss=nn.CrossEntropyLoss(),
            multilabel_loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
            init_cfg=init_cfg,
        )
