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
from otx.algo.classification.classifier import ImageClassifier, SemiSLClassifier
from otx.algo.classification.heads import (
    HierarchicalLinearClsHead,
    MultiLabelLinearClsHead,
    OTXSemiSLVisionTransformerClsHead,
    VisionTransformerClsHead,
)
from otx.algo.classification.losses import AsymmetricAngularLossWithIgnore
from otx.algo.classification.utils import get_classification_layers
from otx.algo.explain.explain_algo import ViTReciproCAM, feature_vector_fn
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.data.entity.base import OTXBatchLossEntity, T_OTXBatchDataEntity, T_OTXBatchPredEntity
from otx.core.data.entity.classification import (
    CLASSIFICATION_BATCH_DATA_ENTITY,
    CLASSIFICATION_BATCH_PRED_ENTITY,
    HlabelClsBatchPredEntity,
    MulticlassClsBatchPredEntity,
    MultilabelClsBatchPredEntity,
)
from otx.core.metrics.accuracy import DefaultClsMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.classification import OTXClassificationModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import HLabelInfo, LabelInfoTypes
from otx.core.types.task import OTXTaskType, OTXTrainType

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallablePerTask


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


class VisionTransformerForClassification(ForwardExplainMixInForViT, OTXClassificationModel):
    """Vision Transformer model for classification tasks.

    Args:
        label_info (LabelInfoTypes): Information about the labels.
        arch (VIT_ARCH_TYPE, optional): Architecture type of the Vision Transformer. Defaults to "vit-tiny".
        lora (bool, optional): Whether to use LoRA attention. Defaults to False.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        optimizer (OptimizerCallable, optional): Optimizer for training. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Learning rate scheduler.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallablePerTask, optional): Metric for evaluation. Defaults to DefaultClsMetricCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
        task (Literal[OTXTaskType.MULTI_CLASS_CLS, OTXTaskType.MULTI_LABEL_CLS, OTXTaskType.H_LABEL_CLS], optional):
            Type of classification task. Defaults to OTXTaskType.MULTI_CLASS_CLS.
        train_type (Literal[OTXTrainType.SUPERVISED, OTXTrainType.SEMI_SUPERVISED], optional): Type of training.
            Defaults to OTXTrainType.SUPERVISED.
    """

    model: ImageClassifier

    def __init__(
        self,
        label_info: LabelInfoTypes,
        arch: VIT_ARCH_TYPE = "vit-tiny",
        lora: bool = False,
        pretrained: bool = True,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallablePerTask = DefaultClsMetricCallable,
        torch_compile: bool = False,
        task: Literal[
            OTXTaskType.MULTI_CLASS_CLS,
            OTXTaskType.MULTI_LABEL_CLS,
            OTXTaskType.H_LABEL_CLS,
        ] = OTXTaskType.MULTI_CLASS_CLS,
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
            task=task,
            train_type=train_type,
        )

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
                download_url_to_file(pretrained_urls[self.arch], str(cache_file), "", progress=True)
            model.backbone.load_pretrained(checkpoint_path=cache_file)

        return model

    def _build_model(self, num_classes: int) -> nn.Module:
        init_cfg = [
            {"std": 0.2, "layer": "Linear", "type": "TruncNormal"},
            {"bias": 0.0, "val": 1.0, "layer": "LayerNorm", "type": "Constant"},
        ]
        vit_backbone = VisionTransformer(arch=self.arch, img_size=224, lora=self.lora)
        classifier = ImageClassifier if self.train_type == OTXTrainType.SUPERVISED else SemiSLClassifier
        head = self._build_head(num_classes, vit_backbone.embed_dim)
        return classifier(
            backbone=vit_backbone,
            neck=None,
            head=head,
            init_cfg=init_cfg,
        )

    def _build_head(self, num_classes: int, embed_dim: int) -> nn.Module:
        if self.task == OTXTaskType.MULTI_CLASS_CLS:
            loss = nn.CrossEntropyLoss(reduction="none")
            if self.train_type == OTXTrainType.SEMI_SUPERVISED:
                return OTXSemiSLVisionTransformerClsHead(
                    num_classes=num_classes,
                    in_channels=embed_dim,
                    loss=loss,
                )
            return VisionTransformerClsHead(
                num_classes=num_classes,
                in_channels=embed_dim,
                topk=(1, 5) if num_classes >= 5 else (1,),
                loss=loss,
            )
        if self.task == OTXTaskType.MULTI_LABEL_CLS:
            if self.train_type == OTXTrainType.SEMI_SUPERVISED:
                msg = "Semi-supervised learning is not supported for multi-label classification."
                raise NotImplementedError(msg)
            return MultiLabelLinearClsHead(
                num_classes=num_classes,
                in_channels=embed_dim,
                loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
            )
        if self.task == OTXTaskType.H_LABEL_CLS:
            if self.train_type == OTXTrainType.SEMI_SUPERVISED:
                msg = "Semi-supervised learning is not supported for h-label classification."
                raise NotImplementedError(msg)
            if not isinstance(self.label_info, HLabelInfo):
                msg = "LabelInfo should be HLabelInfo for H-label classification."
                raise ValueError(msg)
            head_config = deepcopy(self.label_info.as_head_config_dict())
            head_config["num_classes"] = num_classes
            return HierarchicalLinearClsHead(
                in_channels=embed_dim,
                multiclass_loss=nn.CrossEntropyLoss(),
                multilabel_loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
                **head_config,
            )
        msg = f"Unsupported task: {self.task}"
        raise NotImplementedError(msg)

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
        label_type = {
            OTXTaskType.MULTI_CLASS_CLS: "multiclass",
            OTXTaskType.MULTI_LABEL_CLS: "multilabel",
            OTXTaskType.H_LABEL_CLS: "hlabel",
        }[self.task]
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, label_type, add_prefix)

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: CLASSIFICATION_BATCH_DATA_ENTITY,
    ) -> CLASSIFICATION_BATCH_PRED_ENTITY | OTXBatchLossEntity:
        if self.training:
            return OTXBatchLossEntity(loss=outputs)

        entity_kwargs = {
            "batch_size": inputs.batch_size,
            "images": inputs.images,
            "imgs_info": inputs.imgs_info,
        }

        if self.explain_mode:
            # TODO(harimkang): Let's see if we can move it to common
            entity_kwargs["scores"] = outputs["scores"]
            entity_kwargs["labels"] = outputs["labels"]
            entity_kwargs["saliency_map"] = outputs["saliency_map"]
            entity_kwargs["feature_vector"] = outputs["feature_vector"]
        elif self.task == OTXTaskType.H_LABEL_CLS and isinstance(outputs, dict):
            entity_kwargs["scores"] = outputs["scores"]
            entity_kwargs["labels"] = outputs["labels"]
        else:
            # To list, batch-wise
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs["logits"]
            scores = torch.unbind(logits, 0)
            labels = logits.argmax(-1, keepdim=True).unbind(0)
            entity_kwargs["scores"] = scores
            entity_kwargs["labels"] = labels

        if self.task == OTXTaskType.MULTI_CLASS_CLS:
            return MulticlassClsBatchPredEntity(**entity_kwargs)  # type: ignore[arg-type]
        if self.task == OTXTaskType.MULTI_LABEL_CLS:
            return MultilabelClsBatchPredEntity(**entity_kwargs)  # type: ignore[arg-type]
        if self.task == OTXTaskType.H_LABEL_CLS:
            return HlabelClsBatchPredEntity(**entity_kwargs)  # type: ignore[arg-type]
        msg = f"Task type {self.task} is not supported."
        raise NotImplementedError(msg)
