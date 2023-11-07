"""Module for defining SAMClassifier for classification task."""
# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from functools import partial

import torch
from mmcls.models.backbones.vision_transformer import VisionTransformer
from mmcls.models.builder import CLASSIFIERS
from mmcls.models.classifiers.image import ImageClassifier
from mmcls.models.utils import resize_pos_embed

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import ViTReciproCAMHook
from otx.algorithms.common.adapters.mmdeploy.utils import is_mmdeploy_enabled
from otx.algorithms.common.utils.task_adapt import map_class_names
from otx.utils.logger import get_logger

from .mixin import ClsLossDynamicsTrackingMixin, SAMClassifierMixin

logger = get_logger()


def is_hierarchical_chkpt(chkpt: dict):
    """Detect whether previous checkpoint is hierarchical or not."""
    for k, v in chkpt.items():
        if "fc" in k:
            return True
    return False


@CLASSIFIERS.register_module()
class CustomImageClassifier(SAMClassifierMixin, ClsLossDynamicsTrackingMixin, ImageClassifier):
    """SAM-enabled ImageClassifier."""

    def __init__(self, task_adapt=None, **kwargs):
        if "multilabel" in kwargs:
            self.multilabel = kwargs.pop("multilabel")
        if "hierarchical" in kwargs:
            self.hierarchical = kwargs.pop("hierarchical")
        super().__init__(**kwargs)
        self.is_export = False
        # Hooks for redirect state_dict load/save
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(partial(self.load_state_dict_pre_hook, self))
        if task_adapt:
            self._register_load_state_dict_pre_hook(
                partial(
                    self.load_state_dict_mixing_hook,
                    self,  # model
                    task_adapt["dst_classes"],  # model_classes
                    task_adapt["src_classes"],  # chkpt_classes
                )
            )

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.

            **kwargs (Any): Addition keyword arguments.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        x = self.extract_feat(img)

        losses = dict()

        if self.multilabel or self.hierarchical:
            loss = self.head.forward_train(x, gt_label, **kwargs)
        else:
            gt_label = gt_label.squeeze(dim=1)
            loss = self.head.forward_train(x, gt_label)

        losses.update(loss)

        return losses

    @staticmethod
    def state_dict_hook(module, state_dict, prefix, *args, **kwargs):  # noqa: C901
        # pylint: disable=unused-argument, too-many-branches
        """Redirect model as output state_dict for OTX model compatibility."""
        backbone_type = type(module.backbone).__name__
        if backbone_type not in ["OTXMobileNetV3", "OTXEfficientNet", "OTXEfficientNetV2"]:
            return None

        if backbone_type == "OTXMobileNetV3":  # pylint: disable=too-many-nested-blocks
            for key in list(state_dict.keys()):
                val = state_dict.pop(key)
                if not prefix or key.startswith(prefix):
                    key = key.replace(prefix, "", 1)
                    if key.startswith("backbone"):
                        key = key.replace("backbone.", "", 1)
                    elif key.startswith("head"):
                        key = key.replace("head.", "", 1)
                        if "3" in key:  # OTX uses "classifier.4". Convert for OTX compatibility.
                            key = key.replace("3", "4")
                            if module.multilabel and not module.is_export:
                                val = val.t()
                    key = prefix + key
                state_dict[key] = val

        elif backbone_type == "OTXEfficientNet":
            for key in list(state_dict.keys()):
                val = state_dict.pop(key)
                if not prefix or key.startswith(prefix):
                    key = key.replace(prefix, "", 1)
                    if key.startswith("backbone"):
                        key = key.replace("backbone.", "", 1)
                    elif key.startswith("head"):
                        key = key.replace("head", "output", 1)
                        if not module.hierarchical and not module.is_export:
                            key = key.replace("fc", "asl")
                            val = val.t()
                    key = prefix + key
                state_dict[key] = val

        elif backbone_type == "OTXEfficientNetV2":
            for key in list(state_dict.keys()):
                val = state_dict.pop(key)
                if not prefix or key.startswith(prefix):
                    key = key.replace(prefix, "", 1)
                    if key.startswith("backbone"):
                        key = key.replace("backbone.", "", 1)
                    elif key == "head.fc.weight":
                        key = key.replace("head.fc", "model.classifier")
                        if not module.hierarchical and not module.is_export:
                            val = val.t()
                    key = prefix + key
                state_dict[key] = val

        return state_dict

    @staticmethod
    def load_state_dict_pre_hook(module, state_dict, prefix, *args, **kwargs):  # noqa: C901
        # pylint: disable=unused-argument, too-many-branches
        """Redirect input state_dict to model for OTX model compatibility."""
        backbone_type = type(module.backbone).__name__
        if backbone_type not in ["OTXMobileNetV3", "OTXEfficientNet", "OTXEfficientNetV2"]:
            return

        if backbone_type == "OTXMobileNetV3":  # pylint: disable=too-many-nested-blocks
            for key in list(state_dict.keys()):
                val = state_dict.pop(key)
                if not prefix or key.startswith(prefix):
                    key = key.replace(prefix, "", 1)
                    if key.startswith("classifier."):
                        if "4" in key:
                            key = "head." + key.replace("4", "3")
                            if module.multilabel:
                                val = val.t()
                        else:
                            key = "head." + key
                    elif key.startswith("act"):
                        key = "head." + key
                    elif not key.startswith("backbone."):
                        key = "backbone." + key
                    key = prefix + key
                state_dict[key] = val

        elif backbone_type == "OTXEfficientNet":
            for key in list(state_dict.keys()):
                val = state_dict.pop(key)
                if not prefix or key.startswith(prefix):
                    key = key.replace(prefix, "", 1)
                    if key.startswith("features.") and "activ" not in key:
                        key = "backbone." + key
                    elif key.startswith("output."):
                        key = key.replace("output", "head")
                        if not module.hierarchical:
                            key = key.replace("asl", "fc")
                            val = val.t()
                    key = prefix + key
                state_dict[key] = val

        elif backbone_type == "OTXEfficientNetV2":
            for key in list(state_dict.keys()):
                val = state_dict.pop(key)
                if not prefix or key.startswith(prefix):
                    key = key.replace(prefix, "", 1)
                    if key.startswith("model.classifier"):
                        key = key.replace("model.classifier", "head.fc")
                        if not module.hierarchical:
                            val = val.t()
                    elif key.startswith("model"):
                        key = "backbone." + key
                    key = prefix + key
                state_dict[key] = val
        else:
            logger.info("conversion is not required.")

    @staticmethod
    def load_state_dict_mixing_hook(
        model, model_classes, chkpt_classes, chkpt_dict, prefix, *args, **kwargs
    ):  # pylint: disable=unused-argument, too-many-branches, too-many-locals
        """Modify input state_dict according to class name matching before weight loading.

        If previous training is hierarchical training,
        then the current training should be hierarchical training. vice versa.

        """
        backbone_type = type(model.backbone).__name__
        if backbone_type not in ["OTXMobileNetV3", "OTXEfficientNet", "OTXEfficientNetV2"]:
            return

        if model.hierarchical != is_hierarchical_chkpt(chkpt_dict):
            return

        # Dst to src mapping index
        model_classes = list(model_classes)
        chkpt_classes = list(chkpt_classes)
        model2chkpt = map_class_names(model_classes, chkpt_classes)
        logger.info(f"{chkpt_classes} -> {model_classes} ({model2chkpt})")
        model_dict = model.state_dict()

        if backbone_type == "OTXMobileNetV3":
            if model.multilabel:
                param_names = ["classifier.4.weight"]
            else:
                param_names = ["classifier.4.weight", "classifier.4.bias"]

        elif backbone_type == "OTXEfficientNet":
            if not model.hierarchical:
                param_names = ["output.asl.weight"]
            else:
                param_names = ["output.fc.weight"]

        elif backbone_type == "OTXEfficientNetV2":
            param_names = [
                "model.classifier.weight",
            ]
            if "head.fc.bias" in chkpt_dict.keys():
                param_names.append("head.fc.bias")

        for model_name in param_names:
            model_param = model_dict[model_name].clone()
            if backbone_type == "OTXMobileNetV3":
                chkpt_name = "head." + model_name.replace("4", "3")
                if model.multilabel:
                    model_param = model_param.t()
            elif backbone_type in "OTXEfficientNet":
                chkpt_name = model_name.replace("output", "head")
                if not model.hierarchical:
                    chkpt_name = chkpt_name.replace("asl", "fc")
                    model_param = model_param.t()

            elif backbone_type in "OTXEfficientNetV2":
                if model_name.endswith("bias"):
                    chkpt_name = model_name
                else:
                    chkpt_name = model_name.replace("model.classifier", "head.fc")
                    if not model.hierarchical:
                        model_param = model_param.t()

            if model_name not in model_dict or chkpt_name not in chkpt_dict:
                logger.info(f"Skipping weight copy: {chkpt_name}")
                continue

            # Mix weights
            # NOTE: Label mix is not supported for H-label classification.
            if not model.hierarchical:
                chkpt_param = chkpt_dict[chkpt_name]
                for module, c in enumerate(model2chkpt):
                    if c >= 0:
                        model_param[module].copy_(chkpt_param[c])

                # Replace checkpoint weight by mixed weights
                chkpt_dict[chkpt_name] = model_param

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck.

        Overriding for OpenVINO export with features
        """
        x = self.backbone(img)
        # For Global Backbones (det/seg/etc..),
        # In case of tuple or list, only the feat of the last layer is used.
        if isinstance(x, (tuple, list)):
            x = x[-1]

        if self.with_neck:
            x = self.neck(x)

        return x


if is_mmdeploy_enabled():
    from mmdeploy.core import FUNCTION_REWRITER

    from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (  # pylint: disable=ungrouped-imports
        FeatureVectorHook,
        ReciproCAMHook,
    )

    def _extract_vit_feat(model, x):
        """Modified forward from mmcls.models.backbones.vision_transformer.VisionTransformer.forward()."""
        B = x.shape[0]
        x, patch_resolution = model.backbone.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = model.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + resize_pos_embed(
            model.backbone.pos_embed,
            model.backbone.patch_resolution,
            patch_resolution,
            mode=model.backbone.interpolate_mode,
            num_extra_tokens=model.backbone.num_extra_tokens,
        )
        x = model.backbone.drop_after_pos(x)

        if not model.backbone.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        feat = None
        layernorm_feat = None
        for i, layer in enumerate(model.backbone.layers):
            if i == len(model.backbone.layers) - 1:
                layernorm_feat = layer.norm1(x)

            x = layer(x)

            if i == len(model.backbone.layers) - 1 and model.backbone.final_norm:
                x = model.backbone.norm1(x)

            if i in model.backbone.out_indices:
                B, _, C = x.shape
                if model.backbone.with_cls_token:
                    patch_token = x[:, 1:].reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = x[:, 0]
                else:
                    patch_token = x.reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = None
                if model.backbone.output_cls_token:
                    feat = [patch_token, cls_token]
                else:
                    feat = patch_token
        if model.with_neck:
            feat = model.neck(feat)
        return feat, layernorm_feat

    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.classification.adapters.mmcls.models.classifiers.CustomImageClassifier.extract_feat"
    )
    def sam_image_classifier__extract_feat(ctx, self, img):  # pylint: disable=unused-argument
        """Feature extraction function for SAMClassifier with mmdeploy."""
        feat = self.backbone(img)
        # For Global Backbones (det/seg/etc..),
        # In case of tuple or list, only the feat of the last layer is used.
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]
        backbone_feat = feat
        if self.with_neck:
            feat = self.neck(feat)
        return feat, backbone_feat

    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.classification.adapters.mmcls.models.classifiers.CustomImageClassifier.simple_test"
    )
    def sam_image_classifier__simple_test(ctx, self, img, img_metas):  # pylint: disable=unused-argument
        """Simple test function used for inference for SAMClassifier with mmdeploy."""
        vit_backbone = isinstance(self.backbone, VisionTransformer)
        if vit_backbone:
            feat, layernorm_feat = _extract_vit_feat(self, img)
        else:
            feat, backbone_feat = self.extract_feat(img)
        logit = self.head.simple_test(feat)

        if ctx.cfg["dump_features"]:
            if vit_backbone:
                assert self.backbone.with_cls_token
                _, cls_token = feat
                feature_vector = cls_token
                saliency_map = ViTReciproCAMHook(self).func(layernorm_feat)
            else:
                saliency_map = ReciproCAMHook(self).func(backbone_feat)
                feature_vector = FeatureVectorHook.func(backbone_feat)
            return logit, feature_vector, saliency_map

        return logit
