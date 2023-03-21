# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import OrderedDict
from functools import partial

from mmcls.models.builder import CLASSIFIERS
from mmcls.models.classifiers.base import BaseClassifier
from mmcls.models.classifiers.image import ImageClassifier

from otx.algorithms.common.adapters.mmdeploy.utils import is_mmdeploy_enabled
from otx.mpa.modules.utils.task_adapt import map_class_names
from otx.mpa.utils.logger import get_logger

from .sam_classifier_mixin import SAMClassifierMixin

logger = get_logger()


@CLASSIFIERS.register_module()
class SAMImageClassifier(SAMClassifierMixin, ImageClassifier):
    """SAM-enabled ImageClassifier"""

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
    def state_dict_hook(module, state_dict, prefix, *args, **kwargs):
        """Redirect model as output state_dict for OTX model compatibility"""
        backbone_type = type(module.backbone).__name__
        if backbone_type not in ["OTXMobileNetV3", "OTXEfficientNet", "OTXEfficientNetV2"]:
            return

        if backbone_type == "OTXMobileNetV3":
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                if not prefix or k.startswith(prefix):
                    k = k.replace(prefix, "", 1)
                    if k.startswith("backbone"):
                        k = k.replace("backbone.", "", 1)
                    elif k.startswith("head"):
                        k = k.replace("head.", "", 1)
                        if "3" in k:  # MPA uses "classifier.3", OTX uses "classifier.4". Convert for OTX compatibility.
                            k = k.replace("3", "4")
                            if module.multilabel and not module.is_export:
                                v = v.t()
                    k = prefix + k
                state_dict[k] = v

        elif backbone_type == "OTXEfficientNet":
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                if not prefix or k.startswith(prefix):
                    k = k.replace(prefix, "", 1)
                    if k.startswith("backbone"):
                        k = k.replace("backbone.", "", 1)
                    elif k.startswith("head"):
                        k = k.replace("head", "output", 1)
                        if not module.hierarchical and not module.is_export:
                            k = k.replace("fc", "asl")
                            v = v.t()
                    k = prefix + k
                state_dict[k] = v

        elif backbone_type == "OTXEfficientNetV2":
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                if not prefix or k.startswith(prefix):
                    k = k.replace(prefix, "", 1)
                    if k.startswith("backbone"):
                        k = k.replace("backbone.", "", 1)
                    elif k == "head.fc.weight":
                        k = k.replace("head.fc", "model.classifier")
                        if not module.hierarchical and not module.is_export:
                            v = v.t()
                    k = prefix + k
                state_dict[k] = v

        return state_dict

    @staticmethod
    def load_state_dict_pre_hook(module, state_dict, prefix, *args, **kwargs):
        """Redirect input state_dict to model for OTX model compatibility"""
        backbone_type = type(module.backbone).__name__
        if backbone_type not in ["OTXMobileNetV3", "OTXEfficientNet", "OTXEfficientNetV2"]:
            return

        if backbone_type == "OTXMobileNetV3":
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                if not prefix or k.startswith(prefix):
                    k = k.replace(prefix, "", 1)
                    if k.startswith("classifier."):
                        if "4" in k:
                            k = "head." + k.replace("4", "3")
                            if module.multilabel:
                                v = v.t()
                        else:
                            k = "head." + k
                    elif k.startswith("act"):
                        k = "head." + k
                    elif not k.startswith("backbone."):
                        k = "backbone." + k
                    k = prefix + k
                state_dict[k] = v

        elif backbone_type == "OTXEfficientNet":
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                if not prefix or k.startswith(prefix):
                    k = k.replace(prefix, "", 1)
                    if k.startswith("features.") and "activ" not in k:
                        k = "backbone." + k
                    elif k.startswith("output."):
                        k = k.replace("output", "head")
                        if not module.hierarchical:
                            k = k.replace("asl", "fc")
                            v = v.t()
                    k = prefix + k
                state_dict[k] = v

        elif backbone_type == "OTXEfficientNetV2":
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                if not prefix or k.startswith(prefix):
                    k = k.replace(prefix, "", 1)
                    if k.startswith("model.classifier"):
                        k = k.replace("model.classifier", "head.fc")
                        if not module.hierarchical:
                            v = v.t()
                    elif k.startswith("model"):
                        k = "backbone." + k
                    k = prefix + k
                state_dict[k] = v
        else:
            logger.info("conversion is not required.")

    @staticmethod
    def load_state_dict_mixing_hook(model, model_classes, chkpt_classes, chkpt_dict, prefix, *args, **kwargs):
        """Modify input state_dict according to class name matching before weight loading"""
        backbone_type = type(model.backbone).__name__
        if backbone_type not in ["OTXMobileNetV3", "OTXEfficientNet", "OTXEfficientNetV2"]:
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
            chkpt_param = chkpt_dict[chkpt_name]
            for m, c in enumerate(model2chkpt):
                if c >= 0:
                    model_param[m].copy_(chkpt_param[c])

            # Replace checkpoint weight by mixed weights
            chkpt_dict[chkpt_name] = model_param

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
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

    from otx.mpa.modules.hooks.recording_forward_hooks import (
        FeatureVectorHook,
        ReciproCAMHook,
    )

    @FUNCTION_REWRITER.register_rewriter(
        "otx.mpa.modules.models.classifiers.sam_classifier.SAMImageClassifier.extract_feat"
    )
    def sam_image_classifier__extract_feat(ctx, self, img):
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
        "otx.mpa.modules.models.classifiers.sam_classifier.SAMImageClassifier.simple_test"
    )
    def sam_image_classifier__simple_test(ctx, self, img, img_metas):
        feat, backbone_feat = self.extract_feat(img)
        logit = self.head.simple_test(feat)

        if ctx.cfg["dump_features"]:
            saliency_map = ReciproCAMHook(self).func(backbone_feat)
            feature_vector = FeatureVectorHook.func(backbone_feat)
            return logit, feature_vector, saliency_map

        return logit
