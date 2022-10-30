"""SAM Classifier.

Original paper:
- 'Sharpness-Aware Minimization for Efficiently Improving Generalization,' https://arxiv.org/abs/2010.01412.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=unused-argument, too-many-branches, invalid-name
# pylint: disable=too-many-locals, too-many-nested-blocks, abstract-method

import functools
from collections import OrderedDict

import torch
from mmcls.models.builder import CLASSIFIERS
from mmcls.models.classifiers.base import BaseClassifier
from mmcls.models.classifiers.image import ImageClassifier
from mpa.modules.hooks.auxiliary_hooks import FeatureVectorHook, SaliencyMapHook
from mpa.modules.utils.task_adapt import map_class_names
from mpa.utils.logger import get_logger

logger = get_logger()


@CLASSIFIERS.register_module(force=True)
class SAMClassifier(BaseClassifier):
    """SAM-enabled BaseClassifier."""

    def train_step(self, data, optimizer):
        """Train batch and save current batch data to compute SAM gradient."""

        # Rest of SAM logics are implented in SAMOptimizerHook
        self.current_batch = data  # pylint: disable=attribute-defined-outside-init

        return super().train_step(data, optimizer)


@CLASSIFIERS.register_module(force=True)
class SAMImageClassifier(ImageClassifier):
    """SAM-enabled ImageClassifier."""

    def __init__(self, task_adapt=None, **kwargs):
        if "multilabel" in kwargs:
            self.multilabel = kwargs.pop("multilabel")
        if "hierarchical" in kwargs:
            self.hierarchical = kwargs.pop("hierarchical")
        super().__init__(**kwargs)
        self.is_export = False
        self.featuremap = None
        self.current_batch = None
        # Hooks for redirect state_dict load/save
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(functools.partial(self.load_state_dict_pre_hook, self))
        if task_adapt:
            self._register_load_state_dict_pre_hook(
                functools.partial(
                    self.load_state_dict_mixing_hook,
                    self,  # model
                    task_adapt["dst_classes"],  # model_classes
                    task_adapt["src_classes"],  # chkpt_classes
                )
            )

    def train_step(self, data, optimizer):
        """Train batch and save current batch data to compute SAM gradient."""

        # Rest of SAM logics are implented in SAMOptimizerHook
        self.current_batch = data

        return super().train_step(data, optimizer)

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

        losses = {}

        if self.multilabel or self.hierarchical:
            loss = self.head.forward_train(x, gt_label, **kwargs)
        else:
            gt_label = gt_label.squeeze(dim=1)
            loss = self.head.forward_train(x, gt_label)

        losses.update(loss)

        return losses

    @staticmethod
    def state_dict_hook(module, state_dict, *args, **kwargs):
        """Redirect model as output state_dict for OTX model compatibility."""

        backbone_type = type(module.backbone).__name__
        if backbone_type not in ["OTXMobileNetV3", "OTXEfficientNet", "OTXEfficientNetV2"]:
            return None

        output = OrderedDict()
        if backbone_type == "OTXMobileNetV3":
            for k, v in state_dict.items():
                if k.startswith("backbone"):
                    k = k.replace("backbone.", "")
                elif k.startswith("head"):
                    k = k.replace("head.", "")
                    if "3" in k:  # MPA uses "classifier.3", OTX uses "classifier.4". Convert for OTX compatibility.
                        k = k.replace("3", "4")
                        if module.multilabel and not module.is_export:
                            v = v.t()
                output[k] = v

        elif backbone_type == "OTXEfficientNet":
            for k, v in state_dict.items():
                if k.startswith("backbone"):
                    k = k.replace("backbone.", "")
                elif k.startswith("head"):
                    k = k.replace("head", "output")
                    if not module.hierarchical and not module.is_export:
                        k = k.replace("fc", "asl")
                        v = v.t()
                output[k] = v

        elif backbone_type == "OTXEfficientNetV2":
            for k, v in state_dict.items():
                if k.startswith("backbone"):
                    k = k.replace("backbone.", "")
                elif k == "head.fc.weight":
                    k = k.replace("head.fc", "model.classifier")
                    if not module.hierarchical and not module.is_export:
                        v = v.t()
                output[k] = v

        return output

    @staticmethod
    def load_state_dict_pre_hook(module, state_dict, *args, **kwargs):
        """Redirect input state_dict to model for OTX model compatibility."""

        backbone_type = type(module.backbone).__name__
        if backbone_type not in ["OTXMobileNetV3", "OTXEfficientNet", "OTXEfficientNetV2"]:
            return

        if backbone_type == "OTXMobileNetV3":
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                if k.startswith("classifier."):
                    if "4" in k:
                        k = "head." + k.replace("4", "3")
                        if module.multilabel:
                            v = v.t()
                    else:
                        k = "head." + k
                elif not k.startswith("backbone."):
                    k = "backbone." + k
                state_dict[k] = v

        elif backbone_type == "OTXEfficientNet":
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                if k.startswith("features.") and "activ" not in k:
                    k = "backbone." + k
                elif k.startswith("output."):
                    k = k.replace("output", "head")
                    if not module.hierarchical:
                        k = k.replace("asl", "fc")
                        v = v.t()
                state_dict[k] = v

        elif backbone_type == "OTXEfficientNetV2":
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                if k.startswith("model.classifier"):
                    k = k.replace("model.classifier", "head.fc")
                    if not module.hierarchical:
                        v = v.t()
                elif k.startswith("model"):
                    k = "backbone." + k
                state_dict[k] = v
        else:
            logger.info("conversion is not required.")

    @staticmethod
    def load_state_dict_mixing_hook(model, model_classes, chkpt_classes, chkpt_dict, prefix, *args, **kwargs):
        """Modify input state_dict according to class name matching before weight loading."""

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
        """Directly extract features from the backbone + neck.

        Overriding for OpenVINO export with features.
        """
        x = self.backbone(img)
        if torch.onnx.is_in_onnx_export():
            self.featuremap = x

        if self.with_neck:
            x = self.neck(x)
        return x

    def simple_test(self, img, img_metas):
        """Test without augmentation.

        Overriding for OpenVINO export with features
        """
        x = self.extract_feat(img)
        logits = self.head.simple_test(x)
        if self.featuremap is not None and torch.onnx.is_in_onnx_export():
            saliency_map = SaliencyMapHook.func(self.featuremap)
            feature_vector = FeatureVectorHook.func(self.featuremap)
            return logits, feature_vector, saliency_map
        return logits
