"""Module defining for clip classifiers."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import clip
import torch
from mmcls.models.builder import CLASSIFIERS

from .sam_classifier import SAMImageClassifier


@CLASSIFIERS.register_module()
class MMCLSVisionTransformerSAMImageClassifier(SAMImageClassifier):
    """MMCLSVisionTransformerSAMImageClassifier class."""

    def extract_feat(self, img):
        """Extracts the features from the input image using a Vision Transformer backbone.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The extracted features (class token) as a tensor.
        """
        x = super().extract_feat(img)
        if isinstance(x, tuple):
            x = x[-1]
        if self.with_neck:
            cls_token = x
        else:
            _, cls_token = x
        return cls_token


@CLASSIFIERS.register_module()
class FrozenMMCLSVisionTransformerSAMImageClassifier(MMCLSVisionTransformerSAMImageClassifier):
    """FrozenMMCLSVisionTransformerSAMImageClassifier class."""

    @torch.no_grad()
    def extract_feat(self, img):
        """Extracts the features from the input image using a CLIP Vision Transformer backbone without autograd.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The extracted features (class token) as a tensor.
        """
        return super().extract_feat(img)


@CLASSIFIERS.register_module()
class CLIPVisionTransformerSAMImageClassifier(SAMImageClassifier):
    """CLIPVisionTransformerSAMImageClassifier class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backbone = clip.load("ViT-L/14@336px", "cpu")[0].visual
        self.backbone.to(device)


@CLASSIFIERS.register_module()
class FrozenCLIPVisionTransformerSAMImageClassifier(CLIPVisionTransformerSAMImageClassifier):
    """FrozenCLIPVisionTransformerSAMImageClassifier class."""

    @torch.no_grad()
    def extract_feat(self, img):
        """Extracts the features from the input image using a CLIP Vision Transformer backbone without autograd.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The extracted features (class token) as a tensor.
        """
        return super().extract_feat(img)


@CLASSIFIERS.register_module()
class SAMImageClassifierTrainOnlyHead(SAMImageClassifier):
    """SAMImageClassifierTrainOnlyHead class."""

    def extract_feat(self, img):
        """Extracts the features from the input image using a Vision Transformer backbone.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The extracted features (class token) as a tensor.
        """
        return img
