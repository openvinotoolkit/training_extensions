# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Semi-SL Classifier Implementation.

Implementing Classifier with dict logic to handle unlabeled datasets with multi-transforms

"""

from __future__ import annotations

import torch

from .base_classifier import ImageClassifier


class SemiSLClassifier(ImageClassifier):
    """Semi-SL Classifier."""

    def extract_feat(
        self,
        inputs: dict[str, torch.Tensor],
        stage: str = "neck",
    ) -> dict[str, tuple | torch.Tensor] | tuple | torch.Tensor:
        """Extract features from the input tensor with shape (N, C, ...).

        Args:
            inputs (dict[str, Tensor]): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            stage (str): Which stage to output the feature. Choose from:

                - "backbone": The output of backbone network. Returns a tuple
                  including multiple stages features.
                - "neck": The output of neck module. Returns a tuple including
                  multiple stages features.

                Defaults to "neck".

        Returns:
            dict[str, tuple | torch.Tensor] | tuple | Tensor: The output of specified stage.
            The output depends on detailed implementation. In general, the Semi-SL
            output is a dict of labeled feats and unlabeled feats.
        """
        if not isinstance(inputs, dict):
            return super().extract_feat(inputs, stage)

        labeled_inputs = inputs["labeled"]
        unlabeled_weak_inputs = inputs["weak_transforms"]
        unlabeled_strong_inputs = inputs["strong_transforms"]

        x = {}
        x["labeled"] = super().extract_feat(labeled_inputs, stage)
        # For weak augmentation inputs, use no_grad to use as a pseudo-label.
        with torch.no_grad():
            x["unlabeled_weak"] = super().extract_feat(unlabeled_weak_inputs, stage)
        x["unlabeled_strong"] = super().extract_feat(unlabeled_strong_inputs, stage)

        return x
