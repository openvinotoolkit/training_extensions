from __future__ import annotations

import torch

from .base_classifier import ImageClassifier


class SemiSLClassifier(ImageClassifier):
    """Semi-SL Classifier."""

    def extract_feat(self, inputs: dict[str, torch.Tensor], stage: str = "neck") -> dict[str, tuple | torch.Tensor]:
        """Extract features from the input tensor with shape (N, C, ...).

        Args:
            inputs (Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            stage (str): Which stage to output the feature. Choose from:

                - "backbone": The output of backbone network. Returns a tuple
                  including multiple stages features.
                - "neck": The output of neck module. Returns a tuple including
                  multiple stages features.
                - "pre_logits": The feature before the final classification
                  linear layer. Usually returns a tensor.

                Defaults to "neck".

        Returns:
            tuple | Tensor: The output of specified stage.
            The output depends on detailed implementation. In general, the
            output of backbone and neck is a tuple and the output of
            pre_logits is a tensor.
        """
        if not isinstance(inputs, dict):
            return super().extract_feat(inputs, stage)

        labeled_inputs = inputs["labeled"]
        unlabeled_inputs = inputs["unlabeled"]

        x = {}
        x["labeled"] = super().extract_feat(labeled_inputs, stage)
        x["unlabeled"] = super().extract_feat(unlabeled_inputs, stage)

        return x
