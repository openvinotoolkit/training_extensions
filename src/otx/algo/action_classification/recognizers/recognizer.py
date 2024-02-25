# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom 3D recognizer for OTX."""

import torch
from mmaction.models import MODELS
from mmaction.models.recognizers import Recognizer3D


@MODELS.register_module()
class OTXRecognizer3D(Recognizer3D):
    """Custom 3d recognizer class for OTX.

    This is for patching forward function during export procedure.
    """

    def _forward(self, inputs: torch.Tensor, stage: str = "backbone", **kwargs) -> torch.Tensor:
        """Network forward process for export procedure.

        Args:
            inputs (torch.Tensor): Raw Inputs of the recognizer.
            stage (str): Which stage to output the features.

        """
        feats, predict_kwargs = self.extract_feat(inputs, test_mode=True)
        cls_scores = self.cls_head(feats, **predict_kwargs)
        num_segs = cls_scores.shape[0] // inputs.shape[1]
        return self.cls_head.average_clip(cls_scores, num_segs=num_segs)
