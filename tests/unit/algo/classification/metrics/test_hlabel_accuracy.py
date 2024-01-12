# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from otx.algo.classification.metrics import HLabelAccuracy


class TestHLabelAccuracy:
    @pytest.fixture()
    def hlabel_accuracy(self) -> HLabelAccuracy:
        # You may need to adjust the parameters based on your actual use case
        return HLabelAccuracy(
            num_multiclass_heads=2,
            num_multilabel_classes=3,
            threshold_multilabel=0.5,
            head_idx_to_logits_range={"head1": (0, 5), "head2": (5, 10)},
        )

    def test_update_and_compute(self, hlabel_accuracy) -> None:
        preds = torch.rand((10, 5))
        target = torch.randint(0, 2, (10, 5))  # Replace the dimensions with actual dimensions

        hlabel_accuracy.update(preds, target)
        result = hlabel_accuracy.compute()

        assert isinstance(result, torch.Tensor)

    def test_multilabel_only(self) -> None:
        # Test when only multilabel heads are present (should raise an exception)
        with pytest.raises(ValueError, match="The number of multiclass heads should be larger than 0"):
            HLabelAccuracy(
                num_multiclass_heads=0,
                num_multilabel_classes=3,
                threshold_multilabel=0.5,
                head_idx_to_logits_range=None,
            )
