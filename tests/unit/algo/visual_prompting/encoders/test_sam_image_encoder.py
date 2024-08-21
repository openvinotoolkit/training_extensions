# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.algo.visual_prompting.encoders.sam_image_encoder import SAMImageEncoder


class TestSAMImageEncoder:
    @pytest.mark.parametrize(
        ("backbone_type", "expected"),
        [
            ("tiny_vit", "TinyViT"),
            ("vit_b", "ViT"),
        ],
    )
    def test_new(self, backbone_type: str, expected: str) -> None:
        """Test __new__."""
        sam_image_encoder = SAMImageEncoder(backbone_type=backbone_type)

        assert sam_image_encoder.__class__.__name__ == expected

    def test_new_unsupported_backbone(self) -> None:
        """Test __new__ for unsupported backbone."""
        with pytest.raises(ValueError):  # noqa: PT011
            SAMImageEncoder(backbone_type="unsupported_backbone")
