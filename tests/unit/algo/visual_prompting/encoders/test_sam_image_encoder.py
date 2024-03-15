# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.algo.visual_prompting.encoders.sam_image_encoder import SAMImageEncoder


class TestSAMImageEncoder:
    @pytest.mark.parametrize(
        ("backbone", "expected"),
        [
            ("tiny_vit", "TinyViT"),
        ],
    )
    def test_new(self, backbone: str, expected: str) -> None:
        """Test __new__."""
        sam_image_encoder = SAMImageEncoder(backbone=backbone)

        assert sam_image_encoder.__class__.__name__ == expected
