# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from typing import Any

import pytest
from omegaconf import OmegaConf
from otx.core.data.transform_libs.torchvision import TorchVisionTransformLib
from torchvision.transforms import v2


class TestTorchVisionTransformLib:
    @pytest.fixture()
    def fxt_config(self) -> list[dict[str, Any]]:
        # >>> transforms = v2.Compose([
        # >>>     v2.RandomResizedCrop(size=(224, 224), antialias=True),
        # >>>     v2.RandomHorizontalFlip(p=0.5),
        # >>>     v2.ToDtype(torch.float32, scale=True),
        # >>>     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # >>> ])
        prefix = "torchvision.transforms.v2"
        cfg = f"""
        transforms:
          - _target_: {prefix}.RandomResizedCrop
            size: [224, 224]
            antialias: True
          - _target_: {prefix}.RandomHorizontalFlip
            p: 0.5
          - _target_: {prefix}.ToDtype
            dtype: ${{as_torch_dtype:torch.float32}}
            scale: True
          - _target_: {prefix}.Normalize
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
        """
        return OmegaConf.create(cfg)

    def test_transform(
        self,
        fxt_config,
        fxt_dataset_and_data_entity_cls,
        fxt_mock_dm_subset,
    ) -> None:
        transform = TorchVisionTransformLib.generate(fxt_config)
        assert isinstance(transform, v2.Compose)

        dataset_cls, data_entity_cls = fxt_dataset_and_data_entity_cls
        dataset = dataset_cls(
            dm_subset=fxt_mock_dm_subset,
            transforms=transform,
            mem_cache_img_max_size=None,
        )

        item = dataset[0]
        assert isinstance(item, data_entity_cls)

    @pytest.mark.xfail(
        reason="Our `ImageInfo` entity is registered to TorchVision V2 transforms yet",
    )
    def test_image_info(
        self,
        fxt_config,
        fxt_dataset_and_data_entity_cls,
        fxt_mock_dm_subset,
    ) -> None:
        transform = TorchVisionTransformLib.generate(fxt_config)
        assert isinstance(transform, v2.Compose)

        dataset_cls, data_entity_cls = fxt_dataset_and_data_entity_cls
        dataset = dataset_cls(
            dm_subset=fxt_mock_dm_subset,
            transforms=transform,
            mem_cache_img_max_size=None,
        )

        item = dataset[0]
        assert item.img_info.img_shape == item.image.shape[1:]
