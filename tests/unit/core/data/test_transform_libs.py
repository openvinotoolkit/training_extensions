# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

from typing import Any

import pytest
import torch
from lightning.pytorch.cli import instantiate_class
from omegaconf import OmegaConf
from otx.core.config.data import SubsetConfig
from otx.core.data.dataset.action_classification import OTXActionClsDataset
from otx.core.data.dataset.classification import HLabelInfo
from otx.core.data.dataset.instance_segmentation import OTXInstanceSegDataset
from otx.core.data.transform_libs.torchvision import TorchVisionTransformLib
from otx.core.types.image import ImageColorChannel
from torchvision.transforms import v2


class TestTorchVisionTransformLib:
    @pytest.fixture(params=["from_dict", "from_list", "from_compose"])
    def fxt_config(self, request) -> list[dict[str, Any]]:
        if request.param == "from_compose":
            return v2.Compose(
                [
                    v2.RandomResizedCrop(size=(224, 224), antialias=True),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ],
            )
        prefix = "torchvision.transforms.v2"
        cfg = f"""
        transforms:
          - class_path: {prefix}.RandomResizedCrop
            init_args:
                size: [224, 224]
                antialias: True
          - class_path: {prefix}.RandomHorizontalFlip
            init_args:
                p: 0.5
          - class_path: {prefix}.ToDtype
            init_args:
                dtype: ${{as_torch_dtype:torch.float32}}
                scale: True
          - class_path: {prefix}.Normalize
            init_args:
                mean: [0.485, 0.456, 0.406]
                std: [0.229, 0.224, 0.225]
        """
        created = OmegaConf.create(cfg)
        if request.param == "from_obj":
            return SubsetConfig(
                batch_size=1,
                subset_name="dummy",
                transforms=[instantiate_class(args=(), init=transform) for transform in created.transforms],
            )
        return created

    def test_transform(
        self,
        mocker,
        fxt_config,
        fxt_dataset_and_data_entity_cls,
        fxt_mock_dm_subset,
        fxt_mock_hlabelinfo,
    ) -> None:
        transform = TorchVisionTransformLib.generate(fxt_config)
        assert isinstance(transform, v2.Compose)

        dataset_cls, data_entity_cls, kwargs = fxt_dataset_and_data_entity_cls
        if dataset_cls in [OTXInstanceSegDataset, OTXActionClsDataset]:
            pytest.skip(
                "Instance segmentation, and Action classification task are not suitible for torchvision transform",
            )
        mocker.patch.object(HLabelInfo, "from_dm_label_groups", return_value=fxt_mock_hlabelinfo)
        dataset = dataset_cls(
            dm_subset=fxt_mock_dm_subset,
            transforms=transform,
            mem_cache_img_max_size=None,
            **kwargs,
        )
        dataset.num_classes = 1

        item = dataset[0]
        assert isinstance(item, data_entity_cls)

    @pytest.fixture(params=["RGB", "BGR"])
    def fxt_image_color_channel(self, request) -> ImageColorChannel:
        return ImageColorChannel(request.param)

    def test_image_info(
        self,
        mocker,
        fxt_config,
        fxt_dataset_and_data_entity_cls,
        fxt_mock_dm_subset,
        fxt_image_color_channel,
        fxt_mock_hlabelinfo,
    ) -> None:
        transform = TorchVisionTransformLib.generate(fxt_config)
        assert isinstance(transform, v2.Compose)

        dataset_cls, data_entity_cls, kwargs = fxt_dataset_and_data_entity_cls
        if dataset_cls in [OTXInstanceSegDataset, OTXActionClsDataset]:
            pytest.skip(
                "Instance segmentation, and Action classification task are not suitible for torchvision transform",
            )
        mocker.patch.object(HLabelInfo, "from_dm_label_groups", return_value=fxt_mock_hlabelinfo)
        dataset = dataset_cls(
            dm_subset=fxt_mock_dm_subset,
            transforms=transform,
            mem_cache_img_max_size=None,
            image_color_channel=fxt_image_color_channel,
            **kwargs,
        )
        dataset.num_classes = 1

        item = dataset[0]
        assert item.img_info.img_shape == item.image.shape[1:]

        if fxt_image_color_channel == ImageColorChannel.RGB:
            r_pixel = 255.0 * (0.229 * item.image[0, 0, 0] + 0.485)
            g_pixel = 255.0 * (0.224 * item.image[1, 0, 0] + 0.456)
            b_pixel = 255.0 * (0.225 * item.image[2, 0, 0] + 0.406)
        else:
            b_pixel = 255.0 * (0.229 * item.image[0, 0, 0] + 0.485)
            g_pixel = 255.0 * (0.224 * item.image[1, 0, 0] + 0.456)
            r_pixel = 255.0 * (0.225 * item.image[2, 0, 0] + 0.406)

        assert torch.allclose(r_pixel, torch.tensor(2.0))
        assert torch.allclose(g_pixel, torch.tensor(1.0))
        assert torch.allclose(b_pixel, torch.tensor(0.0))
