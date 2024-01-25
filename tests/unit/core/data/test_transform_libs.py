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
from otx.core.data.transform_libs.torchvision import (
    PadtoSquare,
    PerturbBoundingBoxes,
    ResizetoLongestEdge,
    TorchVisionTransformLib,
)
from otx.core.types.image import ImageColorChannel
from torchvision import tv_tensors
from torchvision.transforms import v2


class TestPerturbBoundingBoxes:
    def test_transform(self) -> None:
        transform = PerturbBoundingBoxes(offset=20)
        inputs = tv_tensors.BoundingBoxes(
            [
                [100, 100, 200, 200],  # normal bbox
                [0, 0, 299, 299],  # can be out of size
                [0, 0, 100, 100],  # can be out of size
                [100, 100, 299, 299],  # can be out of size
            ],
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(300, 300),
            dtype=torch.float32,
        )

        results = transform._transform(inputs, None)

        assert isinstance(results, tv_tensors.BoundingBoxes)
        assert results.shape == inputs.shape
        assert torch.all(results[:, :2] >= 0.0)
        assert torch.all(results[:, 2:] <= 299.0)


class TestPadtoSquare:
    def test_transform(self) -> None:
        transform = PadtoSquare()

        # height > width
        inpt = tv_tensors.Image(torch.ones((1, 5, 3)))
        results = transform(inpt, transform._get_params([inpt]))

        assert isinstance(results, tuple)
        assert results[0].shape == torch.Size((1, 5, 5))
        assert results[0][:, :, -2:].sum() == 0

        # height < width
        inpt = tv_tensors.Image(torch.ones((1, 3, 5)))
        results = transform(inpt, transform._get_params([inpt]))

        assert isinstance(results, tuple)
        assert results[0].shape == torch.Size((1, 5, 5))
        assert results[0][:, -2:, :].sum() == 0

        # skip other formats
        inpt = tv_tensors.BoundingBoxes(
            [[1, 1, 3, 3]],
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(5, 3),
            dtype=torch.float32,
        )
        results = transform(inpt.clone(), transform._get_params([inpt]))

        assert torch.all(results[0] == inpt)


class TestResizetoLongestEdge:
    def test_transform(self) -> None:
        transform = ResizetoLongestEdge(size=10)

        # height > width
        inpt = tv_tensors.Image(torch.ones((1, 5, 3)))
        results = transform(inpt, transform._get_params([inpt]))

        assert isinstance(results, tuple)
        assert results[0].shape == torch.Size((1, 10, 6))
        assert results[1]["target_size"] == (10, 6)

        # height < width
        inpt = tv_tensors.Image(torch.ones((1, 3, 5)))
        results = transform(inpt, transform._get_params([inpt]))

        assert isinstance(results, tuple)
        assert results[0].shape == torch.Size((1, 6, 10))
        assert results[1]["target_size"] == (6, 10)

        # square
        inpt = tv_tensors.Image(torch.ones((1, 5, 5)))
        results = transform(inpt, transform._get_params([inpt]))

        assert isinstance(results, tuple)
        assert results[0].shape == torch.Size((1, 10, 10))
        assert results[1]["target_size"] == (10, 10)


class TestTorchVisionTransformLib:
    @pytest.fixture(params=[True, False], ids=["from_dict", "from_obj"])
    def fxt_config(self, request) -> list[dict[str, Any]]:
        # >>> transforms = v2.Compose([
        # >>>     v2.RandomResizedCrop(size=(224, 224), antialias=True),
        # >>>     v2.RandomHorizontalFlip(p=0.5),
        # >>>     v2.ToDtype(torch.float32, scale=True),
        # >>>     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # >>> ])
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
        if request.param:
            return created

        return SubsetConfig(
            batch_size=1,
            subset_name="dummy",
            transforms=[instantiate_class(args=(), init=transform) for transform in created.transforms],
        )

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

    @pytest.fixture(params=["RGB", "BGR"])
    def fxt_image_color_channel(self, request) -> ImageColorChannel:
        return ImageColorChannel(request.param)

    def test_image_info(
        self,
        fxt_config,
        fxt_dataset_and_data_entity_cls,
        fxt_mock_dm_subset,
        fxt_image_color_channel,
    ) -> None:
        transform = TorchVisionTransformLib.generate(fxt_config)
        assert isinstance(transform, v2.Compose)

        dataset_cls, data_entity_cls = fxt_dataset_and_data_entity_cls
        dataset = dataset_cls(
            dm_subset=fxt_mock_dm_subset,
            transforms=transform,
            mem_cache_img_max_size=None,
            image_color_channel=fxt_image_color_channel,
        )

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
