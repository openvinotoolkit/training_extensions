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
from otx.core.data.entity.base import Points
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

        inpt = Points(
            [[1, 1], [3, 3]],
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

    @pytest.fixture()
    def fxt_config_w_input_size(self) -> list[dict[str, Any]]:
        cfg = """
        input_size:
        - 300
        - 200
        transforms:
          - class_path: otx.core.data.transform_libs.torchvision.ResizetoLongestEdge
            init_args:
                size: $(input_size) * 2
          - class_path: otx.core.data.transform_libs.torchvision.RandomResize
            init_args:
                scale: $(input_size) * 0.5
          - class_path: otx.core.data.transform_libs.torchvision.RandomCrop
            init_args:
                crop_size: $(input_size)
          - class_path: otx.core.data.transform_libs.torchvision.RandomResize
            init_args:
                scale: $(input_size) * 1.1
        """
        return OmegaConf.create(cfg)

    def test_configure_input_size(self, fxt_config_w_input_size):
        transform = TorchVisionTransformLib.generate(fxt_config_w_input_size)
        assert isinstance(transform, v2.Compose)
        assert transform.transforms[0].size == 600  # ResizetoLongestEdge gets an integer
        assert transform.transforms[1].scale == (150, 100)  # RandomResize gets sequence of integer
        assert transform.transforms[2].crop_size == (300, 200)  # RandomCrop gets sequence of integer
        assert transform.transforms[3].scale == (round(300 * 1.1), round(200 * 1.1))  # check round

    def test_configure_input_size_none(self, fxt_config_w_input_size):
        """Check input size is None but transform has $(ipnput_size)."""
        fxt_config_w_input_size.input_size = None
        with pytest.raises(RuntimeError, match="input_size is set to None"):
            TorchVisionTransformLib.generate(fxt_config_w_input_size)

    def test_eval_input_size_str(self):
        assert TorchVisionTransformLib._eval_input_size_str("2") == 2
        assert TorchVisionTransformLib._eval_input_size_str("(2, 3)") == (2, 3)
        assert TorchVisionTransformLib._eval_input_size_str("2*3") == 6
        assert TorchVisionTransformLib._eval_input_size_str("(2, 3) *3") == (6, 9)
        assert TorchVisionTransformLib._eval_input_size_str("(5, 5) / 2") == (2, 2)
        assert TorchVisionTransformLib._eval_input_size_str("(10, 11) * -0.5") == (-5, -6)

    @pytest.mark.parametrize("input_str", ["1+1", "1+-5", "rm fake", "hoho", "DecordDecode()"])
    def test_eval_input_size_str_wrong_value(self, input_str):
        with pytest.raises(SyntaxError):
            assert TorchVisionTransformLib._eval_input_size_str(input_str)

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
