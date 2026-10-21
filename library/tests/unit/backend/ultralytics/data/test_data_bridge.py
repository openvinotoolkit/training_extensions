# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Ultralytics data bridge."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from torchvision import tv_tensors

from getitune.backend.ultralytics.data import (
    UltralyticsDatasetAdapter,
    detection_collate_fn,
    instance_seg_collate_fn,
    semantic_collate_fn,
)
from getitune.backend.ultralytics.data.geometry import (
    rescale_bboxes_to_tensor_space,
    rescale_points_to_tensor_space,
)
from getitune.data.dataset.base import VisionDataset
from getitune.data.entity.base import ImageInfo
from getitune.data.entity.polygons import empty_polygons
from getitune.data.entity.rasterization import rasterize


def _polygon_sample_adapter(
    polygons: tv_tensors.KeyPoints,
    *,
    img_shape: tuple[int, int] = (16, 20),
    canvas_size: tuple[int, int] = (16, 20),
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    scale_factor: tuple[float, float] | None = None,
    bboxes: torch.Tensor | None = None,
    masks: tv_tensors.Mask | None = None,
) -> UltralyticsDatasetAdapter:
    """Build a segment-mode adapter over a single polygon-carrying sample."""
    h, w = img_shape
    boxes = bboxes if bboxes is not None else torch.tensor([[2.0, 4.0, 10.0, 12.0]], dtype=torch.float32)
    sample = SimpleNamespace(
        image=tv_tensors.Image(torch.rand(3, h, w, dtype=torch.float32)),
        bboxes=tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
            boxes,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=canvas_size,
        ),
        polygons=polygons,
        masks=masks,
        label=torch.zeros(boxes.shape[0], dtype=torch.uint8),
        img_info=ImageInfo(  # pyrefly: ignore[no-matching-overload]
            img_idx=0,
            img_shape=img_shape,
            ori_shape=canvas_size,
            padding=padding,
            scale_factor=scale_factor,
        ),
    )
    dataset = MagicMock(spec=VisionDataset)
    dataset.__len__.return_value = 1
    dataset.__getitem__.return_value = sample
    return UltralyticsDatasetAdapter(dataset, task_kind="segment")


def test_dataset_adapter_keeps_float_images_and_converts_boxes() -> None:
    image = tv_tensors.Image(torch.rand(3, 16, 20, dtype=torch.float32))
    bboxes = tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
        torch.tensor([[2.0, 4.0, 10.0, 12.0]], dtype=torch.float32),
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=(16, 20),
    )
    sample = SimpleNamespace(
        image=image,
        bboxes=bboxes,
        label=torch.tensor([1]),
        img_info=ImageInfo(  # pyrefly: ignore[no-matching-overload]
            img_idx=0,
            img_shape=(12, 18),
            ori_shape=(16, 20),
            padding=(1, 2, 3, 4),
        ),
    )

    dataset = MagicMock(spec=VisionDataset)
    dataset.__len__.return_value = 1
    dataset.__getitem__.return_value = sample

    adapter = UltralyticsDatasetAdapter(dataset)
    result = adapter[0]

    assert result["img"] is image
    assert result["img"].dtype == torch.float32
    np.testing.assert_allclose(result["bboxes"], np.array([[0.3, 0.5, 0.4, 0.5]], dtype=np.float32))
    assert result["cls"].shape == (1, 1)
    assert result["ori_shape"] == (16, 20)
    assert result["resized_shape"] == (12, 18)
    assert result["ratio_pad"] == (((12 - 2 - 4) / 16, (18 - 1 - 3) / 20), (1, 2))


def test_dataset_adapter_rescales_bboxes_when_canvas_differs_from_tensor() -> None:
    """Val/test subsets use resize_targets=False, so bboxes stay in original coords.

    The adapter must detect this via canvas_size != (tensor_h, tensor_w)
    and rescale bboxes to tensor space before normalisation.
    """
    # Tensor is 16x20 but canvas (original image) is 2x larger.
    image = tv_tensors.Image(torch.rand(3, 16, 20, dtype=torch.float32))
    bboxes = tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
        torch.tensor([[4.0, 8.0, 20.0, 24.0]], dtype=torch.float32),
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=(32, 40),  # original image dims — 2x tensor dims
    )
    sample = SimpleNamespace(
        image=image,
        bboxes=bboxes,
        label=torch.tensor([1]),
        img_info=ImageInfo(  # pyrefly: ignore[no-matching-overload]
            img_idx=0,
            img_shape=(16, 20),
            ori_shape=(32, 40),
            padding=(0, 0, 0, 0),
            scale_factor=(0.5, 0.5),
        ),
    )

    dataset = MagicMock(spec=VisionDataset)
    dataset.__len__.return_value = 1
    dataset.__getitem__.return_value = sample

    adapter = UltralyticsDatasetAdapter(dataset)
    result = adapter[0]

    # After rescaling from canvas (40w, 32h) to tensor (20w, 16h):
    #   x1=4*(20/40)=2, y1=8*(16/32)=4, x2=20*(20/40)=10, y2=24*(16/32)=12
    # Then XYWH normalised:
    #   cx=(2+10)/2/20=0.3, cy=(4+12)/2/16=0.5, w=8/20=0.4, h=8/16=0.5
    np.testing.assert_allclose(result["bboxes"], np.array([[0.3, 0.5, 0.4, 0.5]], dtype=np.float32))
    assert result["ori_shape"] == (32, 40)


def test_dataset_adapter_rescales_bboxes_with_letterbox_padding() -> None:
    """When keep_aspect_ratio=True, the image is resized + padded (letterboxed).

    Bboxes in original coords must be scaled by scale_factor and then offset
    by the padding before normalisation.
    """
    # Original image: 1000x500 (h x w), target tensor: 640x640
    # Aspect-preserving resize: scale = min(640/1000, 640/500) = 0.64
    # resized: 640x320, padding: left=160, right=160, top=0, bottom=0
    image = tv_tensors.Image(torch.rand(3, 640, 640, dtype=torch.float32))
    # Bbox at left edge of original image: x1=0, y1=0, x2=100, y2=200
    bboxes = tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
        torch.tensor([[0.0, 0.0, 100.0, 200.0]], dtype=torch.float32),
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=(1000, 500),
    )
    sample = SimpleNamespace(
        image=image,
        bboxes=bboxes,
        label=torch.tensor([0]),
        img_info=ImageInfo(  # pyrefly: ignore[no-matching-overload]
            img_idx=0,
            img_shape=(640, 320),
            ori_shape=(1000, 500),
            padding=(160, 0, 160, 0),
            scale_factor=(0.64, 0.64),
        ),
    )

    dataset = MagicMock(spec=VisionDataset)
    dataset.__len__.return_value = 1
    dataset.__getitem__.return_value = sample

    adapter = UltralyticsDatasetAdapter(dataset)
    result = adapter[0]

    # After scale + offset from original to tensor:
    #   x1 = 0 * 0.64 + 160 = 160, y1 = 0 * 0.64 + 0 = 0
    #   x2 = 100 * 0.64 + 160 = 224, y2 = 200 * 0.64 + 0 = 128
    # XYWH normalised by tensor 640x640:
    #   cx = (160+224)/2/640 = 0.3, cy = (0+128)/2/640 = 0.1
    #   w  = (224-160)/640 = 0.1,   h  = (128-0)/640 = 0.2
    np.testing.assert_allclose(result["bboxes"], np.array([[0.3, 0.1, 0.1, 0.2]], dtype=np.float32), atol=1e-6)


def test_detection_collate_fn_matches_expected_detection_contract() -> None:
    batch = [
        {
            "img": torch.rand(3, 16, 16),
            "cls": np.array([[0.0], [1.0]], dtype=np.float32),
            "bboxes": np.array([[0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.1, 0.1]], dtype=np.float32),
            "ori_shape": (16, 16),
            "resized_shape": (16, 16),
            "ratio_pad": ((1.0, 1.0), (0, 0)),
            "im_file": "a.jpg",
        },
        {
            "img": torch.rand(3, 16, 16),
            "cls": np.array([[2.0]], dtype=np.float32),
            "bboxes": np.array([[0.4, 0.4, 0.3, 0.3]], dtype=np.float32),
            "ori_shape": (16, 16),
            "resized_shape": (16, 16),
            "ratio_pad": ((1.0, 1.0), (0, 0)),
            "im_file": "b.jpg",
        },
    ]

    collated = detection_collate_fn(batch)

    assert collated["img"].shape == (2, 3, 16, 16)
    assert collated["cls"].shape == (3, 1)
    assert collated["bboxes"].shape == (3, 4)
    assert torch.equal(collated["batch_idx"], torch.tensor([0.0, 0.0, 1.0]))
    assert collated["im_file"] == ["a.jpg", "b.jpg"]


def test_instance_seg_collate_fn_builds_overlap_maps_and_reorders_targets() -> None:
    # One image with two masks: a small one (class 1) and a large one (class 2).
    # Area-descending sort places the large mask first, so the small mask
    # (painted last) must win the overlap and the cls/bboxes order must flip.
    small_mask = torch.zeros((16, 16), dtype=torch.uint8)
    small_mask[0:2, 0:2] = 1  # area 4
    large_mask = torch.zeros((16, 16), dtype=torch.uint8)
    large_mask[0:8, 0:8] = 1  # area 64, overlaps small_mask entirely

    batch = [
        {
            "img": torch.rand(3, 16, 16),
            "cls": np.array([[1.0], [2.0]], dtype=np.float32),
            "bboxes": np.array([[0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.3, 0.3]], dtype=np.float32),
            "masks": torch.stack([small_mask, large_mask]),
            "ori_shape": (16, 16),
            "resized_shape": (16, 16),
            "ratio_pad": ((1.0, 1.0), (0, 0)),
            "im_file": "a.jpg",
        },
    ]

    collated = instance_seg_collate_fn(batch)

    assert collated["img"].shape == (1, 3, 16, 16)
    assert collated["masks"].shape == (1, 16, 16)
    assert collated["sem_masks"].shape == (1, 16, 16)
    # Large mask painted first (index 1), small mask painted last (index 2)
    # and wins the overlap region.
    assert collated["masks"][0, 0, 0] == 2
    assert collated["masks"][0, 6, 6] == 1
    # sem_masks carries the class id of the owning instance at each pixel.
    assert collated["sem_masks"][0, 0, 0] == 1.0
    assert collated["sem_masks"][0, 6, 6] == 2.0
    # cls/bboxes are reordered to match the area-descending mask order
    # (large/class-2 first, small/class-1 second).
    assert torch.equal(collated["cls"], torch.tensor([[2.0], [1.0]]))
    np.testing.assert_allclose(
        collated["bboxes"].numpy(), np.array([[0.3, 0.3, 0.3, 0.3], [0.1, 0.1, 0.1, 0.1]], dtype=np.float32)
    )
    assert torch.equal(collated["batch_idx"], torch.tensor([0.0, 0.0]))


def test_dataset_adapter_segment_mode_includes_masks_and_sem_masks() -> None:
    image = tv_tensors.Image(torch.rand(3, 16, 20, dtype=torch.float32))
    bboxes = tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
        torch.tensor([[2.0, 4.0, 10.0, 12.0]], dtype=torch.float32),
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=(16, 20),
    )
    # Two non-overlapping instance masks for class 1 and class 2.
    mask_0 = torch.zeros((16, 20), dtype=torch.uint8)
    mask_0[4:8, 2:6] = 1
    mask_1 = torch.zeros((16, 20), dtype=torch.uint8)
    mask_1[10:14, 12:16] = 1
    masks = tv_tensors.Mask(torch.stack([mask_0, mask_1]))

    sample = SimpleNamespace(
        image=image,
        bboxes=bboxes,
        label=torch.tensor([1, 2]),
        masks=masks,
        img_info=ImageInfo(  # pyrefly: ignore[no-matching-overload]
            img_idx=0,
            img_shape=(16, 20),
            ori_shape=(16, 20),
            padding=(0, 0, 0, 0),
        ),
    )

    dataset = MagicMock(spec=VisionDataset)
    dataset.__len__.return_value = 1
    dataset.__getitem__.return_value = sample

    adapter = UltralyticsDatasetAdapter(dataset, task_kind="segment")
    result = adapter[0]

    assert result["img"] is image
    assert "masks" in result
    assert "sem_masks" in result
    assert result["masks"].shape == (2, 16, 20)
    assert result["sem_masks"].shape == (16, 20)
    # Overlap index map: smaller-area instance should overwrite larger.
    # Both masks have equal area here, so sort is stable by original order.
    assert result["masks"][0, 4, 2] == 1
    assert result["masks"][1, 10, 12] == 1
    # sem_masks pixels should carry the class id of the owning instance.
    assert result["sem_masks"][4, 2] == 1.0
    assert result["sem_masks"][10, 12] == 2.0


def test_dataset_adapter_semantic_mode_extracts_mask() -> None:
    image = tv_tensors.Image(torch.rand(3, 16, 20, dtype=torch.float32))
    mask = torch.zeros((1, 16, 20), dtype=torch.uint8)
    mask[0, 4:8, 2:6] = 1
    mask[0, 10:14, 12:16] = 2
    masks = tv_tensors.Mask(mask)

    sample = SimpleNamespace(
        image=image,
        masks=masks,
        img_info=ImageInfo(  # pyrefly: ignore[no-matching-overload]
            img_idx=0,
            img_shape=(16, 20),
            ori_shape=(16, 20),
            padding=(0, 0, 0, 0),
        ),
    )

    dataset = MagicMock(spec=VisionDataset)
    dataset.__len__.return_value = 1
    dataset.__getitem__.return_value = sample

    adapter = UltralyticsDatasetAdapter(dataset, task_kind="semantic")
    result = adapter[0]

    assert result["img"] is image
    assert "semantic_mask" in result
    assert result["semantic_mask"].shape == (16, 20)
    assert result["semantic_mask"].dtype == torch.int32
    assert result["semantic_mask"][4, 2] == 1
    assert result["semantic_mask"][10, 12] == 2


def test_semantic_collate_fn_stacks_masks() -> None:
    batch = [
        {
            "img": torch.rand(3, 16, 16),
            "semantic_mask": torch.randint(0, 3, (16, 16), dtype=torch.int32),
            "ori_shape": (16, 16),
            "resized_shape": (16, 16),
            "ratio_pad": ((1.0, 1.0), (0, 0)),
            "im_file": "a.jpg",
        },
        {
            "img": torch.rand(3, 16, 16),
            "semantic_mask": torch.randint(0, 3, (16, 16), dtype=torch.int32),
            "ori_shape": (16, 16),
            "resized_shape": (16, 16),
            "ratio_pad": ((1.0, 1.0), (0, 0)),
            "im_file": "b.jpg",
        },
    ]

    collated = semantic_collate_fn(batch)

    assert collated["img"].shape == (2, 3, 16, 16)
    assert collated["semantic_mask"].shape == (2, 16, 16)
    assert collated["semantic_mask"].dtype == torch.int32
    assert collated["im_file"] == ["a.jpg", "b.jpg"]


def test_dataset_adapter_segment_emits_normalised_segments_without_masks() -> None:
    square = torch.tensor([[2.0, 4.0], [10.0, 4.0], [10.0, 12.0], [2.0, 12.0]], dtype=torch.float32)
    polygons = tv_tensors.KeyPoints(square.unsqueeze(0), canvas_size=(16, 20))

    result = _polygon_sample_adapter(polygons)[0]

    assert "segments" in result
    # The point of the phase: no dense carrier is built on the polygon path.
    assert "masks" not in result
    assert "sem_masks" not in result

    segments = result["segments"]
    assert segments.shape == (1, 4, 2)
    assert segments.dtype == torch.float32
    # Normalised by tensor dims (w=20, h=16), not by the bbox canvas.
    expected = torch.stack([square[:, 0] / 20.0, square[:, 1] / 16.0], dim=-1)
    torch.testing.assert_close(segments[0], expected)


def test_dataset_adapter_segment_polygons_agree_with_bboxes_under_letterbox() -> None:
    """Polygon extent must reproduce the independently-computed bbox.

    ``_extract_bboxes`` and ``_extract_segments`` apply the letterbox affine
    through separate code paths; a sign flip or an (h, w) transposition in
    either one shows up here as a mismatch.
    """
    ring = torch.tensor([[0.0, 0.0], [100.0, 0.0], [100.0, 200.0], [0.0, 200.0]], dtype=torch.float32)
    polygons = tv_tensors.KeyPoints(ring.unsqueeze(0), canvas_size=(1000, 500))

    result = _polygon_sample_adapter(
        polygons,
        img_shape=(640, 640),
        canvas_size=(1000, 500),
        padding=(160, 0, 160, 0),
        scale_factor=(0.64, 0.64),
        bboxes=torch.tensor([[0.0, 0.0, 100.0, 200.0]], dtype=torch.float32),
    )[0]

    seg = result["segments"][0]
    x_min, x_max = float(seg[:, 0].min()), float(seg[:, 0].max())
    y_min, y_max = float(seg[:, 1].min()), float(seg[:, 1].max())
    from_polygons = np.array(
        [[(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(result["bboxes"], from_polygons, atol=1e-6)


def test_dataset_adapter_segment_preserves_nan_padding_and_ring_separators() -> None:
    nan = float("nan")
    inst = torch.tensor(
        [
            [2.0, 4.0],
            [10.0, 4.0],
            [10.0, 12.0],
            [2.0, 12.0],
            [nan, nan],  # ring separator
            [4.0, 6.0],
            [8.0, 6.0],
            [8.0, 10.0],
            [4.0, 10.0],
            [nan, nan],  # trailing padding
        ],
        dtype=torch.float32,
    )
    polygons = tv_tensors.KeyPoints(inst.unsqueeze(0), canvas_size=(16, 20))

    seg = _polygon_sample_adapter(polygons)[0]["segments"][0]

    finite = torch.isfinite(inst).all(dim=-1)
    torch.testing.assert_close(torch.isfinite(seg).all(dim=-1), finite)
    assert torch.isfinite(seg[finite]).all()


def test_dataset_adapter_segment_falls_back_to_masks_for_empty_polygons() -> None:
    result = _polygon_sample_adapter(
        empty_polygons((16, 20)),
        bboxes=torch.zeros((0, 4), dtype=torch.float32),
    )[0]

    assert "segments" not in result
    assert result["masks"].shape == (0, 16, 20)
    assert result["sem_masks"].shape == (16, 20)


def test_rescale_points_matches_rescale_bboxes_on_corner_points() -> None:
    boxes = np.array([[0.0, 0.0, 100.0, 200.0], [50.0, 25.0, 60.0, 80.0]], dtype=np.float32)
    kwargs = {
        "tensor_h": 640,
        "tensor_w": 640,
        "canvas_size": (1000, 500),
        "scale_factor": (0.64, 0.64),
        "padding": (160, 0, 160, 0),
    }
    scaled_boxes = rescale_bboxes_to_tensor_space(boxes.copy(), **kwargs)
    corners = torch.as_tensor(boxes).reshape(-1, 2, 2)  # [(x1,y1), (x2,y2)] per box
    scaled_pts = rescale_points_to_tensor_space(corners, **kwargs).reshape(-1, 4)

    np.testing.assert_allclose(scaled_pts.numpy(), scaled_boxes, atol=1e-6)


def test_rescale_points_handles_empty_and_nan() -> None:
    empty = torch.zeros((0, 0, 2), dtype=torch.float32)
    assert rescale_points_to_tensor_space(empty, 16, 20, (16, 20)).shape == (0, 0, 2)

    nan_pts = torch.tensor([[[float("nan"), float("nan")], [1.0, 2.0]]], dtype=torch.float32)
    out = rescale_points_to_tensor_space(nan_pts, 32, 40, (16, 20))
    assert torch.isnan(out[0, 0]).all()
    torch.testing.assert_close(out[0, 1], torch.tensor([2.0, 4.0]))


def test_instance_seg_collate_fn_polygon_branch_matches_mask_branch() -> None:
    """The two carriers must produce the same overlap map.

    Polygons rasterized at proto resolution must agree with the
    dense path rasterized upstream, or the loss silently trains
    on different targets.
    """
    # 32x32 image, mask_ratio=1 so both branches share a resolution.
    small = torch.tensor([[[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]]])
    large = torch.tensor([[[0.0, 0.0], [24.0, 0.0], [24.0, 24.0], [0.0, 24.0]]])
    coords = torch.cat([small, large], dim=0)  # (2, 4, 2)

    common = {
        "img": torch.rand(3, 32, 32),
        "cls": np.array([[1.0], [2.0]], dtype=np.float32),
        "bboxes": np.array([[0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.3, 0.3]], dtype=np.float32),
        "ori_shape": (32, 32),
        "resized_shape": (32, 32),
        "ratio_pad": ((1.0, 1.0), (0, 0)),
        "im_file": "a.jpg",
    }
    poly = instance_seg_collate_fn([{**common, "segments": coords / 32.0}], mask_ratio=1)
    dense = instance_seg_collate_fn([{**common, "masks": rasterize(coords, canvas_size=(32, 32))}])

    inter = ((poly["masks"] > 0) & (dense["masks"] > 0)).sum()
    union = ((poly["masks"] > 0) | (dense["masks"] > 0)).sum()
    assert inter / union >= 0.98
    assert torch.equal(poly["cls"], dense["cls"])  # same area sort
    assert torch.equal(poly["masks"], dense["masks"])  # same instance indices


def test_instance_seg_collate_fn_rasterizes_at_proto_resolution() -> None:
    coords = torch.tensor([[[0.0, 0.0], [32.0, 0.0], [32.0, 32.0], [0.0, 32.0]]]) / 64.0
    batch = [
        {
            "img": torch.rand(3, 64, 64),
            "cls": np.array([[1.0]], dtype=np.float32),
            "bboxes": np.array([[0.25, 0.25, 0.5, 0.5]], dtype=np.float32),
            "segments": coords,
            "ori_shape": (64, 64),
            "resized_shape": (64, 64),
            "ratio_pad": ((1.0, 1.0), (0, 0)),
            "im_file": "a.jpg",
        }
    ]

    collated = instance_seg_collate_fn(batch)

    assert collated["masks"].shape == (1, 16, 16)  # 64 / 4
    assert collated["sem_masks"].shape == (1, 16, 16)
    assert collated["masks"][0, 0, 0] == 1
    assert collated["masks"][0, 15, 15] == 0


def test_instance_seg_collate_fn_mixes_polygon_and_empty_samples() -> None:
    """An image with no instances falls back to the mask carrier; shapes must still stack."""
    coords = torch.tensor([[[0.0, 0.0], [16.0, 0.0], [16.0, 16.0], [0.0, 16.0]]]) / 32.0
    geo = {"ori_shape": (32, 32), "resized_shape": (32, 32), "ratio_pad": ((1.0, 1.0), (0, 0)), "im_file": ""}
    batch = [
        {
            "img": torch.rand(3, 32, 32),
            "cls": np.array([[1.0]], dtype=np.float32),
            "bboxes": np.array([[0.25, 0.25, 0.5, 0.5]], dtype=np.float32),
            "segments": coords,
            **geo,
        },
        {
            "img": torch.rand(3, 32, 32),
            "cls": np.zeros((0, 1), dtype=np.float32),
            "bboxes": np.zeros((0, 4), dtype=np.float32),
            "masks": torch.zeros((0, 32, 32), dtype=torch.uint8),
            "sem_masks": torch.zeros((32, 32), dtype=torch.float32),
            **geo,
        },
    ]

    collated = instance_seg_collate_fn(batch)

    assert collated["masks"].shape == (2, 8, 8)
    assert collated["masks"][1].sum() == 0
    assert torch.equal(collated["batch_idx"], torch.tensor([0.0]))
