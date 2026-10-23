# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Recipe geometry tests for HF instance segmentation models.

The HF instance segmentation wrapper (``HFInstSegModel``) covers MaskFormer-family
models (Mask2Former, EoMT, ...) whose own heads are boxless: the exported model
(``model_type: DETRInstSeg``) derives boxes from mask extents, and ModelAPI's
runtime decode resizes each raw mask from the *full input canvas* straight to the
original image dimensions (``_full_image_mask_postprocess``), without any
letterbox-pad removal.

That decode is only geometrically exact when the training preprocessing does
*not* preserve aspect ratio: the input canvas must be a plain stretch resize of
the original image, so that rescaling the whole canvas back to the original size
inverts the resize exactly. With a letterbox resize, the padded region travels
into the decoded mask and the content is squashed along the axis that was padded
(masks are vertically misplaced and squashed, e.g. a 0.56x vertical factor for a
16:9 image resized into a square canvas).

This is the same convention the upstream Hugging Face checkpoints use
(``EomtImageProcessor`` resizes with ``shortest_edge == longest_edge``, i.e. no
aspect-ratio preservation) and the one the RF-DETR instance segmentation recipes
(``rfdetr_seg_*.yaml``) already follow in Geti (issue #7829: EoMT and Mask2Former
predictions rendered with vertically squashed, misplaced masks).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from getitune.backend.huggingface.tools.utils import load_recipe
from getitune.types.task import TaskType

RECIPE_DIR = Path("src/getitune/recipe/instance_segmentation")

# Recipes of HF-instance-seg (MaskFormer family) models: no box head, the whole
# exported mask grid spans the input canvas, so ModelAPI decodes masks with a
# full-image resize. These recipes must use non-AR-preserving (stretch) resize
# so that decode is exact.
HF_FULL_IMAGE_MASK_RECIPES = (
    "eomt_large_640.yaml",
    "eomt_dinov3_large_640.yaml",
    "mask2former_swin_t.yaml",
    "mask2former_swin_s.yaml",
    "mask2former_swin_b.yaml",
    "mask2former_swin_l.yaml",
)

_RESIZE_CLASS_PATH = "getitune.data.augmentation.transforms.Resize"


def _resize_transforms(augmentations: list[dict]) -> list[dict]:
    return [aug for aug in augmentations if aug.get("class_path") == _RESIZE_CLASS_PATH]


def _recipe(recipe_name: str) -> dict:
    recipe_file = RECIPE_DIR / recipe_name
    assert recipe_file.exists()
    return load_recipe(recipe_file)


def test_hf_instance_seg_task_type() -> None:
    for recipe_name in HF_FULL_IMAGE_MASK_RECIPES:
        recipe = _recipe(recipe_name)
        assert TaskType(recipe["task"]) is TaskType.INSTANCE_SEGMENTATION


def test_hf_full_image_mask_recipes_use_stretch_resize() -> None:
    """All hf full-image-mask recipes must stretch the input canvas, not letterbox it.

    The ModelAPI ``DETRInstSeg`` full-image mask decode resizes the raw mask
    grid to the original image size as a whole; a letterbox preprocessing would
    therefore squashe and misplace the decoded masks (issue #7829). The
    ``resize_mode`` metadata embedded at export time must be the matching
    ``standard`` (no aspect-ratio preservation), so downstream OF eval pipelines
    and the auto-configurator apply the same stretch resize.
    """
    for recipe_name in HF_FULL_IMAGE_MASK_RECIPES:
        recipe = _recipe(recipe_name)
        init_args = recipe["model"]["init_args"]
        assert init_args["resize_mode"] == "standard", f"{recipe_name}: encode metadata must be 'standard'"

        data = recipe["data"]
        for subset_name in ("train_subset", "val_subset", "test_subset"):
            subset = data[subset_name]
            resizes = _resize_transforms(subset.get("augmentations_cpu", []))
            assert resizes, f"{recipe_name}: {subset_name} must contain a Resize transform"
            for resize in resizes:
                assert resize["init_args"]["keep_aspect_ratio"] is False, (
                    f"{recipe_name}: {subset_name} Resize must not preserve aspect ratio"
                )
                if subset_name != "train_subset":
                    assert resize["init_args"].get("resize_targets") is False

        # the model input size and the data pipeline input size must agree
        # (recipes without an explicit init input_size use the wrapper default
        # for the COCO checkpoints: 1024x1024, same as the shared data base)
        assert list(data["input_size"]) == list(init_args.get("input_size", [1024, 1024])), (
            f"{recipe_name}: data input size must match the model input size"
        )


@pytest.mark.parametrize("resize_type", ["letterbox", "stretch"])
def test_full_image_mask_decode_geometry(resize_type: str) -> None:
    """Numeric demonstration of the decode contract (issue #7829 reproduction).

    A 1280x720 image with a ground-truth square instance is resized into a
    640x640 canvas (bottom-right padded letterbox vs plain stretch), the mask
    grid is simulated in that canvas, and decoded the way ModelAPI's
    ``_full_image_mask_postprocess`` does (whole-canvas resize to (im_w, im_h)).
    """
    ori_h, ori_w = 720, 1280
    gt = np.zeros((ori_h, ori_w), dtype=np.uint8)
    gt[200:400, 400:800] = 1

    scale = min(640 / ori_w, 640 / ori_h)
    canvas_h, canvas_w = round(ori_h * scale), round(ori_w * scale)  # 360x640
    canvas = np.zeros((640, 640), dtype=np.uint8)
    if resize_type == "letterbox":
        content = cv2_resize(np.float32(gt), (canvas_w, canvas_h))
        canvas[:canvas_h, :canvas_w] = content  # bottom-right pad (Geti default)
    else:
        canvas = cv2_resize(np.float32(gt), (640, 640))  # stretch

    # ModelAPI decode: whole-canvas resize to original dims, threshold at 0.5
    decoded = cv2_resize(canvas.astype(np.float32), (ori_w, ori_h)) > 0.5

    inter = float(np.logical_and(decoded, gt).sum())
    union = float(np.logical_or(decoded, gt).sum())
    iou = inter / union

    if resize_type == "letterbox":
        assert iou < 0.5, "letterbox preprocessing must NOT be sent to the full-image mask decode"
    else:
        assert iou > 0.99, "stretch preprocessing must decode back exactly"


def cv2_resize(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Histogram-in-the-loop-free linear resize, mirroring ModelAPI's cv2.resize usage."""
    import cv2

    return cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
