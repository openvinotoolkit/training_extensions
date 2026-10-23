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

RECIPE_DIR = Path(__file__).resolve().parents[3] / "src" / "getitune" / "recipe" / "instance_segmentation"

_RESIZE_CLASS_PATH = "getitune.data.augmentation.transforms.Resize"

# One recipe per distinct recipe shape (issue #7829):
# - eomt_large_640: pins the data pipeline to the checkpoint's 640x640 via overrides;
# - mask2former_swin_t: inherits the shared base 1024x1024 with no explicit input_size.
HF_FULL_IMAGE_MASK_RECIPES = (
    "eomt_large_640.yaml",
    "mask2former_swin_t.yaml",
)


@pytest.mark.parametrize("recipe_name", HF_FULL_IMAGE_MASK_RECIPES)
def test_full_image_mask_recipe_consistency(recipe_name: str) -> None:
    """Recipes driving full-image mask decodes must reference the stretch data base.

    The ModelAPI ``DETRInstSeg`` full-image mask decode resizes the raw mask
    grid to the original image size as a whole; letterbox preprocessing would
    squash and misplace the decoded masks (issue #7829). The ``resize_mode``
    metadata embedded at export time must be the matching ``standard`` so that
    downstream OV eval pipelines and the auto-configurator apply the same
    stretch resize.
    """
    recipe = load_recipe(RECIPE_DIR / recipe_name)
    init_args = recipe["model"]["init_args"]

    data = recipe["data"]
    assert "input_size" in data

    for subset_name in ("train_subset", "val_subset", "test_subset"):
        subset = data[subset_name]
        resizes = [aug for aug in subset["augmentations_cpu"] if aug.get("class_path") == _RESIZE_CLASS_PATH]
        assert resizes, f"{recipe_name}: {subset_name} must contain a Resize transform"
        for resize in resizes:
            assert resize["init_args"]["keep_aspect_ratio"] is False, (
                f"{recipe_name}: {subset_name} Resize must not preserve aspect ratio"
            )
            if subset_name != "train_subset":
                assert resize["init_args"].get("resize_targets") is False

        # the GPU-side augs of the shared base (flip + normalize) must survive
        gpu_paths = [aug.get("class_path", "") for aug in subset.get("augmentations_gpu", [])]
        assert any("Normalize" in cp for cp in gpu_paths), f"{recipe_name}: Normalize must remain in {subset_name}"

    assert init_args["resize_mode"] == "standard", f"{recipe_name}: export metadata must be 'standard'"

    # the model input size and the data pipeline input size must agree
    # (recipes without an explicit init input_size use the wrapper default
    # for the COCO checkpoints: 1024x1024, same as the shared data base)
    assert list(data["input_size"]) == list(init_args.get("input_size", [1024, 1024])), (
        f"{recipe_name}: data input size must match the model input size"
    )


def test_letterbox_is_rejected_by_real_modelapi_decode() -> None:
    """The real ModelAPI decode (imported, not re-implemented) aligns masks only under stretch.

    Reproduces GH #7829 with ModelAPI's own ``_full_image_mask_postprocess``:
    the decode resizes the whole raw mask canvas to the original size, so a
    letterboxed canvas decodes squashed/offset while a stretch canvas decodes
    back exactly.
    """
    from model_api.models.instance_segmentation import _full_image_mask_postprocess

    ori_h, ori_w = 720, 1280
    gt = np.zeros((ori_h, ori_w), dtype=np.uint8)
    gt[200:400, 400:800] = 1

    letterbox = np.zeros((640, 640), dtype=np.float32)
    letterbox[:360, :640] = _resize_mask(gt.astype(np.float32), (640, 360))  # content + bottom-right pad
    stretch = _resize_mask(gt.astype(np.float32), (640, 640))

    def iou(decoded: np.ndarray) -> float:
        return float(np.logical_and(decoded, gt).sum()) / float(np.logical_or(decoded, gt).sum())

    assert iou(_full_image_mask_postprocess(letterbox, ori_h, ori_w) > 0.5) < 0.5, (
        "letterbox must NOT be decoded by ModelAPI's full-image mask decode"
    )
    assert iou(_full_image_mask_postprocess(stretch, ori_h, ori_w) > 0.5) > 0.99, "stretch must decode back exactly"


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Linear resize through cv2, the same preprocessing the decode inverts."""
    import cv2

    return cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
