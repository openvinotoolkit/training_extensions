<!-- Copyright (C) 2026 Intel Corporation -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# OpenVINO IR / ONNX export metadata (`rt_info["model_info"]`)

All keys are written under the `model_info` section of the OpenVINO IR `rt_info`
(or as `"model_info <key>"` entries in ONNX `metadata_props`). Values are strings.

For the values written by each model that Geti can train, see [model-ir-metadata-inventory.md](model-ir-metadata-inventory.md).

## How the metadata is consumed

The keys are the contract between getitune and [ModelAPI](https://github.com/open-edge-platform/model_api), the runtime used by Geti:

1. **Wrapper selection:** ModelAPI picks the wrapper class whose name matches `model_type`, ignoring case (e.g. `ssd` selects `SSD`).
2. **Parameters:** each wrapper declares a set of parameters and reads them from `rt_info/model_info/<name>`.
   A missing key falls back to the wrapper default (see the "ModelAPI default" columns below).
   Values passed in the `configuration` argument of `Model.create_model()` override `rt_info`.
3. **Preprocessing is not part of the exported graph.** ModelAPI builds it when the model is loaded:
   for static input shapes it embeds the steps into the OpenVINO graph, so the compiled model takes a raw `[1, H, W, C]` u8 image;
   on NPU it runs the same steps in Python.
   getitune does not set `embedded_processing`, so the `.xml` on disk expects an already preprocessed NCHW float tensor.

If you run the IR with plain OpenVINO, implement the [preprocessing](#preprocessing-parameters) and
the [postprocessing for the model type](#output-contract-per-model_type) yourself.

## Sources

| Source                                        | File                                                               | Content                                                 |
| --------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------- |
| `TaskLevelExportParameters.to_metadata()`     | `library/src/getitune/types/export.py`                             | Task info, labels, postprocessing, tiling               |
| `ModelExporter._extend_model_metadata()`      | `library/src/getitune/backend/lightning/exporter/base.py`          | Preprocessing (normalization, resize, intensity)        |
| `HFModelExporter.metadata`                    | `library/src/getitune/backend/huggingface/exporter/hf_exporter.py` | Default intensity keys when no `IntensityConfig` is set |
| `ModelExporter._embed_openvino_ir_metadata()` | `library/src/getitune/backend/lightning/exporter/base.py`          | Writes all keys via `ov_model.set_rt_info()`            |

Where a key comes from both sources, the value from `to_metadata()` is used.

## Preprocessing parameters

ModelAPI applies the steps in this order:

1. Repeat a single channel to 3 channels (`intensity_repeat_channels`).
2. Resize and pad (`resize_type`, `pad_value`), on the raw input pixels. Interpolation is bilinear
   (ModelAPI `interpolation_mode`, not written by getitune, default `LINEAR`).
3. Intensity mapping (`input_dtype`, `intensity_*`), converting to float32 `[0, 1]`.
4. Channel reversal (`reverse_input_channels`).
5. Normalization: `x = (x - mean_values) / scale_values`.
6. Layout HWC → NCHW, batch size 1.

| Key                         | Type / values                                                        | When written                                               | Action at inference                                                                                                                                                                                                                                                                                                                        |
| --------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `input_dtype`               | `u8` \| `u16` \| `i16` \| `f32`                                      | Intensity config present (HF: always, defaults to `u8`)    | Accept and decode the input image with this storage dtype. Do not downcast high-bit-depth data to uint8.                                                                                                                                                                                                                                   |
| `intensity_mode`            | `scale_to_unit` \| `window` \| `percentile` \| `range_scale`         | Intensity config present (HF: always)                      | Choose the intensity mapping that converts raw pixels to float32 `[0, 1]`. See the table below.                                                                                                                                                                                                                                            |
| `intensity_max_value`       | float                                                                | Intensity config with `max_value` set (HF default `255.0`) | Upper bound / divisor (`scale_to_unit`, `range_scale`).                                                                                                                                                                                                                                                                                    |
| `intensity_window_center`   | float                                                                | `window_center` set                                        | Window center for the `window` mode.                                                                                                                                                                                                                                                                                                       |
| `intensity_window_width`    | float                                                                | `window_width` set                                         | Window width for the `window` mode.                                                                                                                                                                                                                                                                                                        |
| `intensity_percentile_low`  | float (default `1.0`)                                                | Intensity config present                                   | Lower quantile for the `percentile` mode.                                                                                                                                                                                                                                                                                                  |
| `intensity_percentile_high` | float (default `99.0`)                                               | Intensity config present                                   | Upper quantile for the `percentile` mode.                                                                                                                                                                                                                                                                                                  |
| `intensity_scale_factor`    | float (default `1.0`)                                                | Intensity config present                                   | Multiply raw pixels by this value before clipping (`range_scale`).                                                                                                                                                                                                                                                                         |
| `intensity_min_value`       | float (default `0.0`)                                                | Intensity config present                                   | Lower clip bound (`range_scale`).                                                                                                                                                                                                                                                                                                          |
| `intensity_repeat_channels` | `"True"`                                                             | `repeat_channels > 0`                                      | Repeat a single-channel image to 3 channels before feeding the model.                                                                                                                                                                                                                                                                      |
| `reverse_input_channels`    | `"True"` / `"False"`                                                 | Always                                                     | If `True`, reverse the channel order of the input image. ModelAPI documents this as BGR→RGB because it assumes OpenCV (BGR) input, but getitune trains on RGB and Geti feeds RGB images: with `False`, pass RGB unchanged; with `True`, the network receives BGR.                                                                          |
| `resize_type`               | `standard` \| `crop` \| `fit_to_window` \| `fit_to_window_letterbox` | Always                                                     | `standard`: resize to W×H, ignoring aspect ratio. `crop`: center-crop to the target aspect ratio, then resize. `fit_to_window`: scale by `min(W/w, H/h)`, then pad at the **bottom and right**. `fit_to_window_letterbox`: same scale, new size `round(w·s) × round(h·s)`, **centered** with padding `(W-nw)//2` left and `(H-nh)//2` top. |
| `pad_value`                 | int (Lightning/HF `0`, Ultralytics `114`)                            | Always                                                     | Fill value for `fit_to_window_letterbox` padding, in raw input pixel units (padding happens before intensity mapping and normalization). ModelAPI's Python path pads `fit_to_window` with `0`.                                                                                                                                             |
| `mean_values`               | space-separated floats                                               | Always (may be empty)                                      | Subtract the per-channel mean: `x = x - mean`.                                                                                                                                                                                                                                                                                             |
| `scale_values`              | space-separated floats                                               | Always (may be empty)                                      | Divide by the per-channel scale (std): `x = x / scale`.                                                                                                                                                                                                                                                                                    |

### Intensity modes

| `intensity_mode` | Action                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| `scale_to_unit`  | `x = clip(x / max_value, 0, 1)` (`max_value` defaults to 255 for u8, 65535 for u16, 32767 for i16)     |
| `window`         | `lo = center - width/2`, `hi = center + width/2`; `x = (clip(x, lo, hi) - lo) / (hi - lo)`             |
| `percentile`     | `lo, hi = percentile(x, low), percentile(x, high)` per image; `x = (clip(x, lo, hi) - lo) / (hi - lo)` |
| `range_scale`    | `x = x * scale_factor`; `x = (clip(x, min_value, max_value) - min_value) / (max_value - min_value)`    |

### Mapping predictions back to the original image

ModelAPI maps coordinates from the network input (W×H) back to the original image (w×h) as follows:

1. `inv_x = w / W`, `inv_y = h / H`.
2. For `fit_to_window` and `fit_to_window_letterbox`, use one factor for both axes: `inv_x = inv_y = max(inv_x, inv_y)`.
3. For `fit_to_window_letterbox` only: `pad_left = (W - round(w / inv_x)) // 2`, `pad_top = (H - round(h / inv_y)) // 2`; otherwise both are 0.
4. `x_orig = (x - pad_left) * inv_x`, `y_orig = (y - pad_top) * inv_y`, then round and clip to the image.

Semantic segmentation class maps are instead resized to the original size with nearest-neighbor interpolation, without undoing the padding.

## Postprocessing parameters

| Key                      | Type / values        | Tasks                                    | Action at inference                                                                                                                                     | ModelAPI default                     |
| ------------------------ | -------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `multilabel`             | `"True"` / `"False"` | Classification                           | `True`: sigmoid on each class, keep classes with score `> confidence_threshold`. `False`: softmax, keep the top-k classes (ModelAPI `topk`, default 1). | `False`                              |
| `output_raw_scores`      | `"True"` / `"False"` | Classification                           | Also return the full score vector for all classes.                                                                                                      | `False`                              |
| `confidence_threshold`   | float                | Classification, Detection, Instance seg. | Drop predictions with a score not strictly greater than this value.                                                                                     | `0.5` (`YOLO11`, `YOLO-seg`: `0.25`) |
| `iou_threshold`          | float                | Detection, Instance seg.                 | IoU threshold for NMS.                                                                                                                                  | `0.5` (`YOLO11`: `0.7`)              |
| `nms_execute`            | `"True"` / `"False"` | Detection, Instance seg.                 | `True`: the model has no built-in NMS, so the runtime must run NMS (`iou_threshold`, `agnostic_nms`, `nms_max_predictions`). `False`: skip NMS.         | `False` (`YOLO11`: `True`)           |
| `agnostic_nms`           | `"True"` / `"False"` | Detection, Instance seg.                 | `True`: run NMS across all classes together. `False`: run NMS per class. getitune never writes this key, so the wrapper default always applies.         | `False` (`YOLO11`: `True`)           |
| `nms_max_predictions`    | int                  | Detection, Instance seg.                 | Keep at most N boxes after NMS.                                                                                                                         | `200` (`YOLO11`: `30000`)            |
| `return_soft_prediction` | `"True"` / `"False"` | Semantic seg.                            | Also return the per-pixel class probability maps.                                                                                                       | `True`                               |
| `soft_threshold`         | float                | Semantic seg.                            | Only used when `blur_strength != -1`: after blurring, probabilities below this value are set to 0 before the argmax.                                    | `-inf`                               |
| `blur_strength`          | int                  | Semantic seg.                            | Box-filter kernel size applied to the probability maps before the argmax. `-1` disables blurring (and `soft_threshold`).                                | `-1`                                 |

In every detection and instance segmentation wrapper, the NMS steps described below only run when `nms_execute=True`.

## Tiling parameters

| Key               | Type        | Action at inference                                                                                                                  | ModelAPI default |
| ----------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------- |
| `tile_size`       | int         | Split the input image into square tiles of this size and run the model on each tile (plus the full image, depending on the runtime). | `400`            |
| `tiles_overlap`   | float (0–1) | Fraction of overlap between neighboring tiles.                                                                                       | `0.5`            |
| `max_pred_number` | int         | Maximum number of predictions kept after merging the tiles (merge with NMS).                                                         | `100`            |

## Model / label info (no pixel processing)

| Key                   | Action at inference                                                                                                                                                                        |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `model_type`          | Choose the ModelAPI wrapper class, matched case-insensitively (e.g. `ssd` → `SSD`). Defines the expected outputs; see [Output contract per `model_type`](#output-contract-per-model_type). |
| `task_type`           | Choose the task-specific result conversion.                                                                                                                                                |
| `model_name`          | Informational.                                                                                                                                                                             |
| `labels`              | Space-separated label names (spaces in names become `_`). Map output class index → name; the offset depends on the model type.                                                             |
| `label_ids`           | Space-separated label IDs, in the same order as `labels`.                                                                                                                                  |
| `label_info`          | Full JSON `LabelInfo` (hierarchy, groups). Use it for hierarchical or multi-group classification decoding.                                                                                 |
| `optimization_config` | JSON NNCF PTQ configuration. Used by the OpenVINO backend when quantizing; not used for inference.                                                                                         |
| `getitune_version`    | Informational / compatibility check.                                                                                                                                                       |

## Output contract per `model_type`

This section describes the outputs that each ModelAPI wrapper used by getitune expects, and the postprocessing it applies.
The optional outputs `saliency_map` and `feature_vector` (explain mode) are passed through unchanged.

### `Classification`

- **Output:** logits `[1, C]`: the first output that is not `saliency_map` or `feature_vector`.
  If `C == len(labels) + 1`, ModelAPI inserts an `other` label at index 0.
- **Multi-class:** softmax (skipped if the values already sum to 1), then top-k.
- **Multi-label:** sigmoid, then keep the classes with score `> confidence_threshold`.
- **Labels:** class index `i` maps to `labels[i]`.

### `ssd` (wrapper `SSD`)

Used by all getitune detection models except the Ultralytics ones. ModelAPI tries three output layouts, in this order:

1. A single output `[1, 1, N, 7]` with rows `[_, label, score, x1, y1, x2, y2]`, normalized to `[0, 1]`.
2. Three outputs whose names contain `bboxes`, `scores` and `labels`: boxes `[1, N, 4]` as `x1, y1, x2, y2` **normalized to `[0, 1]`**, scores `[1, N]`, labels `[1, N]`.
3. An output with last dimension 5: `[1, N, 5]` rows `x1, y1, x2, y2, score` **in network-input pixels**, plus an optional `labels` output `[1, N]` (all labels are 0 if it is missing).

Postprocessing:

1. Run NMS if `nms_execute=True`.
2. Map boxes back to the original image ([see above](#mapping-predictions-back-to-the-original-image)).
3. Keep boxes with area `> 1` and score `> confidence_threshold`.

Class index `i` maps to `labels[i]`.

### `YOLO11` (same decoder as `YOLOv5` / `YOLOv8`)

- **Output:** a single tensor `[1, 4 + C, N]`: per anchor, `cx, cy, w, h` in network-input pixels followed by `C` class scores (no objectness).
  ModelAPI checks that `len(labels) + 4` equals the second dimension.
- **Postprocessing:**
  1. Keep anchors whose max class score is `> confidence_threshold`; the label is the argmax.
  2. Convert boxes to `x1, y1, x2, y2`.
  3. Run NMS **across all classes** (class-agnostic): the wrapper default is `agnostic_nms=True`, and getitune does not write an override.
     Overlapping boxes with different labels are merged, keeping the highest-scoring one. Per-class NMS gives different results.
  4. Map boxes back to the original image.
- **Labels:** class index `i` maps to `labels[i]`.
- The wrapper defaults (`reverse_input_channels=True`, `scale_values=255`) only apply if the keys are missing; getitune always writes them.

### `MaskRCNN` and `DETRInstSeg`

- **Outputs** (matched by number of dimensions; outputs whose names start with `TopK` are ignored):
  boxes `[1, N, 5]` as `x1, y1, x2, y2, score` **in network-input pixels**, labels `[1, N]`, masks `[1, N, h, w]`. The unbatched layout without the leading 1 is also accepted.
- **Labels are shifted by +1:** class index `i` maps to `labels[i + 1]`. This is why getitune inserts the placeholder `getitune_empty_lbl` at index 0.
- **Postprocessing:**
  1. Map boxes back to the original image.
  2. Keep boxes with score `> confidence_threshold`, area `> 1` and a valid label.
  3. Run NMS.
  4. Build full-image masks:
     - `MaskRCNN`: masks are per-box crops. Pad the mask by 1 pixel, enlarge the box by `s / (s - 2)`, resize the mask to the box, threshold at 0.5 and paste it into an empty image.
     - `DETRInstSeg`: masks cover the full network input. Resize them to the original image and threshold at 0.5.

### `YOLO-seg`

- **Outputs:** a detection tensor `[1, 4 + C + M, N]` (`cx, cy, w, h`, `C` class scores, `M` mask coefficients) and a prototype tensor `[1, M, ph, pw]`.
- **Postprocessing:**
  1. Keep anchors whose max class score is `> confidence_threshold`; the label is the argmax.
  2. Convert boxes to `x1, y1, x2, y2` and run NMS.
  3. Masks: `sigmoid(coefficients @ prototypes)`, zeroed outside the box, resized to the network input, cropped to remove the padding, resized to the original image and thresholded at 0.5.
  4. Map boxes back to the original image.
- **Labels:** class index `i` maps to `labels[i]` (no placeholder).

### `Segmentation`

- **Output:** exactly one output besides `feature_vector`: `[1, C, H, W]` per-class probabilities.
  A 3D output, or `C < 2`, is treated as an already computed class map.
- **Postprocessing:**
  1. Transpose to HWC.
  2. If `blur_strength != -1`, blur the maps and set values below `soft_threshold` to 0.
  3. Take the argmax per pixel.
  4. Resize the class map to the original image with nearest-neighbor interpolation.
- **Labels:** channel 0 is background; channel `k ≥ 1` maps to `labels[k - 1]`.

## Defaults per backend

| Backend      | Exporter                   | `resize_type`                    | `pad_value`               | mean / std                | Intensity                                      |
| ------------ | -------------------------- | -------------------------------- | ------------------------- | ------------------------- | ---------------------------------------------- |
| Lightning    | `LightningModelExporter`   | Set per model (often `standard`) | Set per model (often `0`) | Model's `DataInputParams` | Only if `IntensityConfig` is set               |
| Hugging Face | `HFModelExporter`          | `standard`                       | `0`                       | Model's `DataInputParams` | Falls back to `u8` + `scale_to_unit` + `255.0` |
| Ultralytics  | `UltralyticsModelExporter` | `fit_to_window_letterbox`        | `114`                     | `0,0,0` / `1,1,1`         | `scale_to_unit`, `uint8`                       |

Ultralytics also writes `nms_execute = not export_nms` and a separate `metadata.yaml` (for the Ultralytics CLI, not `rt_info`).
It does not write `agnostic_nms`, so runtime NMS for `YOLO11` models is class-agnostic (ModelAPI default `True`).

## Parameters used by exported models

The following parameters subset is used when exporting model. So any client that wants to run inference using OpenVINO on the exported model should be aware of these parameters and their default values.
See the sections above for what each parameter does.

| Stage           | Parameter              | Description                                                                                         |
| --------------- | ---------------------- | --------------------------------------------------------------------------------------------------- |
| Pre-processing  | `resize_type`          | How the input image is resized to the network input size (`standard`, `crop`, `fit_to_window`, `fit_to_window_letterbox`). |
| Pre-processing  | `pad_value`            | Fill value for padded areas, in raw input pixel units.                                              |
| Pre-processing  | `mean_values`          | Per-channel mean subtracted from the input.                                                         |
| Pre-processing  | `scale_values`         | Per-channel scale (std) the input is divided by.                                                    |
| Post-processing | `confidence_threshold` | Predictions with a score not above this value are dropped.                                          |
| Post-processing | `iou_threshold`        | IoU threshold for NMS.                                                                              |
| Post-processing | `nms_execute`          | Whether the runtime must run NMS (`True` when NMS is not in the graph).                             |
| Post-processing | `agnostic_nms`         | Run NMS across all classes (`True`) or per class (`False`).                                         |
| Post-processing | `nms_max_predictions`  | Maximum number of boxes kept after NMS.                                                             |

