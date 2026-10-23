<!-- Copyright (C) 2026 Intel Corporation -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inventory of models and their exported IR metadata

This page lists the `rt_info["model_info"]` values that each recipe used by Geti writes into its exported model.
Together these values define the pre- and post-processing that the runtime (ModelAPI) applies.
For the meaning of each key, see [openvino-ir-metadata.md](openvino-ir-metadata.md);
for the outputs and postprocessing implied by each `model_type`, see its [output contract](openvino-ir-metadata.md#output-contract-per-model_type).
When a key is not written, ModelAPI uses the wrapper default listed in that page; the notes below each table describe the effect.

The values below assume the default Geti training flow:

- the recipe is selected via `RecipeResolver.TEMPLATE_ID_MAPPING` (`application/backend/app/execution/common/recipe_resolver.py`);
- tiling is disabled and intensity mapping is left at its default;
- the model is exported with `export_nms=False` and without explain outputs.

## How values are resolved

### `mean_values` / `scale_values`

| Backend      | Resolution order                                                                                                                                                               |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Lightning    | test-subset GPU `Normalize` → CPU pipeline `Normalize` → model `_default_preprocessing_params` → `(0,0,0)/(1,1,1)` (see `GPUAugmentationCallback._update_model_normalization`) |
| Hugging Face | model `_default_preprocessing_params` (recipe `Normalize` is not used)                                                                                                         |
| Ultralytics  | model `_default_preprocessing_params` (recipe `Normalize` is not used)                                                                                                         |

In the tables below, **ImageNet** stands for `mean_values="0.485 0.456 0.406"`, `scale_values="0.229 0.224 0.225"`,
and **identity** stands for `mean_values="0.0 0.0 0.0"`, `scale_values="1.0 1.0 1.0"`.

### Keys written for every model

| Key                                                      | Value                                                                                                                                                                                                                                                                                                                                                                                                                |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `labels`, `label_ids`, `label_info`                      | From the project labels. Lightning and HF instance segmentation models add a leading `getitune_empty_lbl` label with id `None`, because the ModelAPI `MaskRCNN` / `DETRInstSeg` wrappers map class index `i` to `labels[i + 1]`. Lightning semantic segmentation models drop a leading `getitune_background_lbl` label, if present, because the ModelAPI `Segmentation` wrapper maps channel `k` to `labels[k - 1]`. |
| `getitune_version`                                       | Version of the library used for export.                                                                                                                                                                                                                                                                                                                                                                              |
| `input_dtype`                                            | `u8`                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `intensity_mode`                                         | `scale_to_unit`                                                                                                                                                                                                                                                                                                                                                                                                      |
| `intensity_percentile_low` / `intensity_percentile_high` | `1.0` / `99.0`                                                                                                                                                                                                                                                                                                                                                                                                       |
| `intensity_scale_factor` / `intensity_min_value`         | `1.0` / `0.0`                                                                                                                                                                                                                                                                                                                                                                                                        |

`intensity_max_value` is written only if the project sets an intensity mapping.
In the default flow, the effective preprocessing is `u8 → /255 → (x − mean_values) / scale_values`.

### Confidence threshold

| Backend / task                                | `confidence_threshold`                                                                               |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Lightning detection                           | Best F1 threshold from validation; if never computed, the key is not written and ModelAPI uses `0.5` |
| Lightning instance segmentation               | Best F1 threshold from validation, falling back to `0.05`                                            |
| HF detection / instance segmentation          | Best F1 threshold from validation, falling back to `0.25` / `0.05`                                   |
| Ultralytics detection / instance segmentation | `min(best validation threshold, 0.25)` (recipe `export.confidence_threshold`)                        |

### Tiling

If tiling is enabled (Lightning detection, instance segmentation and semantic segmentation only), Geti switches to the `*_tile.yaml` recipe.
That adds `tile_size`, `tiles_overlap`, and `max_pred_number`.
YOLOX also switches to `resize_type=standard`, and the tile recipes may use a different `Normalize` stage.

## Classification

`task_type=classification`, `model_type=Classification`.

| Manifest ID                                                  | Recipe (`multi_class_cls/`)                    | Backend      | `model_name`                                            | Input        | `resize_type` | `pad_value` | `reverse_input_channels` | mean / scale                       | Postprocessing keys                          |
| ------------------------------------------------------------ | ---------------------------------------------- | ------------ | ------------------------------------------------------- | ------------ | ------------- | ----------- | ------------------------ | ---------------------------------- | -------------------------------------------- |
| image-classification-efficientnet-b0                         | `efficientnet_b0.yaml`                         | Lightning    | `efficientnet_b0`                                       | 224          | `standard`    | 0           | False                    | ImageNet                           | `multilabel=False`, `output_raw_scores=True` |
| image-classification-efficientnet-b3                         | `efficientnet_b3.yaml`                         | Lightning    | `efficientnet_b3`                                       | 224          | `standard`    | 0           | False                    | ImageNet                           | same                                         |
| image-classification-efficientnet-v2-s                       | `efficientnet_v2.yaml`                         | Lightning    | `tf_efficientnetv2_s.in21k`                             | 224          | `standard`    | 0           | False                    | ImageNet                           | same                                         |
| image-classification-mobilenet-v3-large                      | `mobilenet_v3_large.yaml`                      | Lightning    | `mobilenetv3_large`                                     | 224          | `standard`    | 0           | False                    | ImageNet                           | same                                         |
| image-classification-vit-tiny                                | `vit_tiny.yaml`                                | Lightning    | `vit-tiny`                                              | 224          | `standard`    | 0           | False                    | ImageNet                           | same                                         |
| image-classification-dinov2                                  | `dino_v2.yaml`                                 | Lightning    | `dinov2-small`                                          | 224          | `standard`    | 0           | False                    | ImageNet                           | same                                         |
| timm catalog IDs                                             | `timm_generic.yaml`                            | Lightning    | timm backbone name                                      | per backbone | `standard`    | 0           | False                    | Backbone `pretrained_cfg` mean/std | same                                         |
| image-classification-convnextv2-{atto,base,large}            | `convnextv2_{atto,base,large}.yaml`            | Hugging Face | `facebook/convnextv2-{atto,base,large}-1k-224`          | 224          | `standard`    | 0           | False                    | ImageNet                           | same                                         |
| image-classification-dinov3-convnext-{tiny,small,base,large} | `dinov3_convnext_{tiny,small,base,large}.yaml` | Hugging Face | `timm/convnext_{tiny,small,base,large}.dinov3_lvd1689m` | 224          | `standard`    | 0           | False                    | ImageNet                           | same                                         |
| image-classification-dinov3-{vits,vitb16,vitl16}             | `dinov3_{vits,vitb16,vitl16}.yaml`             | Hugging Face | `timm/vit_{small,base,large}_patch16_dinov3.lvd1689m`   | 256          | `standard`    | 0           | False                    | ImageNet                           | same                                         |
| image-classification-yolo26-{n,s,m,l,x}                      | `yolo26_{n,s,m,l,x}_cls.yaml`                  | Ultralytics  | `yolo26{n,s,m,l,x}-cls`                                 | 224          | `standard`    | 0           | False                    | identity                           | `multilabel=False`, `nms_execute=False`      |

### Multi-label classification

For `sub_task_type=MULTI_LABEL_CLS`, the same manifest IDs resolve to the matching recipe in `multi_label_cls/`.
The preprocessing keys match the multi-class table.
Only the postprocessing keys change:

| Backend      | Postprocessing keys                                                                          |
| ------------ | -------------------------------------------------------------------------------------------- |
| Lightning    | `multilabel=True`, `output_raw_scores=True`, `confidence_threshold=0.5`                      |
| Hugging Face | `multilabel=True`, `output_raw_scores=True`                                                  |
| Ultralytics  | `multilabel=True`, `output_raw_scores=True`, `confidence_threshold=0.5`, `nms_execute=False` |

All classifiers output logits `[1, C]`; ModelAPI applies softmax (multi-class) or sigmoid (multi-label).
Effect of the keys that are not written:

- Ultralytics multi-class models: `output_raw_scores` defaults to `False`, so ModelAPI does not return the full score vector.
- Hugging Face multi-label models: `confidence_threshold` defaults to `0.5`, the same value the other backends write.

## Detection

`task_type=detection`.

| Manifest ID                                  | Recipe (`detection/`)                   | Backend      | `model_name`                             | `model_type` | Input                 | `resize_type`             | `pad_value` | `reverse_input_channels` | mean / scale                                                                                                                  | Postprocessing keys                     |
| -------------------------------------------- | --------------------------------------- | ------------ | ---------------------------------------- | ------------ | --------------------- | ------------------------- | ----------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| object-detection-atss-mobilenet-v2           | `atss_mobilenetv2.yaml`                 | Lightning    | `atss_mobilenetv2`                       | `ssd`        | 800×992               | `standard`                | 0           | False                    | identity                                                                                                                      | `iou_threshold=0.5`, `nms_execute=True` |
| object-detection-ssd-mobilenet-v2            | `ssd_mobilenetv2.yaml`                  | Lightning    | `ssd_mobilenetv2`                        | `ssd`        | 864                   | `standard`                | 0           | False                    | identity                                                                                                                      | `iou_threshold=0.5`, `nms_execute=True` |
| object-detection-yolox-tiny                  | `yolox_tiny.yaml`                       | Lightning    | `yolox_tiny`                             | `ssd`        | 416                   | `fit_to_window_letterbox` | 114         | False                    | ImageNet                                                                                                                      | `iou_threshold=0.5`, `nms_execute=True` |
| object-detection-yolox-{s,l,x}               | `yolox_{s,l,x}.yaml`                    | Lightning    | `yolox_{s,l,x}`                          | `ssd`        | 640                   | `fit_to_window_letterbox` | 114         | **True**                 | `mean_values="0.0 0.0 0.0"`, `scale_values="0.003921568627451 0.003921568627451 0.003921568627451"` (rescales input to 0–255) | `iou_threshold=0.5`, `nms_execute=True` |
| object-detection-rt-detr-r50                 | `rtdetr_50.yaml`                        | Lightning    | `rtdetr_50`                              | `ssd`        | 640                   | `standard`                | 0           | False                    | identity                                                                                                                      | `iou_threshold=0.8`, `nms_execute=True` |
| object-detection-dfine-{m,l,x}               | `deim_dfine_{m,l,x}.yaml`               | Lightning    | `deim_dfine_hgnetv2_{m,l,x}`             | `ssd`        | 640                   | `standard`                | 0           | False                    | identity                                                                                                                      | `iou_threshold=0.8`, `nms_execute=True` |
| object-detection-dinov3-detr-{s,m,l}         | `deimv2_{s,m,l}.yaml`                   | Lightning    | `deimv2_{s,m,l}`                         | `ssd`        | 640                   | `standard`                | 0           | False                    | ImageNet                                                                                                                      | `iou_threshold=0.8`, `nms_execute=True` |
| object-detection-rfdetr-{n,s,m,l}            | `rfdetr_{nano,small,medium,large}.yaml` | Lightning    | `rfdetr_{nano,small,medium,large}`       | `ssd`        | 384 / 512 / 576 / 704 | `standard`                | 0           | False                    | ImageNet                                                                                                                      | `iou_threshold=0.8`, `nms_execute=True` |
| object-detection-edgecrafter-{s,m,l,x}       | `edgecrafter_{s,m,l,x}.yaml`            | Lightning    | `edgecrafter_{s,m,l,x}`                  | `ssd`        | 640                   | `standard`                | 0           | False                    | ImageNet                                                                                                                      | `iou_threshold=0.5`, `nms_execute=True` |
| object-detection-rtdetrv2-{r18,r34,r50,r101} | `rtdetrv2_{r18,r34,r50,r101}.yaml`      | Hugging Face | `PekingU/rtdetr_v2_{r18,r34,r50,r101}vd` | `ssd`        | 640                   | `standard`                | 0           | False                    | ImageNet                                                                                                                      | `iou_threshold=0.5`                     |
| object-detection-yolo11-{n,s,m,l,x}          | `yolo11_{n,s,m,l,x}.yaml`               | Ultralytics  | `yolo11{n,s,m,l,x}.yaml`                 | `YOLO11`     | 640                   | `fit_to_window_letterbox` | 114         | False                    | identity                                                                                                                      | `iou_threshold=0.5`, `nms_execute=True` |
| object-detection-yolo12-{n,s,m,l,x}          | `yolo12_{n,s,m,l,x}.yaml`               | Ultralytics  | `yolo12{n,s,m,l,x}.yaml`                 | `YOLO11`     | 640                   | `fit_to_window_letterbox` | 114         | False                    | identity                                                                                                                      | `iou_threshold=0.5`, `nms_execute=True` |
| object-detection-yolo26-{n,s,m,l,x}          | `yolo26_{n,s,m,l,x}.yaml`               | Ultralytics  | `yolo26{n,s,m,l,x}.yaml`                 | `YOLO11`     | 640                   | `fit_to_window_letterbox` | 114         | False                    | identity                                                                                                                      | `iou_threshold=0.5`, `nms_execute=True` |

### Detection model outputs

| Models                                                   | Outputs                                                                                                                 | ModelAPI parser              |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| ATSS, SSD, YOLOX                                         | Boxes `[1, N, 5]` as `x1, y1, x2, y2, score` in network-input pixels, labels `[1, N]`; one entry per prior, without NMS | `SSD`, boxes + labels layout |
| RT-DETR, D-FINE, DEIMv2, RF-DETR, EdgeCrafter, RT-DETRv2 | `bboxes` `[1, K, 4]` as `x1, y1, x2, y2` normalized to `[0, 1]`, `labels` `[1, K]`, `scores` `[1, K]`; top-K queries    | `SSD`, three-output layout   |
| YOLO11, YOLO12, YOLO26                                   | A single `[1, 4 + C, N]` tensor: `cx, cy, w, h` in network-input pixels and `C` class scores per anchor, without NMS    | `YOLO11`                     |

Notes:

- **RT-DETRv2 (Hugging Face):** `nms_execute` is not written, so ModelAPI does not run NMS on its outputs.
- **YOLO12 / YOLO26:** they export `model_type=YOLO11` and are decoded with the `YOLO11` wrapper.

## Instance segmentation

`task_type=instance_segmentation`.

| Manifest ID                                     | Recipe (`instance_segmentation/`)                          | Backend      | `model_name`                                                 | `model_type`  | Input                             | `resize_type`             | `pad_value` | `reverse_input_channels` | mean / scale                                                    | Postprocessing keys                                     |
| ----------------------------------------------- | ---------------------------------------------------------- | ------------ | ------------------------------------------------------------ | ------------- | --------------------------------- | ------------------------- | ----------- | ------------------------ | --------------------------------------------------------------- | ------------------------------------------------------- |
| instance-segmentation-mask-rcnn-efficientnet-b2 | `maskrcnn_efficientnetb2b.yaml`                            | Lightning    | `maskrcnn_efficientnet_b2b`                                  | `MaskRCNN`    | 1024                              | `fit_to_window`           | 0           | False                    | `mean_values="0.485 0.456 0.406"`, `scale_values="1.0 1.0 1.0"` | `iou_threshold=0.5`, `nms_execute=False` (NMS in graph) |
| instance-segmentation-mask-rcnn-swin-t          | `maskrcnn_swint.yaml`                                      | Lightning    | `maskrcnn_swin_tiny`                                         | `MaskRCNN`    | 1344                              | `fit_to_window`           | 0           | False                    | ImageNet                                                        | `iou_threshold=0.5`, `nms_execute=False` (NMS in graph) |
| instance-segmentation-mask-rcnn-resnet50        | `maskrcnn_r50.yaml`                                        | Lightning    | `maskrcnn_resnet_50`                                         | `MaskRCNN`    | 1024                              | `fit_to_window`           | 0           | False                    | ImageNet                                                        | `iou_threshold=0.5`, `nms_execute=False` (NMS in graph) |
| instance-segmentation-rtmdet-tiny               | `rtmdet_inst_tiny.yaml`                                    | Lightning    | `rtmdet_inst_tiny`                                           | `MaskRCNN`    | 640                               | `fit_to_window_letterbox` | 114         | False                    | ImageNet                                                        | `iou_threshold=0.5`, `nms_execute=False` (NMS in graph) |
| instance-segmentation-rfdetr-{n,s,m,l,xl,2xl}   | `rfdetr_seg_{nano,small,medium,large,xlarge,2xlarge}.yaml` | Lightning    | `rfdetr_seg_{n,s,m,l,xl,2xl}`                                | `DETRInstSeg` | 312 / 384 / 432 / 504 / 624 / 768 | `standard`                | 0           | False                    | ImageNet                                                        | `iou_threshold=0.8`, `nms_execute=True`                 |
| instance-segmentation-mask2former-swin-{s,b,l}  | `mask2former_swin_{s,b,l}.yaml`                            | Hugging Face | `facebook/mask2former-swin-{small,base,large}-coco-instance` | `DETRInstSeg` | 1024                              | `fit_to_window`           | 0           | False                    | ImageNet                                                        | `iou_threshold=0.5`                                     |
| instance-segmentation-eomt-large-640            | `eomt_large_640.yaml`                                      | Hugging Face | `tue-mps/coco_instance_eomt_large_640`                       | `DETRInstSeg` | 640                               | `fit_to_window`           | 0           | False                    | ImageNet                                                        | `iou_threshold=0.5`                                     |
| instance-segmentation-eomt-dinov3-large-640     | `eomt_dinov3_large_640.yaml`                               | Hugging Face | `tue-mps/eomt-dinov3-coco-instance-large-640`                | `DETRInstSeg` | 640                               | `fit_to_window`           | 0           | False                    | ImageNet                                                        | `iou_threshold=0.5`                                     |
| instance-segmentation-yolo11-{n,s,m,l,x}        | `yolo11_{n,s,m,l,x}_seg.yaml`                              | Ultralytics  | `yolo11{n,s,m,l,x}-seg.yaml`                                 | `YOLO-seg`    | 640                               | `fit_to_window_letterbox` | 114         | False                    | identity                                                        | `iou_threshold=0.5`, `nms_execute=True`                 |
| instance-segmentation-yolo26-{n,s,m,l,x}        | `yolo26_{n,s,m,l,x}_seg.yaml`                              | Ultralytics  | `yolo26{n,s,m,l,x}-seg.yaml`                                 | `YOLO-seg`    | 640                               | `fit_to_window_letterbox` | 114         | False                    | identity                                                        | `iou_threshold=0.5`, `nms_execute=True`                 |

### Instance segmentation model outputs

| Models                                                       | Outputs                                                                                                                                                                                                                 |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mask R-CNN (EfficientNet-B2, Swin-T, ResNet-50), RTMDet-Inst | Boxes `[1, N, 5]` as `x1, y1, x2, y2, score` in network-input pixels, labels `[1, N]`, per-box masks `[1, N, h, w]`; NMS already applied in the graph                                                                   |
| RF-DETR-Seg                                                  | `boxes` `[1, N, 5]` in network-input pixels, `labels` `[1, N]`, full-image `masks`                                                                                                                                      |
| Mask2Former, EoMT                                            | `boxes` `[1, Q, 5]` in network-input pixels (derived from the mask extents; score is the top class probability), `labels` `[1, Q]`, `masks` `[1, Q, H, W]` with per-query probabilities at the network-input resolution |
| YOLO11-seg, YOLO26-seg                                       | Detection tensor `[1, 4 + C + M, N]` and prototype tensor `[1, M, ph, pw]`                                                                                                                                              |

Notes:

- **Label offset:** for `MaskRCNN` and `DETRInstSeg`, output class index `i` maps to `labels[i + 1]` (index 0 is the placeholder).
  For `YOLO-seg`, index `i` maps to `labels[i]`.
- **Mask2Former / EoMT (Hugging Face):** `nms_execute` is not written, so ModelAPI does not run NMS on their outputs.

## Semantic segmentation

`task_type=segmentation`, `model_type=Segmentation`.

Geti has no manifests for semantic segmentation yet, so this table is keyed by getitune recipe only.
The values are the recipe defaults.

| Recipe (`semantic_segmentation/`) | Backend      | `model_name`                                | Input | `resize_type` | `pad_value` | `reverse_input_channels` | mean / scale | Postprocessing keys                                                                                                      | Model output                                                                                     |
| --------------------------------- | ------------ | ------------------------------------------- | ----- | ------------- | ----------- | ------------------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| `litehrnet_{18,s,x}.yaml`         | Lightning    | `lite_hrnet_{18,s,x}`                       | 512   | `standard`    | 0           | False                    | ImageNet     | `return_soft_prediction=True`, `soft_threshold=0.5`, `blur_strength=-1`                                                  | Per-class softmax probabilities                                                                  |
| `segnext_{t,s,b}.yaml`            | Lightning    | `segnext_{tiny,small,base}`                 | 512   | `standard`    | 0           | False                    | ImageNet     | same                                                                                                                     | Per-class softmax probabilities                                                                  |
| `dino_v2.yaml`                    | Lightning    | `dinov2-small-seg`                          | 518   | `standard`    | 0           | False                    | ImageNet     | same                                                                                                                     | Per-class softmax probabilities                                                                  |
| `segformer_b0.yaml`               | Hugging Face | `nvidia/segformer-b0-finetuned-ade-512-512` | 512   | `standard`    | 0           | False                    | ImageNet     | same                                                                                                                     | Per-class softmax probabilities, upsampled in the graph to the input resolution (output `preds`) |
| `yolo26_{n,s,m,l,x}_sem.yaml`     | Ultralytics  | `yolo26{n,s,m,l,x}-sem.yaml`                | 512   | `standard`    | 0           | False                    | identity     | `return_soft_prediction=True`, `blur_strength=-1`, `confidence_threshold=0.0`, `nms_execute=False` (no `soft_threshold`) | Per-class float logits; no softmax and no argmax in the graph                                    |

The tiled Lightning recipes (`*_tile.yaml`) also write `tile_size`, `tiles_overlap` and `max_pred_number`.
The tiled DINOv2 recipe normalizes in the CPU pipeline with the same ImageNet values.

Notes:

- **`soft_threshold` has no effect:** ModelAPI only applies it when `blur_strength != -1`, and every recipe writes `blur_strength=-1`.
  The hard prediction is the per-pixel argmax.
- **YOLO26-sem logits:** the argmax is the same as for probabilities, but the soft prediction returned with `return_soft_prediction=True` contains logits, not probabilities.
- **Labels:** ModelAPI maps channel `k ≥ 1` to `labels[k - 1]` and treats channel 0 as background.
  Lightning models remove the background label to match this; Hugging Face and Ultralytics models export the project labels unchanged.
