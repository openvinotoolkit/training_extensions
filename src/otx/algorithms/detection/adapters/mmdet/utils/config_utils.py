"""Collection of utils for task implementation in Detection Task."""
# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from mmcv import Config, ConfigDict
from mmcv.utils import ext_loader
from torchvision.ops import nms as tv_nms
from torchvision.ops import roi_align as tv_roi_align

from otx.algorithms.common.adapters.mmcv.utils import (
    InputSizeManager,
    get_configs_by_pairs,
)
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.algorithms.detection.utils.data import (
    adaptive_tile_params,
    format_list_to_str,
    get_anchor_boxes,
    get_sizes_from_dataset_entity,
)
from otx.api.entities.datasets import DatasetEntity, DatasetPurpose
from otx.api.entities.subset import Subset
from otx.utils.logger import get_logger

try:
    from sklearn.cluster import KMeans

    __all__ = ["KMeans"]

    KMEANS_IMPORT = True
except ImportError:
    KMEANS_IMPORT = False


logger = get_logger()
ext_module = ext_loader.load_ext("_ext", ["nms", "softnms", "nms_match", "nms_rotated", "nms_quadri"])


def should_cluster_anchors(model_cfg: Config):
    """Check whether cluster anchors or not."""
    if (
        hasattr(model_cfg.model, "bbox_head")
        and hasattr(model_cfg.model.bbox_head, "anchor_generator")
        and getattr(
            model_cfg.model.bbox_head.anchor_generator,
            "reclustering_anchors",
            False,
        )
    ):
        return True
    return False


def cluster_anchors(recipe_config: Config, dataset: DatasetEntity):
    """Update configs for cluster_anchors."""
    if not KMEANS_IMPORT:
        raise ImportError(
            "Sklearn package is not installed. To enable anchor boxes clustering, please install "
            "packages from requirements/optional.txt or just scikit-learn package."
        )

    logger.info("Collecting statistics from training dataset to cluster anchor boxes...")
    [target_wh] = [
        transforms.img_scale
        for transforms in recipe_config.data.test.pipeline
        if transforms.type == "MultiScaleFlipAug"
    ]
    prev_generator = recipe_config.model.bbox_head.anchor_generator
    group_as = [len(width) for width in prev_generator.widths]
    wh_stats = get_sizes_from_dataset_entity(dataset, list(target_wh))

    if len(wh_stats) < sum(group_as):
        logger.warning(
            f"There are not enough objects to cluster: {len(wh_stats)} were detected, while it should be "
            f"at least {sum(group_as)}. Anchor box clustering was skipped."
        )
        return

    widths, heights = get_anchor_boxes(wh_stats, group_as)
    logger.info(
        f"Anchor boxes widths have been updated from {format_list_to_str(prev_generator.widths)} "
        f"to {format_list_to_str(widths)}"
    )
    logger.info(
        f"Anchor boxes heights have been updated from {format_list_to_str(prev_generator.heights)} "
        f"to {format_list_to_str(heights)}"
    )
    config_generator = recipe_config.model.bbox_head.anchor_generator
    config_generator.widths, config_generator.heights = widths, heights

    recipe_config.model.bbox_head.anchor_generator = config_generator


def patch_tiling(config, hparams, dataset=None):
    """Update config for tiling.

    Args:
        config (dict): OTX config containing configuration settings.
        hparams (DetectionConfig): DetectionConfig containing hyperparameters.
        dataset (DatasetEntity, optional): A dataset entity. Defaults to None.

    Returns:
        dict: The updated configuration dictionary.
    """
    if hparams.tiling_parameters.enable_tiling:
        logger.info("Tiling enabled")

        if dataset and dataset.purpose != DatasetPurpose.INFERENCE and hparams.tiling_parameters.enable_adaptive_params:
            adaptive_tile_params(hparams.tiling_parameters, dataset.get_subset(Subset.TRAINING))

        if hparams.tiling_parameters.enable_tile_classifier:
            logger.info("Tile classifier enabled")
            logger.info(f"Patch model from: {config.model.type} to CustomMaskRCNNTileOptimized")
            config.model.type = "CustomMaskRCNNTileOptimized"

            for subset in ("val", "test"):
                if "transforms" in config.data[subset].pipeline[0]:
                    transforms = config.data[subset].pipeline[0]["transforms"]
                    if transforms[-1]["type"] == "Collect":
                        transforms[-1]["keys"].append("full_res_image")

            if config.model.backbone.type == "efficientnet_b2b":
                learning_rate = 0.002
                logger.info(
                    f"Patched {config.model.backbone.type} LR: "
                    f"{hparams.learning_parameters.learning_rate} -> {learning_rate}"
                )
                hparams.learning_parameters.learning_rate = learning_rate

            config.data.train.filter_empty_gt = False

        config.data.train.sampling_ratio = hparams.tiling_parameters.tile_sampling_ratio
        config.data.val.sampling_ratio = hparams.tiling_parameters.tile_sampling_ratio
        if hparams.tiling_parameters.tile_sampling_ratio < 1.0:
            config.custom_hooks.append(ConfigDict({"type": "TileSamplingHook"}))

        tiling_params = ConfigDict(
            tile_size=int(hparams.tiling_parameters.tile_size),
            overlap_ratio=float(hparams.tiling_parameters.tile_overlap),
            max_per_img=int(hparams.tiling_parameters.tile_max_number),
        )
        config.update(
            ConfigDict(
                data=ConfigDict(
                    train=tiling_params,
                    val=tiling_params,
                    test=tiling_params,
                )
            )
        )
        config.update(dict(evaluation=dict(iou_thr=[0.5])))

    return config


def patch_input_preprocessing(cfg: ConfigDict, deploy_cfg: ConfigDict):
    """Update backend configuration with input preprocessing options.

    - If `"to_rgb"` in Normalize config is truthy, it adds `"--reverse_input_channels"` as a flag.

    The function then sets default values for the backend configuration in `deploy_cfg`.

    Args:
        cfg (mmcv.ConfigDict): Config object containing test pipeline and other configurations.
        deploy_cfg (mmcv.ConfigDict): DeployConfig object containing backend configuration.

    Returns:
        None: This function updates the input `deploy_cfg` object directly.
    """
    normalize_cfgs = get_configs_by_pairs(cfg.data.test.pipeline, dict(type="Normalize"))
    assert len(normalize_cfgs) == 1
    normalize_cfg: dict = normalize_cfgs[0]

    # Set options based on Normalize config
    options = {
        "flags": ["--reverse_input_channels"] if normalize_cfg.get("to_rgb", False) else [],
        "args": {
            "--mean_values": list(normalize_cfg.get("mean", [])),
            "--scale_values": list(normalize_cfg.get("std", [])),
        },
    }

    # Set default backend configuration
    mo_options = deploy_cfg.backend_config.get("mo_options", ConfigDict())
    mo_options = ConfigDict() if mo_options is None else mo_options
    mo_options.args = mo_options.get("args", ConfigDict())
    mo_options.flags = mo_options.get("flags", [])

    # Override backend configuration with options from Normalize config
    mo_options.args.update(options["args"])
    mo_options.flags = list(set(mo_options.flags + options["flags"]))

    deploy_cfg.backend_config.mo_options = mo_options


def patch_input_shape(cfg: ConfigDict, deploy_cfg: ConfigDict):
    """Update backend configuration with input shape information.

    This function retrieves the input size from `cfg.data.test.pipeline`,
    then sets the input shape for the backend model in `deploy_cfg`

    ```
    {
        "opt_shapes": {
            "input": [1, 3, *size]
        }
    }
    ```

    Args:
        cfg (Config): Config object containing test pipeline and other configurations.
        deploy_cfg (DeployConfig): DeployConfig object containing backend configuration.

    Returns:
        None: This function updates the input `deploy_cfg` object directly.
    """
    input_size_manager = InputSizeManager(cfg)
    size = input_size_manager.get_input_size_from_cfg("test")

    assert all(isinstance(i, int) and i > 0 for i in size)
    # default is static shape to prevent an unexpected error
    # when converting to OpenVINO IR
    w, h = size
    logger.info(f"Patching OpenVINO IR input shape: {size}")
    deploy_cfg.ir_config.input_shape = (w, h)
    deploy_cfg.backend_config.model_inputs = [ConfigDict(opt_shapes=ConfigDict(input=[-1, 3, h, w]))]


def patch_ir_scale_factor(deploy_cfg: ConfigDict, hyper_parameters: DetectionConfig):
    """Patch IR scale factor inplace from hyper parameters to deploy config.

    Args:
        deploy_cfg (ConfigDict): mmcv deploy config
        hyper_parameters (DetectionConfig): OTX detection hyper parameters
    """

    if hyper_parameters.tiling_parameters.enable_tiling:
        scale_ir_input = deploy_cfg.get("scale_ir_input", False)
        if scale_ir_input:
            tile_ir_scale_factor = hyper_parameters.tiling_parameters.tile_ir_scale_factor
            logger.info(f"Apply OpenVINO IR scale factor: {tile_ir_scale_factor}")
            ir_input_shape = deploy_cfg.backend_config.model_inputs[0].opt_shapes.input
            ir_input_shape[2] = int(ir_input_shape[2] * tile_ir_scale_factor)  # height
            ir_input_shape[3] = int(ir_input_shape[3] * tile_ir_scale_factor)  # width
            deploy_cfg.ir_config.input_shape = (ir_input_shape[3], ir_input_shape[2])  # width, height
            deploy_cfg.backend_config.model_inputs = [
                ConfigDict(opt_shapes=ConfigDict(input=[1, 3, ir_input_shape[2], ir_input_shape[3]]))
            ]
            print(f"-----------------> x {tile_ir_scale_factor} = {ir_input_shape}")


def monkey_patched_nms(ctx, bboxes, scores, iou_threshold, offset, score_threshold, max_num):
    """Runs MMCVs NMS with torchvision.nms, or forces NMS from MMCV to run on CPU."""
    is_filtering_by_score = score_threshold > 0
    if is_filtering_by_score:
        valid_mask = scores > score_threshold
        bboxes, scores = bboxes[valid_mask], scores[valid_mask]
        valid_inds = torch.nonzero(valid_mask, as_tuple=False).squeeze(dim=1)

    if bboxes.dtype == torch.bfloat16:
        bboxes = bboxes.to(torch.float32)
    if scores.dtype == torch.bfloat16:
        scores = scores.to(torch.float32)

    if offset == 0:
        inds = tv_nms(bboxes, scores, float(iou_threshold))
    else:
        device = bboxes.device
        bboxes = bboxes.to("cpu")
        scores = scores.to("cpu")
        inds = ext_module.nms(bboxes, scores, iou_threshold=float(iou_threshold), offset=offset)
        bboxes = bboxes.to(device)
        scores = scores.to(device)

    if max_num > 0:
        inds = inds[:max_num]
    if is_filtering_by_score:
        inds = valid_inds[inds]
    return inds


def monkey_patched_roi_align(self, input, rois):
    """Replaces MMCVs roi align with the one from torchvision.

    Args:
        self: patched instance
        input: NCHW images
        rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
    """

    if "aligned" in tv_roi_align.__code__.co_varnames:
        return tv_roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned)
    else:
        if self.aligned:
            rois -= rois.new_tensor([0.0] + [0.5 / self.spatial_scale] * 4)
        return tv_roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio)
