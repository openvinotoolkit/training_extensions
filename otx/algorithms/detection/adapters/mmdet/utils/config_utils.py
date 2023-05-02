"""Collection of utils for task implementation in Detection Task."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import math
from typing import List, Optional, Union

from mmcv import Config, ConfigDict

from otx.algorithms.common.adapters.mmcv.utils import (
    get_configs_by_keys,
    get_configs_by_pairs,
    get_dataset_configs,
    get_meta_keys,
    is_epoch_based_runner,
    patch_color_conversion,
    prepare_work_dir,
    remove_from_config,
    update_config,
)
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.algorithms.detection.utils.data import (
    adaptive_tile_params,
    format_list_to_str,
    get_anchor_boxes,
    get_sizes_from_dataset_entity,
)
from otx.api.entities.datasets import DatasetEntity, DatasetPurpose
from otx.api.entities.label import Domain, LabelEntity

try:
    from sklearn.cluster import KMeans

    __all__ = ["KMeans"]

    KMEANS_IMPORT = True
except ImportError:
    KMEANS_IMPORT = False


logger = get_logger()


def patch_config(
    config: Config,
    work_dir: str,
    labels: List[LabelEntity],
):  # pylint: disable=too-many-branches
    """Update config function."""

    # Add training cancelation hook.
    if "custom_hooks" not in config:
        config.custom_hooks = []
    if "CancelTrainingHook" not in {hook.type for hook in config.custom_hooks}:
        config.custom_hooks.append(ConfigDict({"type": "CancelTrainingHook"}))

    # Remove high level data pipelines definition leaving them only inside `data` section.
    remove_from_config(config, "train_pipeline")
    remove_from_config(config, "test_pipeline")

    evaluation_metric = config.evaluation.get("metric")
    if evaluation_metric is not None:
        config.evaluation.save_best = evaluation_metric
    config.checkpoint_config.max_keep_ckpts = 5
    config.checkpoint_config.interval = config.evaluation.get("interval", 1)

    set_data_classes(config, labels)

    config.gpu_ids = range(1)
    config.work_dir = work_dir


def patch_model_config(
    config: Config,
    labels: List[LabelEntity],
):
    """Patch model config."""
    set_num_classes(config, len(labels))


def set_hyperparams(config: Config, hyperparams: DetectionConfig):
    """Set function for hyperparams (DetectionConfig)."""
    config.data.samples_per_gpu = int(hyperparams.learning_parameters.batch_size)
    config.data.workers_per_gpu = int(hyperparams.learning_parameters.num_workers)
    config.optimizer.lr = float(hyperparams.learning_parameters.learning_rate)

    total_iterations = int(hyperparams.learning_parameters.num_iters)

    config.lr_config.warmup_iters = int(hyperparams.learning_parameters.learning_rate_warmup_iters)
    if config.lr_config.warmup_iters == 0:
        config.lr_config.warmup = None
    if is_epoch_based_runner(config.runner):
        config.runner.max_epochs = total_iterations
    else:
        config.runner.max_iters = total_iterations


def patch_adaptive_repeat_dataset(
    config: Union[Config, ConfigDict],
    num_samples: int,
    decay: float = -0.002,
    factor: float = 30,
):
    """Patch the repeat times and training epochs adatively.

    Frequent dataloading inits and evaluation slow down training when the
    sample size is small. Adjusting epoch and dataset repetition based on
    empirical exponential decay improves the training time by applying high
    repeat value to small sample size dataset and low repeat value to large
    sample.

    :param config: mmcv config
    :param num_samples: number of training samples
    :param decay: decaying rate
    :param factor: base repeat factor
    """
    data_train = config.data.train
    if data_train.type == "MultiImageMixDataset":
        data_train = data_train.dataset
    if data_train.type == "RepeatDataset" and getattr(data_train, "adaptive_repeat_times", False):
        if is_epoch_based_runner(config.runner):
            cur_epoch = config.runner.max_epochs
            new_repeat = max(round(math.exp(decay * num_samples) * factor), 1)
            new_epoch = math.ceil(cur_epoch / new_repeat)
            if new_epoch == 1:
                return
            config.runner.max_epochs = new_epoch
            data_train.times = new_repeat


def prepare_for_training(
    config: Union[Config, ConfigDict],
    data_config: ConfigDict,
) -> Union[Config, ConfigDict]:
    """Prepare configs for training phase."""
    prepare_work_dir(config)

    train_num_samples = 0
    for subset in ["train", "val", "test"]:
        data_config_ = data_config.data.get(subset)
        config_ = config.data.get(subset)
        if data_config_ is None:
            continue
        for key in ["otx_dataset", "labels"]:
            found = get_configs_by_keys(data_config_, key, return_path=True)
            if len(found) == 0:
                continue
            assert len(found) == 1
            if subset == "train" and key == "otx_dataset":
                found_value = list(found.values())[0]
                if found_value:
                    train_num_samples = len(found_value)
            update_config(config_, found)

    if train_num_samples > 0:
        patch_adaptive_repeat_dataset(config, train_num_samples)

    return config


def set_data_classes(config: Config, labels: List[LabelEntity]):
    """Setter data classes into config."""
    # Save labels in data configs.
    for subset in ("train", "val", "test"):
        for cfg in get_dataset_configs(config, subset):
            cfg.labels = labels
            #  config.data[subset].labels = labels


def set_num_classes(config: Config, num_classes: int):
    """Set num classes."""
    # Set proper number of classes in model's detection heads.
    head_names = ("mask_head", "bbox_head", "segm_head")
    if "roi_head" in config.model:
        for head_name in head_names:
            if head_name in config.model.roi_head:
                if isinstance(config.model.roi_head[head_name], List):
                    for head in config.model.roi_head[head_name]:
                        head.num_classes = num_classes
                else:
                    config.model.roi_head[head_name].num_classes = num_classes
    else:
        for head_name in head_names:
            if head_name in config.model:
                config.model[head_name].num_classes = num_classes
    # FIXME. ?
    # self.config.model.CLASSES = label_names


def patch_datasets(
    config: Config,
    domain: Domain = Domain.DETECTION,
    subsets: Optional[List[str]] = None,
    **kwargs,
):
    """Update dataset configs."""
    assert "data" in config
    assert "type" in kwargs

    # This code is for nncf, if we don't consider nncf, this code could be
    # domain = config.get("domain", Domain.DETECTION)
    domain = config.get("domain", domain)

    if subsets is None:
        subsets = ["train", "val", "test", "unlabeled"]

    def update_pipeline(cfg):
        if subset == "train":
            for collect_cfg in get_configs_by_pairs(cfg, dict(type="Collect")):
                get_meta_keys(collect_cfg)
        for cfg_ in get_configs_by_pairs(cfg, dict(type="LoadImageFromFile")):
            cfg_.type = "LoadImageFromOTXDataset"
        for cfg_ in get_configs_by_pairs(cfg, dict(type="LoadAnnotations")):
            cfg_.type = "LoadAnnotationFromOTXDataset"
            cfg_.domain = domain
            cfg_.min_size = cfg.pop("min_size", -1)

    for subset in subsets:
        if subset not in config.data:
            continue
        config.data[f"{subset}_dataloader"] = config.data.get(f"{subset}_dataloader", ConfigDict())

        remove_from_config(config.data[subset], "ann_file")
        remove_from_config(config.data[subset], "img_prefix")
        remove_from_config(config.data[subset], "classes")  # Get from DatasetEntity

        cfgs = get_dataset_configs(config, subset)
        for cfg in cfgs:
            cfg.domain = domain
            cfg.otx_dataset = None
            cfg.labels = None
            cfg.update(kwargs)

            remove_from_config(cfg, "ann_file")
            remove_from_config(cfg, "img_prefix")
            remove_from_config(cfg, "classes")  # Get from DatasetEntity

            update_pipeline(cfg)

    patch_color_conversion(config)


def patch_evaluation(config: Config):
    """Update evaluation configs."""
    cfg = config.evaluation
    # CocoDataset.evaluate -> CustomDataset.evaluate
    cfg.pop("classwise", None)
    cfg.metric = "mAP"
    cfg.save_best = "mAP"
    # EarlyStoppingHook
    config.early_stop_metric = "mAP"


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
        config (dict): MPA config containing configuration settings.
        hparams (DetectionConfig): DetectionConfig containing hyperparameters.
        dataset (DatasetEntity, optional): A dataset entity. Defaults to None.

    Returns:
        dict: The updated configuration dictionary.
    """
    if hparams.tiling_parameters.enable_tiling:
        logger.info("Tiling enabled")

        if dataset and dataset.purpose != DatasetPurpose.INFERENCE and hparams.tiling_parameters.enable_adaptive_params:
            adaptive_tile_params(hparams.tiling_parameters, dataset)

        if dataset and dataset.purpose == DatasetPurpose.INFERENCE:
            config.get("data", ConfigDict()).get("val_dataloader", ConfigDict()).update(ConfigDict(samples_per_gpu=1))
            config.get("data", ConfigDict()).get("test_dataloader", ConfigDict()).update(ConfigDict(samples_per_gpu=1))
            config.get("data", ConfigDict(samples_per_gpu=1))

        if hparams.tiling_parameters.enable_tile_classifier:
            logger.info("Tile classifier enabled")
            logger.info(f"Patch model from: {config.model.type} to CustomMaskRCNNTileOptimized")
            config.model.type = "CustomMaskRCNNTileOptimized"

            if config.model.backbone.type == "efficientnet_b2b":
                learning_rate = 0.002
                logger.info(
                    f"Patched {config.model.backbone.type} LR: "
                    f"{hparams.learning_parameters.learning_rate} -> {learning_rate}"
                )
                hparams.learning_parameters.learning_rate = learning_rate

            config.data.train.filter_empty_gt = False

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
