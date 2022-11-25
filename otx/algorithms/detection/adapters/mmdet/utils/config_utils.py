"""Collection of utils for task implementation in Detection Task."""

# Copyright (C) 2021 Intel Corporation
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
import os
from collections import defaultdict
from typing import List, Optional, Union

import torch
from mmcv import Config, ConfigDict
from mmdet.models.detectors import BaseDetector

from mpa.utils.logger import get_logger
from otx.algorithms.common.adapters.mmcv.utils import (
    get_meta_keys,
    is_epoch_based_runner,
    patch_color_conversion,
    prepare_work_dir,
    remove_from_config,
)
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.algorithms.detection.utils.data import (
    format_list_to_str,
    get_anchor_boxes,
    get_sizes_from_dataset_entity,
)
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import Domain, LabelEntity
from otx.api.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    DirectoryPathCheck,
    check_input_parameters_type,
)

try:
    from sklearn.cluster import KMeans

    __all__ = ["KMeans"]

    KMEANS_IMPORT = True
except ImportError:
    KMEANS_IMPORT = False


logger = get_logger()


@check_input_parameters_type({"work_dir": DirectoryPathCheck})
def patch_config(
    config: Config,
    work_dir: str,
    labels: List[LabelEntity],
    domain: Domain,
    random_seed: Optional[int] = None,
):  # pylint: disable=too-many-branches
    """Update config function."""
    # Set runner if not defined.
    if "runner" not in config:
        config.runner = ConfigDict({"type": "EpochBasedRunner"})

    # Check that there is no conflict in specification of number of training epochs.
    # Move global definition of epochs inside runner config.
    if "total_epochs" in config:
        if is_epoch_based_runner(config.runner):
            if config.runner.max_epochs != config.total_epochs:
                logger.warning("Conflicting declaration of training epochs number.")
            config.runner.max_epochs = config.total_epochs
        else:
            logger.warning(f"Total number of epochs set for an iteration based runner {config.runner.type}.")
        remove_from_config(config, "total_epochs")

    # Change runner's type.
    if is_epoch_based_runner(config.runner):
        logger.info(f"Replacing runner from {config.runner.type} to EpochRunnerWithCancel.")
        config.runner.type = "EpochRunnerWithCancel"
    else:
        logger.info(f"Replacing runner from {config.runner.type} to IterBasedRunnerWithCancel.")
        config.runner.type = "IterBasedRunnerWithCancel"

    # Add training cancelation hook.
    if "custom_hooks" not in config:
        config.custom_hooks = []
    if "CancelTrainingHook" not in {hook.type for hook in config.custom_hooks}:
        config.custom_hooks.append({"type": "CancelTrainingHook"})

    # Remove high level data pipelines definition leaving them only inside `data` section.
    remove_from_config(config, "train_pipeline")
    remove_from_config(config, "test_pipeline")
    # Patch data pipeline, making it OTX-compatible.
    patch_datasets(config, domain)

    # Remove FP16 config if running on CPU device and revert to FP32
    # https://github.com/pytorch/pytorch/issues/23377
    if not torch.cuda.is_available() and "fp16" in config:
        logger.info("Revert FP16 to FP32 on CPU device")
        remove_from_config(config, "fp16")

    if "log_config" not in config:
        config.log_config = ConfigDict()
    if "evaluation" not in config:
        config.evaluation = ConfigDict()
    evaluation_metric = config.evaluation.get("metric")
    if evaluation_metric is not None:
        config.evaluation.save_best = evaluation_metric
    if "checkpoint_config" not in config:
        config.checkpoint_config = ConfigDict()
    config.checkpoint_config.max_keep_ckpts = 5
    config.checkpoint_config.interval = config.evaluation.get("interval", 1)
    set_data_classes(config, labels)
    config.gpu_ids = range(1)
    config.work_dir = work_dir
    config.seed = random_seed


@check_input_parameters_type()
def set_hyperparams(config: Config, hyperparams: DetectionConfig):
    """Set function for hyperparams (DetectionConfig)."""
    config.optimizer.lr = float(hyperparams.learning_parameters.learning_rate)
    config.lr_config.warmup_iters = int(hyperparams.learning_parameters.learning_rate_warmup_iters)
    if config.lr_config.warmup_iters == 0:
        config.lr_config.warmup = None
    config.data.samples_per_gpu = int(hyperparams.learning_parameters.batch_size)
    config.data.workers_per_gpu = int(hyperparams.learning_parameters.num_workers)
    total_iterations = int(hyperparams.learning_parameters.num_iters)
    if is_epoch_based_runner(config.runner):
        config.runner.max_epochs = total_iterations
    else:
        config.runner.max_iters = total_iterations


@check_input_parameters_type()
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


@check_input_parameters_type({"train_dataset": DatasetParamTypeCheck, "val_dataset": DatasetParamTypeCheck})
def prepare_for_training(
    config: Union[Config, ConfigDict],
    train_dataset: DatasetEntity,
    val_dataset: DatasetEntity,
    time_monitor: TimeMonitorCallback,
    learning_curves: defaultdict,
) -> Config:
    """Prepare configs for training phase."""
    prepare_work_dir(config)
    data_train = get_data_cfg(config)
    data_train.otx_dataset = train_dataset
    config.data.val.otx_dataset = val_dataset
    patch_adaptive_repeat_dataset(config, len(train_dataset))
    config.custom_hooks.append({"type": "OTXProgressHook", "time_monitor": time_monitor, "verbose": True})
    config.log_config.hooks.append({"type": "OTXLoggerHook", "curves": learning_curves})
    return config


@check_input_parameters_type()
def set_data_classes(config: Config, labels: List[LabelEntity]):
    """Setter data classes into config."""
    # Save labels in data configs.
    for subset in ("train", "val", "test"):
        cfg = get_data_cfg(config, subset)
        cfg.labels = labels
        config.data[subset].labels = labels

    # Set proper number of classes in model's detection heads.
    head_names = ("mask_head", "bbox_head", "segm_head")
    num_classes = len(labels)
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


@check_input_parameters_type()
def patch_datasets(config: Config, domain: Domain):
    """Update dataset configs."""

    assert "data" in config
    for subset in ("train", "val", "test", "unlabeled"):
        cfg = config.data.get(subset, None)
        if not cfg:
            continue
        if cfg.type in ("RepeatDataset", "MultiImageMixDataset"):
            cfg = cfg.dataset
        cfg.type = "MPADetDataset"
        cfg.domain = domain
        cfg.otx_dataset = None
        cfg.labels = None
        remove_from_config(cfg, "ann_file")
        remove_from_config(cfg, "img_prefix")
        remove_from_config(cfg, "classes")  # Get from DatasetEntity
        for pipeline_step in cfg.pipeline:
            if pipeline_step.type == "LoadImageFromFile":
                pipeline_step.type = "LoadImageFromOTXDataset"
            if pipeline_step.type == "LoadAnnotations":
                pipeline_step.type = "LoadAnnotationFromOTXDataset"
                pipeline_step.domain = domain
                pipeline_step.min_size = cfg.pop("min_size", -1)
            if subset == "train" and pipeline_step.type == "Collect":
                pipeline_step = get_meta_keys(pipeline_step)
        patch_color_conversion(cfg.pipeline)


def patch_evaluation(config: Config):
    """Update evaluation configs."""
    cfg = config.evaluation
    # CocoDataset.evaluate -> CustomDataset.evaluate
    cfg.pop("classwise", None)
    cfg.metric = "mAP"
    cfg.save_best = "mAP"
    # EarlyStoppingHook
    config.early_stop_metric = "mAP"


# TODO Replace this with function in common
def patch_data_pipeline(config: Config, template_file_path: str):
    """Update data_pipeline configs."""
    base_dir = os.path.abspath(os.path.dirname(template_file_path))
    # FIXME Loading data pipeline is hard-coded, it should be loaded from recipe of algorithm
    if hasattr(config, "data") and hasattr(config.data, "unlabeled"):
        data_pipeline_path = os.path.join(base_dir, "data_pipeline_semisl.py")
    else:
        data_pipeline_path = os.path.join(base_dir, "data_pipeline.py")
    if os.path.exists(data_pipeline_path):
        data_pipeline_cfg = Config.fromfile(data_pipeline_path)
        config.merge_from_dict(data_pipeline_cfg)


@check_input_parameters_type({"dataset": DatasetParamTypeCheck})
def cluster_anchors(config: Config, dataset: DatasetEntity, model: BaseDetector):
    """Update configs for cluster_anchors."""
    if not KMEANS_IMPORT:
        raise ImportError(
            "Sklearn package is not installed. To enable anchor boxes clustering, please install "
            "packages from requirements/optional.txt or just scikit-learn package."
        )

    logger.info("Collecting statistics from training dataset to cluster anchor boxes...")
    [target_wh] = [
        transforms.img_scale for transforms in config.data.test.pipeline if transforms.type == "MultiScaleFlipAug"
    ]
    prev_generator = config.model.bbox_head.anchor_generator
    group_as = [len(width) for width in prev_generator.widths]
    wh_stats = get_sizes_from_dataset_entity(dataset, list(target_wh))

    if len(wh_stats) < sum(group_as):
        logger.warning(
            f"There are not enough objects to cluster: {len(wh_stats)} were detected, while it should be "
            f"at least {sum(group_as)}. Anchor box clustering was skipped."
        )
        return config, model

    widths, heights = get_anchor_boxes(wh_stats, group_as)
    logger.info(
        f"Anchor boxes widths have been updated from {format_list_to_str(prev_generator.widths)} "
        f"to {format_list_to_str(widths)}"
    )
    logger.info(
        f"Anchor boxes heights have been updated from {format_list_to_str(prev_generator.heights)} "
        f"to {format_list_to_str(heights)}"
    )
    config_generator = config.model.bbox_head.anchor_generator
    config_generator.widths, config_generator.heights = widths, heights

    model_generator = model.bbox_head.anchor_generator
    model_generator.widths, model_generator.heights = widths, heights
    model_generator.base_anchors = model_generator.gen_base_anchors()

    config.model.bbox_head.anchor_generator = config_generator
    model.bbox_head.anchor_generator = model_generator
    return config, model


# TODO Replace this with function in common
@check_input_parameters_type()
def get_data_cfg(config: Union[Config, ConfigDict], subset: str = "train") -> Config:
    """Return dataset configs."""
    data_cfg = config.data[subset]
    while "dataset" in data_cfg:
        data_cfg = data_cfg.dataset
    return data_cfg
