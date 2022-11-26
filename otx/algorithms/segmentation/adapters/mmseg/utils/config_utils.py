"""Collection of utils for task implementation in Segmentation Task."""

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


import copy
import logging
import math
from collections import defaultdict
from typing import List, Optional, Union

from mmcv import Config, ConfigDict

from otx.algorithms.common.adapters.mmcv.utils import (  # patch_color_conversion,
    get_meta_keys,
    is_epoch_based_runner,
    prepare_work_dir,
    remove_from_config,
)
from otx.algorithms.segmentation.configs.base import SegmentationConfig
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import Domain, LabelEntity
from otx.api.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    DirectoryPathCheck,
    check_input_parameters_type,
)

logger = logging.getLogger(__name__)


@check_input_parameters_type({"work_dir": DirectoryPathCheck})
def patch_config(
    config: Config,
    work_dir: str,
    labels: List[LabelEntity],
    random_seed: Optional[int] = None,
    distributed: bool = False,
):  # pylint: disable=too-many-branches
    """Update config function."""
    # Set runner if not defined.
    if "runner" not in config:
        config.runner = {"type": "IterBasedRunner"}

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
    patch_datasets(config)

    if "log_config" not in config:
        config.log_config = ConfigDict()
    config.log_config.hooks = []

    if "evaluation" not in config:
        config.evaluation = ConfigDict()
    config.evaluation["efficient_test"] = True
    evaluation_metric = config.evaluation.get("metric")
    if evaluation_metric is not None:
        config.evaluation.save_best = evaluation_metric

    if "checkpoint_config" not in config:
        config.checkpoint_config = ConfigDict()
    config.checkpoint_config.max_keep_ckpts = 5
    config.checkpoint_config.interval = config.evaluation.get("interval", 1)

    label_names = ["background"] + [label.name for label in labels]
    set_data_classes(config, label_names)
    set_num_classes(config, len(label_names))

    if "test_cfg" not in config.model:
        config.model.test_cfg = ConfigDict()
    config.model.test_cfg.return_repr_vector = True

    set_distributed_mode(config, distributed)
    remove_from_config(config, "img_norm_cfg")

    config.gpu_ids = range(1)
    config.work_dir = work_dir
    config.seed = random_seed


@check_input_parameters_type()
def set_hyperparams(config: Config, hyperparams: SegmentationConfig):
    """Set function for hyperparams (SegmentationConfig)."""
    config.data.samples_per_gpu = int(hyperparams.learning_parameters.batch_size)
    config.data.workers_per_gpu = int(hyperparams.learning_parameters.num_workers)
    config.optimizer.lr = float(hyperparams.learning_parameters.learning_rate)

    # set proper number of iterations
    fixed_iters = int(hyperparams.learning_parameters.learning_rate_fixed_iters)
    warmup_iters = int(hyperparams.learning_parameters.learning_rate_warmup_iters)
    main_iters = int(hyperparams.learning_parameters.num_iters)
    total_iterations = fixed_iters + warmup_iters + main_iters

    config.params_config.iters = fixed_iters
    config.lr_config.fixed_iters = fixed_iters
    config.find_unused_parameters = fixed_iters > 0
    config.lr_config.warmup_iters = warmup_iters
    if is_epoch_based_runner(config.runner):
        init_num_iterations = config.runner.max_epochs
        config.runner.max_epochs = total_iterations
    else:
        init_num_iterations = config.runner.max_iters
        config.runner.max_iters = total_iterations

    # rescale the learning schedules
    schedule_scale = float(total_iterations) / float(init_num_iterations) if init_num_iterations > 0 else 0.0
    rescale_num_iterations(config, schedule_scale)


@check_input_parameters_type()
def rescale_num_iterations(config: Union[Config, ConfigDict], schedule_scale: float):
    """Rescale number of iterations for lr scheduler."""
    if config.lr_config.policy == "customstep":
        config.lr_config.step = [int(schedule_scale * step) for step in config.lr_config.step]
    elif config.lr_config.policy == "customcos":
        config.lr_config.periods = [int(schedule_scale * period) for period in config.lr_config.periods]

    def _rescale_num_iters(head, schedule_scale):
        losses = head.loss_decode
        if not isinstance(losses, (tuple, list)):
            losses = [losses]

        for loss in losses:
            for target_attr in ["scale_cfg", "conf_penalty_weight"]:
                if not hasattr(loss, target_attr):
                    continue

                if not hasattr(loss[target_attr], "num_iters"):
                    continue

                loss[target_attr].num_iters = int(schedule_scale * loss[target_attr].num_iters)
        head.loss_decode = losses

    # rescale number of iterations for
    for head_type in ["decode_head", "auxiliary_head"]:
        if not hasattr(config.model, head_type):
            continue

        heads = config.model[head_type]
        if isinstance(heads, (tuple, list)):
            for head in heads:
                _rescale_num_iters(head, schedule_scale)
        elif isinstance(heads, dict):
            _rescale_num_iters(heads, schedule_scale)
        config.model[head_type] = heads


@check_input_parameters_type()
def patch_adaptive_repeat_dataset(
    config: Union[Config, ConfigDict], num_samples: int, decay: float = 0.002, factor: float = 10
):
    """Patch the repeat times and training epochs adatively."""
    if config.data.train.type != "RepeatDataset":
        return

    if not config.data.train.get("adaptive_repeat", False):
        return

    if not is_epoch_based_runner(config.runner):
        return

    # calculate new number of iterations
    init_max_epoch = config.runner.max_epochs
    new_repeat_times = max(round(math.exp(-decay * num_samples) * factor), 1)
    new_max_epoch = math.ceil(init_max_epoch / new_repeat_times)
    if new_max_epoch <= 1:
        return

    # set proper number of iterations
    config.runner.max_epochs = new_max_epoch
    config.data.train.times = new_repeat_times

    # rescale the learning schedules
    schedule_scale = float(new_max_epoch) / float(init_max_epoch)
    config.params_config.iters = math.ceil(schedule_scale * config.params_config.iters)
    config.lr_config.fixed_iters = math.ceil(schedule_scale * config.lr_config.fixed_iters)
    config.lr_config.warmup_iters = math.ceil(schedule_scale * config.lr_config.warmup_iters)
    rescale_num_iterations(config, schedule_scale)


@check_input_parameters_type({"train_dataset": DatasetParamTypeCheck, "val_dataset": DatasetParamTypeCheck})
def prepare_for_training(
    config: Config,
    train_dataset: DatasetEntity,
    val_dataset: DatasetEntity,
    time_monitor: TimeMonitorCallback,
    learning_curves: defaultdict,
) -> Config:
    """Prepare configs for training phase."""
    config = copy.deepcopy(config)
    prepare_work_dir(config)

    config.data.val.otx_dataset = val_dataset
    config.data.val.labels = val_dataset.get_labels()
    if "otx_dataset" in config.data.train:
        config.data.train.otx_dataset = train_dataset
        config.data.train.labels = train_dataset.get_labels()
    else:
        config.data.train.dataset.otx_dataset = train_dataset
        config.data.train.dataset.labels = train_dataset.get_labels()

    train_num_samples = len(train_dataset)
    patch_adaptive_repeat_dataset(config, train_num_samples)

    config.custom_hooks.append({"type": "OTXProgressHook", "time_monitor": time_monitor, "verbose": True})
    config.log_config.hooks.append({"type": "OTXLoggerHook", "curves": learning_curves})

    return config


@check_input_parameters_type()
def set_distributed_mode(config: Config, distributed: bool):
    """Setter distributed into config."""
    if distributed:
        return

    norm_cfg = {"type": "BN", "requires_grad": True}

    def _replace_syncbn(_node, _norm_cfg):
        if _node.norm_cfg.type != "SyncBN":
            return

        _node.norm_cfg = _norm_cfg

    config.model.backbone.norm_cfg = norm_cfg
    for head_type in ("decode_head", "auxiliary_head"):
        head = config.model.get(head_type, None)
        if head is None:
            continue

        if isinstance(head, (tuple, list)):
            for sub_head in head:
                _replace_syncbn(sub_head, norm_cfg)
        else:
            _replace_syncbn(head, norm_cfg)


@check_input_parameters_type()
def set_data_classes(config: Config, label_names: List[str]):
    """Setter data classes into config."""
    # Save labels in data configs.
    for subset in ("train", "val", "test"):
        cfg = config.data[subset]
        if cfg.type == "RepeatDataset":
            cfg.dataset.classes = label_names
        else:
            cfg.classes = label_names
        config.data[subset].classes = label_names


@check_input_parameters_type()
def set_num_classes(config: Config, num_classes: int):
    """Setter num_classes function."""
    assert num_classes > 1

    for head_type in ("decode_head", "auxiliary_head"):
        heads = config.model.get(head_type, None)
        if heads is None:
            continue

        if isinstance(heads, (tuple, list)):
            for head in heads:
                if hasattr(head, "loss_target") and head.loss_target == "gt_class_borders":
                    continue

                head.num_classes = num_classes
        elif isinstance(heads, dict):
            heads["num_classes"] = num_classes


@check_input_parameters_type()
def patch_datasets(config: Config, domain=Domain.SEGMENTATION):
    """Update dataset configs."""
    assert "data" in config
    for subset in ("train", "val", "test", "unlabeled"):
        cfg = config.data.get(subset, None)
        if not cfg:
            continue
        if cfg.type == "RepeatDataset":
            cfg = cfg.dataset

        if subset == "unlabeled":
            cfg.type = "UnlabeledSegDataset"
            cfg.orig_type = "MPASegDataset"
        else:
            cfg.type = "MPASegDataset"
        cfg.otx_dataset = None
        cfg.labels = None

        remove_from_config(cfg, "ann_dir")
        remove_from_config(cfg, "img_dir")
        remove_from_config(cfg, "data_root")
        remove_from_config(cfg, "split")
        remove_from_config(cfg, "classes")

        for pipeline_step in cfg.pipeline:
            if subset == "train" and pipeline_step.type == "Collect":
                pipeline_step = get_meta_keys(pipeline_step)
        patch_color_conversion(cfg.pipeline)


def patch_color_conversion(pipeline):
    """Default data format for OTX is RGB, while mmx uses BGR, so negate the color conversion flag."""
    if isinstance(pipeline, dict):
        for k in pipeline.keys():
            patch_color_conversion(pipeline[k])
    else:
        for pipeline_step in pipeline:
            if pipeline_step.type == "Normalize":
                to_rgb = False
                if "to_rgb" in pipeline_step:
                    to_rgb = pipeline_step.to_rgb
                to_rgb = not bool(to_rgb)
                pipeline_step.to_rgb = to_rgb
            elif pipeline_step.type == "MultiScaleFlipAug":
                patch_color_conversion(pipeline_step.transforms)


def patch_evaluation(config: Config):
    """Update evaluation configs."""
    cfg = config.evaluation
    cfg.pop("classwise", None)
    cfg.metric = "mDice"
    cfg.save_best = "mDice"
    cfg.rule = "greater"
    # EarlyStoppingHook
    config.early_stop_metric = "mDice"
