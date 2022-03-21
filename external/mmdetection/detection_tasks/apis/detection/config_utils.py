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
import glob
import math
import os
import tempfile
from collections import defaultdict
from typing import List, Optional

from mmcv import Config, ConfigDict
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label import LabelEntity, Domain
from ote_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback

from detection_tasks.extension.datasets.data_utils import get_anchor_boxes, \
    get_sizes_from_dataset_entity, format_list_to_str
from mmdet.models.detectors import BaseDetector
from mmdet.utils.logger import get_root_logger

from .configuration import OTEDetectionConfig

try:
    from sklearn.cluster import KMeans
    kmeans_import = True
except ImportError:
    kmeans_import = False


logger = get_root_logger()


def is_epoch_based_runner(runner_config: ConfigDict):
    return 'Epoch' in runner_config.type


def patch_config(config: Config, work_dir: str, labels: List[LabelEntity], domain: Domain, random_seed: Optional[int] = None):
    # Set runner if not defined.
    if 'runner' not in config:
        config.runner = {'type': 'EpochBasedRunner'}

    # Check that there is no conflict in specification of number of training epochs.
    # Move global definition of epochs inside runner config.
    if 'total_epochs' in config:
        if is_epoch_based_runner(config.runner):
            if config.runner.max_epochs != config.total_epochs:
                logger.warning('Conflicting declaration of training epochs number.')
            config.runner.max_epochs = config.total_epochs
        else:
            logger.warning(f'Total number of epochs set for an iteration based runner {config.runner.type}.')
        remove_from_config(config, 'total_epochs')

    # Change runner's type.
    if is_epoch_based_runner(config.runner):
        logger.info(f'Replacing runner from {config.runner.type} to EpochRunnerWithCancel.')
        config.runner.type = 'EpochRunnerWithCancel'
    else:
        logger.info(f'Replacing runner from {config.runner.type} to IterBasedRunnerWithCancel.')
        config.runner.type = 'IterBasedRunnerWithCancel'

    # Add training cancelation hook.
    if 'custom_hooks' not in config:
        config.custom_hooks = []
    if 'CancelTrainingHook' not in {hook.type for hook in config.custom_hooks}:
        config.custom_hooks.append({'type': 'CancelTrainingHook'})

    # Remove high level data pipelines definition leaving them only inside `data` section.
    remove_from_config(config, 'train_pipeline')
    remove_from_config(config, 'test_pipeline')

    # Patch data pipeline, making it OTE-compatible.
    patch_datasets(config, domain)

    if 'log_config' not in config:
        config.log_config = ConfigDict()
    # config.log_config.hooks = []

    if 'evaluation' not in config:
        config.evaluation = ConfigDict()
    evaluation_metric = config.evaluation.get('metric')
    if evaluation_metric is not None:
        config.evaluation.save_best = evaluation_metric

    if 'checkpoint_config' not in config:
        config.checkpoint_config = ConfigDict()
    config.checkpoint_config.max_keep_ckpts = 5
    config.checkpoint_config.interval = config.evaluation.get('interval', 1)

    set_data_classes(config, labels)

    config.gpu_ids = range(1)
    config.work_dir = work_dir
    config.seed = random_seed


def set_hyperparams(config: Config, hyperparams: OTEDetectionConfig):
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


def patch_adaptive_repeat_dataset(config: Config, num_samples: int,
    decay: float = -0.002, factor: float = 30):
    """ Patch the repeat times and training epochs adatively

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
    if config.data.train.type == 'RepeatDataset' and getattr(
      config.data.train, 'adaptive_repeat_times', False):
        if is_epoch_based_runner(config.runner):
            cur_epoch = config.runner.max_epochs
            new_repeat = max(round(math.exp(decay * num_samples) * factor), 1)
            new_epoch = math.ceil(cur_epoch / new_repeat)
            if new_epoch == 1:
                return
            config.runner.max_epochs = new_epoch
            config.data.train.times = new_repeat


def prepare_for_testing(config: Config, dataset: DatasetEntity) -> Config:
    config = copy.deepcopy(config)
    # FIXME. Should working directories be modified here?
    config.data.test.ote_dataset = dataset
    return config


def prepare_for_training(config: Config, train_dataset: DatasetEntity, val_dataset: DatasetEntity,
                         time_monitor: TimeMonitorCallback, learning_curves: defaultdict) -> Config:
    config = copy.deepcopy(config)
    prepare_work_dir(config)
    config.data.val.ote_dataset = val_dataset
    if 'ote_dataset' in config.data.train:
        config.data.train.ote_dataset = train_dataset
    else:
        config.data.train.dataset.ote_dataset = train_dataset
    patch_adaptive_repeat_dataset(config, len(train_dataset))
    config.custom_hooks.append({'type': 'OTEProgressHook', 'time_monitor': time_monitor, 'verbose': True})
    config.log_config.hooks.append({'type': 'OTELoggerHook', 'curves': learning_curves})
    return config


def config_to_string(config: Config) -> str:
    """
    Convert a full mmdetection config to a string.

    :param config: configuration object to convert
    :return str: string representation of the configuration
    """
    config_copy = copy.deepcopy(config)
    # Clean config up by removing dataset as this causes the pretty text parsing to fail.
    config_copy.data.test.ote_dataset = None
    config_copy.data.test.labels = None
    config_copy.data.val.ote_dataset = None
    config_copy.data.val.labels = None
    if 'ote_dataset' in config_copy.data.train:
        config_copy.data.train.ote_dataset = None
        config_copy.data.train.labels = None
    else:
        config_copy.data.train.dataset.ote_dataset = None
        config_copy.data.train.dataset.labels = None
    return Config(config_copy).pretty_text


def config_from_string(config_string: str) -> Config:
    """
    Generate an mmdetection config dict object from a string.

    :param config_string: string to parse
    :return config: configuration object
    """
    with tempfile.NamedTemporaryFile('w', suffix='.py') as temp_file:
        temp_file.write(config_string)
        temp_file.flush()
        return Config.fromfile(temp_file.name)


def save_config_to_file(config: Config):
    """ Dump the full config to a file. Filename is 'config.py', it is saved in the current work_dir. """
    filepath = os.path.join(config.work_dir, 'config.py')
    config_string = config_to_string(config)
    with open(filepath, 'w') as f:
        f.write(config_string)


def prepare_work_dir(config: Config) -> str:
    base_work_dir = config.work_dir
    checkpoint_dirs = glob.glob(os.path.join(base_work_dir, "checkpoints_round_*"))
    train_round_checkpoint_dir = os.path.join(base_work_dir, f"checkpoints_round_{len(checkpoint_dirs)}")
    os.makedirs(train_round_checkpoint_dir)
    logger.info(f"Checkpoints and logs for this training run are stored in {train_round_checkpoint_dir}")
    config.work_dir = train_round_checkpoint_dir
    if 'meta' not in config.runner:
        config.runner.meta = ConfigDict()
    config.runner.meta.exp_name = f"train_round_{len(checkpoint_dirs)}"
    # Save training config for debugging. It is saved in the checkpoint dir for this training round.
    # save_config_to_file(config)
    return train_round_checkpoint_dir


def set_data_classes(config: Config, labels: List[LabelEntity]):
    # Save labels in data configs.
    for subset in ('train', 'val', 'test'):
        cfg = config.data[subset]
        if cfg.type == 'RepeatDataset' or cfg.type == 'MultiImageMixDataset':
            cfg.dataset.labels = labels
        else:
            cfg.labels = labels
        config.data[subset].labels = labels

    # Set proper number of classes in model's detection heads.
    head_names = ('mask_head', 'bbox_head', 'segm_head')
    num_classes = len(labels)
    if 'roi_head' in config.model:
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


def patch_datasets(config: Config, domain):

    def patch_color_conversion(pipeline):
        # Default data format for OTE is RGB, while mmdet uses BGR, so negate the color conversion flag.
        for pipeline_step in pipeline:
            if pipeline_step.type == 'Normalize':
                to_rgb = False
                if 'to_rgb' in pipeline_step:
                    to_rgb = pipeline_step.to_rgb
                to_rgb = not bool(to_rgb)
                pipeline_step.to_rgb = to_rgb
            elif pipeline_step.type == 'MultiScaleFlipAug':
                patch_color_conversion(pipeline_step.transforms)

    assert 'data' in config
    for subset in ('train', 'val', 'test'):
        cfg = config.data[subset]
        if cfg.type == 'RepeatDataset' or cfg.type == 'MultiImageMixDataset':
            cfg = cfg.dataset
        cfg.type = 'OTEDataset'
        cfg.domain = domain
        cfg.ote_dataset = None
        cfg.labels = None
        remove_from_config(cfg, 'ann_file')
        remove_from_config(cfg, 'img_prefix')
        for pipeline_step in cfg.pipeline:
            if pipeline_step.type == 'LoadImageFromFile':
                pipeline_step.type = 'LoadImageFromOTEDataset'
            if pipeline_step.type == 'LoadAnnotations':
                pipeline_step.type = 'LoadAnnotationFromOTEDataset'
                pipeline_step.domain = domain
                pipeline_step.min_size = cfg.pop('min_size', -1)
        patch_color_conversion(cfg.pipeline)


def remove_from_config(config, key: str):
    if key in config:
        if isinstance(config, Config):
            del config._cfg_dict[key]
        elif isinstance(config, ConfigDict):
            del config[key]
        else:
            raise ValueError(f'Unknown config type {type(config)}')

def cluster_anchors(config: Config, dataset: DatasetEntity, model: BaseDetector):
    if not kmeans_import:
        raise ImportError('Sklearn package is not installed. To enable anchor boxes clustering, please install '
                          'packages from requirements/optional.txt or just scikit-learn package.')

    logger.info('Collecting statistics from training dataset to cluster anchor boxes...')
    [target_wh] = [transforms.img_scale for transforms in config.data.test.pipeline
                 if transforms.type == 'MultiScaleFlipAug']
    prev_generator = config.model.bbox_head.anchor_generator
    group_as = [len(width) for width in prev_generator.widths]
    wh_stats = get_sizes_from_dataset_entity(dataset, target_wh)

    if len(wh_stats) < sum(group_as):
        logger.warning(f'There are not enough objects to cluster: {len(wh_stats)} were detected, while it should be '
                       f'at least {sum(group_as)}. Anchor box clustering was skipped.')
        return config, model

    widths, heights = get_anchor_boxes(wh_stats, group_as)
    logger.info(f'Anchor boxes widths have been updated from {format_list_to_str(prev_generator.widths)} '
                                                        f'to {format_list_to_str(widths)}')
    logger.info(f'Anchor boxes heights have been updated from {format_list_to_str(prev_generator.heights)} '
                                                         f'to {format_list_to_str(heights)}')
    config_generator = config.model.bbox_head.anchor_generator
    config_generator.widths, config_generator.heights = widths, heights

    model_generator = model.bbox_head.anchor_generator
    model_generator.widths, model_generator.heights = widths, heights
    model_generator.base_anchors = model_generator.gen_base_anchors()

    config.model.bbox_head.anchor_generator = config_generator
    model.bbox_head.anchor_generator = model_generator
    return config, model
