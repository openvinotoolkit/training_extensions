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

import logging
import math
import os
import re
from copy import deepcopy
from typing import List, Optional

import torchreid
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.metrics import (CurveMetric, LineChartInfo, LineMetricsGroup,
                                      MetricsGroup, Performance, ScoreMetric)
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import default_progress_callback, TrainParameters
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask
from scripts.default_config import imagedata_kwargs, lr_scheduler_kwargs, optimizer_kwargs
from torchreid.apis.training import run_lr_finder, run_training
from torchreid_tasks.inference_task import OTEClassificationInferenceTask
from torchreid_tasks.monitors import DefaultMetricsMonitor
from torchreid_tasks.utils import (OTEClassificationDataset, TrainingProgressCallback)
from torchreid.ops import DataParallel
from torchreid.utils import load_pretrained_weights, set_random_seed
from ote_sdk.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)

logger = logging.getLogger(__name__)


class OTEClassificationTrainingTask(OTEClassificationInferenceTask, ITrainingTask):

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment):
        super().__init__(task_environment)
        self._aux_model_snap_paths = {}

    def cancel_training(self):
        """
        Called when the user wants to abort training.
        In this example, this is not implemented.

        :return: None
        """
        logger.info("Cancel training requested.")
        self.stop_callback.stop()

    @check_input_parameters_type()
    def save_model(self, output_model: ModelEntity):
        for name, path in self._aux_model_snap_paths.items():
            with open(path, 'rb') as read_file:
                output_model.set_data(name, read_file.read())
        self._save_model(output_model)

    def _generate_training_metrics_group(self) -> Optional[List[MetricsGroup]]:
        """
        Parses the classification logs to get metrics from the latest training run
        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []

        # Learning curves
        if self.metrics_monitor is not None:
            for key in self.metrics_monitor.get_metric_keys():
                metric_curve = CurveMetric(xs=self.metrics_monitor.get_metric_timestamps(key),
                                           ys=self.metrics_monitor.get_metric_values(key), name=key)
                visualization_info = LineChartInfo(name=key, x_axis_label="Timestamp", y_axis_label=key)
                output.append(LineMetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))

        return output

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def train(self, dataset: DatasetEntity, output_model: ModelEntity,
              train_parameters: Optional[TrainParameters] = None):
        """ Trains a model on a dataset """

        train_model = deepcopy(self._model)

        if train_parameters is not None:
            update_progress_callback = train_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback

        if self._multilabel:
            self._cfg.train.lr = self._cfg.train.lr / self._cfg.sc_integration.lr_scale
            self._cfg.train.max_epoch = max(int(self._cfg.train.max_epoch / self._cfg.sc_integration.epoch_scale), 1)

        time_monitor = TrainingProgressCallback(update_progress_callback, num_epoch=self._cfg.train.max_epoch,
                                                num_train_steps=math.ceil(len(dataset.get_subset(Subset.TRAINING)) /
                                                                          self._cfg.train.batch_size),
                                                num_val_steps=0, num_test_steps=0)

        self.metrics_monitor = DefaultMetricsMonitor()
        self.stop_callback.reset()

        set_random_seed(self._cfg.train.seed)
        train_subset = dataset.get_subset(Subset.TRAINING)
        val_subset = dataset.get_subset(Subset.VALIDATION)
        self._cfg.custom_datasets.roots = [OTEClassificationDataset(train_subset, self._labels, self._multilabel,
                                                                    self._hierarchical, self._multihead_class_info,
                                                                    keep_empty_label=self._empty_label in self._labels),
                                           OTEClassificationDataset(val_subset, self._labels, self._multilabel,
                                                                    self._hierarchical, self._multihead_class_info,
                                                                    keep_empty_label=self._empty_label in self._labels)]
        datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self._cfg))

        num_aux_models = len(self._cfg.mutual_learning.aux_configs)

        if self._cfg.use_gpu:
            main_device_ids = list(range(self.num_devices))
            extra_device_ids = [main_device_ids for _ in range(num_aux_models)]
            train_model = DataParallel(train_model, device_ids=main_device_ids,
                                       output_device=0).cuda(main_device_ids[0])
        else:
            extra_device_ids = [None for _ in range(num_aux_models)]

        optimizer = torchreid.optim.build_optimizer(train_model, **optimizer_kwargs(self._cfg))

        if self._cfg.lr_finder.enable:
            scheduler = None
        else:
            scheduler = torchreid.optim.build_lr_scheduler(optimizer, num_iter=datamanager.num_iter,
                                                           **lr_scheduler_kwargs(self._cfg))

        if self._cfg.lr_finder.enable:
            _, train_model, optimizer, scheduler = \
                        run_lr_finder(self._cfg, datamanager, train_model, optimizer, scheduler, None,
                                      rebuild_model=False, gpu_num=self.num_devices, split_models=False)

        _, final_acc = run_training(self._cfg, datamanager, train_model, optimizer,
                                    scheduler, extra_device_ids, self._cfg.train.lr,
                                    tb_writer=self.metrics_monitor, perf_monitor=time_monitor,
                                    stop_callback=self.stop_callback)

        training_metrics = self._generate_training_metrics_group()

        self.metrics_monitor.close()
        if self.stop_callback.check_stop():
            logger.info('Training cancelled.')
            return

        logger.info("Training finished.")

        best_snap_path = os.path.join(self._scratch_space, 'best.pth')
        if os.path.isfile(best_snap_path):
            load_pretrained_weights(self._model, best_snap_path)

        for filename in os.listdir(self._scratch_space):
            match = re.match(r'best_(aux_model_[0-9]+\.pth)', filename)
            if match:
                aux_model_name = match.group(1)
                best_aux_snap_path = os.path.join(self._scratch_space, filename)
                self._aux_model_snap_paths[aux_model_name] = best_aux_snap_path

        self.save_model(output_model)
        performance = Performance(score=ScoreMetric(value=final_acc, name="accuracy"),
                                  dashboard_metrics=training_metrics)
        logger.info(f'FINAL MODEL PERFORMANCE {performance}')
        output_model.performance = performance
