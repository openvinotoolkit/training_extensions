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

import time
import colorsys
import importlib
import random
from typing import Callable, Optional, Sequence, Union

import numpy as np
import yaml
from ote_sdk.entities.color import Color
from ote_sdk.entities.id import ID
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from ote_sdk.entities.train_parameters import UpdateProgressCallback
from ote_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from ote_sdk.utils.argument_checks import (
    YamlFilePathCheck,
    check_input_parameters_type,
)


class ColorPalette:
    @check_input_parameters_type()
    def __init__(self, n: int, rng: Optional[random.Random] = None):
        assert n > 0

        if rng is None:
            rng = random.Random(0xACE)

        candidates_num = 100
        hsv_colors = [(1.0, 1.0, 1.0)]
        for _ in range(1, n):
            colors_candidates = [(rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0))
                                 for _ in range(candidates_num)]
            min_distances = [self._min_distance(hsv_colors, c) for c in colors_candidates]
            arg_max = np.argmax(min_distances)
            hsv_colors.append(colors_candidates[arg_max])

        self.palette = [Color(*self._hsv2rgb(*hsv)) for hsv in hsv_colors]

    @staticmethod
    def _dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    @classmethod
    def _min_distance(cls, colors_set, color_candidate):
        distances = [cls._dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    @staticmethod
    def _hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    @check_input_parameters_type()
    def __getitem__(self, n: int):
        return self.palette[n % len(self.palette)]

    def __len__(self):
        return len(self.palette)


@check_input_parameters_type()
def generate_label_schema(label_names: Sequence[str], label_domain: Domain = Domain.DETECTION):
    colors = ColorPalette(len(label_names)) if len(label_names) > 0 else []
    not_empty_labels = [LabelEntity(name=name, color=colors[i], domain=label_domain, id=ID(f"{i:08}")) for i, name in
                        enumerate(label_names)]
    emptylabel = LabelEntity(name=f"Empty label", color=Color(42, 43, 46),
                             is_empty=True, domain=label_domain, id=ID(f"{len(not_empty_labels):08}"))

    label_schema = LabelSchemaEntity()
    exclusive_group = LabelGroup(name="labels", labels=not_empty_labels, group_type=LabelGroupType.EXCLUSIVE)
    empty_group = LabelGroup(name="empty", labels=[emptylabel], group_type=LabelGroupType.EMPTY_LABEL)
    label_schema.add_group(exclusive_group)
    label_schema.add_group(empty_group)
    return label_schema


@check_input_parameters_type({"path": YamlFilePathCheck})
def load_template(path):
    with open(path) as f:
        template = yaml.safe_load(f)
    return template


@check_input_parameters_type()
def get_task_class(path: str):
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class TrainingProgressCallback(TimeMonitorCallback):
    def __init__(self, update_progress_callback: UpdateProgressCallback):
        super().__init__(0, 0, 0, 0, update_progress_callback=update_progress_callback)

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        self.update_progress_callback(self.get_progress())

    def on_epoch_end(self, epoch, logs=None):
        self.past_epoch_duration.append(time.time() - self.start_epoch_time)
        self._calculate_average_epoch()
        score = None
        if hasattr(self.update_progress_callback, 'metric') and isinstance(logs, dict):
            score = logs.get(self.update_progress_callback.metric, None)
            score = float(score) if score is not None else None
        self.update_progress_callback(self.get_progress(), score=score)


class InferenceProgressCallback(TimeMonitorCallback):
    def __init__(self, num_test_steps, update_progress_callback: Callable[[int], None]):
        super().__init__(
            num_epoch=0,
            num_train_steps=0,
            num_val_steps=0,
            num_test_steps=num_test_steps,
            update_progress_callback=update_progress_callback)

    def on_test_batch_end(self, batch=None, logs=None):
        super().on_test_batch_end(batch, logs)
        self.update_progress_callback(int(self.get_progress()))


class OptimizationProgressCallback(TrainingProgressCallback):
    """ Progress callback used for optimization using NNCF
        There are three stages to the progress bar:
           - 5 % model is loaded
           - 10 % compressed model is initialized
           - 10-100 % compressed model is being fine-tuned
    """
    def __init__(self, update_progress_callback: UpdateProgressCallback, loading_stage_progress_percentage: int = 5,
                 initialization_stage_progress_percentage: int = 5):
        super().__init__(update_progress_callback=update_progress_callback)
        if loading_stage_progress_percentage + initialization_stage_progress_percentage >= 100:
            raise RuntimeError('Total optimization progress percentage is more than 100%')

        self.loading_stage_progress_percentage = loading_stage_progress_percentage
        self.initialization_stage_progress_percentage = initialization_stage_progress_percentage

        # set loading_stage_progress_percentage from the start as the model is already loaded at this point
        self.update_progress_callback(loading_stage_progress_percentage)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        # Callback initialization takes place here after OTEProgressHook.before_run() is called
        train_percentage = 100 - self.loading_stage_progress_percentage - self.initialization_stage_progress_percentage
        loading_stage_steps = self.total_steps * self.loading_stage_progress_percentage / train_percentage
        initialization_stage_steps = self.total_steps * self.initialization_stage_progress_percentage / train_percentage
        self.total_steps += loading_stage_steps + initialization_stage_steps

        self.current_step = loading_stage_steps + initialization_stage_steps
        self.update_progress_callback(self.get_progress())

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.update_progress_callback(self.get_progress(), score=logs)

    def on_initialization_end(self):
        self.update_progress_callback(self.loading_stage_progress_percentage +
                                      self.initialization_stage_progress_percentage)
