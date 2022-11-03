"""Collection of utils for task implementation in Action Task."""

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

import os
from typing import List

from mmcv.utils import Config

from otx.algorithms.common.adapters.mmcv.utils import get_data_cfg
from otx.api.entities.label import LabelEntity
from otx.api.utils.argument_checks import check_input_parameters_type


@check_input_parameters_type()
def patch_config(config: Config, base_dir: str, work_dir: str):
    """Patch recipe config suitable to mmaction."""
    # FIXME omnisource is hard coded
    config.omnisource = None
    config.work_dir = work_dir
    patch_data_pipeline(config, base_dir)
    patch_datasets(config)


@check_input_parameters_type()
def patch_data_pipeline(config: Config, base_dir: str):
    """Replace data pipeline to data_pipeline.py if it exist."""
    data_pipeline_path = os.path.join(base_dir, "data_pipelne.py")
    if os.path.exists(data_pipeline_path):
        data_pipeline_cfg = Config.fromfile(data_pipeline_path)
        config.merge_from_dcit(data_pipeline_cfg)


@check_input_parameters_type()
def patch_datasets(config: Config):
    """Patch dataset config suitable to mmaction."""

    def patch_color_conversion(pipeline):
        # Default data format for OTX is RGB, while mmdet uses BGR, so negate the color conversion flag.
        for pipeline_step in pipeline:
            if pipeline_step.type == "Normalize":
                to_rgb = False
                if "to_rgb" in pipeline_step:
                    to_rgb = pipeline_step.to_rgb
                to_rgb = not bool(to_rgb)
                pipeline_step.to_rgb = to_rgb
            elif pipeline_step.type == "MultiScaleFlipAug":
                patch_color_conversion(pipeline_step.transforms)

    # FIXME start_index and modality is hard-coded
    assert "data" in config
    for subset in ("train", "val", "test", "unlabeled"):
        cfg = config.data.get(subset, None)
        if not cfg:
            continue
        cfg.type = "OTXRawframeDataset"
        cfg.start_index = 1
        cfg.modality = "RGB"
        cfg.otx_dataset = None
        cfg.labels = None


@check_input_parameters_type()
def set_data_classes(config: Config, labels: List[LabelEntity]):
    """Setter data classes into config."""
    for subset in ("train", "val", "test"):
        cfg = get_data_cfg(config, subset)
        cfg.labels = labels
        config.data[subset].labels = labels

    # FIXME classification head name is hard-coded
    config.model["cls_head"].num_classes = len(labels)
