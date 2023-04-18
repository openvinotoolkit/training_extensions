"""Utils for hadnling metadata of segmentation models."""

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


from mmcv.utils import ConfigDict
from otx.api.entities.label_schema import LabelSchemaEntity


def get_seg_model_api_configuration(label_schema: LabelSchemaEntity, hyperparams: ConfigDict):
    """Get ModelAPI config."""
    omz_config = {}
    omz_config[("model_info", "model_type")] = "segmentation"

    omz_config[("model_info", "soft_threshold")] = hyperparams.postprocessing.soft_threshold
    omz_config[("model_info", "blur_strength")] = hyperparams.postprocessing.blur_strength

    all_labels = ""
    for lbl in label_schema.get_labels(include_empty=False):
        all_labels += lbl.name.replace(" ", "_") + " "
    all_labels = all_labels.strip()

    omz_config[("model_info", "labels")] = all_labels

    return omz_config