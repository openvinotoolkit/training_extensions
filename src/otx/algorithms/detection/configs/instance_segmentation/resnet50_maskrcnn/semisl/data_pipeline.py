"""Data Pipeline of Mask-RCNN ResNet50 model for Semi-Supervised Learning Instance Segmentation Task."""

# Copyright (C) 2023 Intel Corporation
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

# pylint: disable=invalid-name

# This is from otx/mpa/recipes/stages/_base_/data/pipelines/ubt.py
# This could be needed sync with incr-learning's data pipeline

_base_ = ["../../../base/data/base_semisl_is_res_data_pipeline.py"]
