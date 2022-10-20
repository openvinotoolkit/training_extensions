"""Collection of utils for task implementation in Classification Task."""

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


from .cli import ClassificationDatasetAdapter
from .label import generate_label_schema, get_multihead_class_info, get_task_class

__all__ = ["ClassificationDatasetAdapter", "generate_label_schema", "get_multihead_class_info", "get_task_class"]
