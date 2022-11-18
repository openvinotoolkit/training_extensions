"""NNCF Task for OTX Classification."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from otx.algorithms.classification.adapters.deep_object_reid.tasks import (
    ClassificationNNCFTask,
)
from otx.api.entities.task_environment import TaskEnvironment


# pylint: disable=too-many-ancestors
class OTXClassificationNNCFTask(ClassificationNNCFTask):
    """Task for compressing classification models using NNCF."""

    def __init__(self, task_environment: TaskEnvironment):  # pylint: disable=useless-parent-delegation
        super().__init__(task_environment)
