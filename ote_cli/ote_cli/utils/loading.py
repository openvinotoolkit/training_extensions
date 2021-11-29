"""
Utils for dynamically importing stuff
"""

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

import io
from ote_sdk.entities.model_template import TaskType
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity


def load_model_weights(path):
    """
    Loads binary weights of a model.

        Args:
            path: A path where to load model from.
    """

    with open(path, "rb") as read_file:
        return read_file.read()


def read_label_schema(model_bytes):
    """
    Reads serialized representation from binary snapshot and returns deserialized LabelSchema.
    """
    
    import torch
    return torch.load(io.BytesIO(model_bytes))["label_schema"]
    

def generate_label_schema(dataset, task_type):
    """
    Generates label schema depending on task type.
    """

    if task_type == TaskType.CLASSIFICATION and dataset.is_multilabel():
        not_empty_labels = dataset.get_labels()
        assert len(not_empty_labels) > 1
        label_schema = LabelSchemaEntity()
        empty_label = LabelEntity(name="Empty label", is_empty=True, domain=Domain.CLASSIFICATION)
        empty_group = LabelGroup(name="empty", labels=[empty_label], group_type=LabelGroupType.EMPTY_LABEL)
        single_groups = []
        for label in not_empty_labels:
            single_groups.append(LabelGroup(name=label.name, labels=[label], group_type=LabelGroupType.EXCLUSIVE))
            label_schema.add_group(single_groups[-1])
        label_schema.add_group(empty_group, exclusive_with=single_groups)
        return label_schema

    return LabelSchemaEntity.from_labels(dataset.get_labels())

