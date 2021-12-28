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

import json
import os

from ote_sdk.entities.id import ID
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.model_template import TaskType
from ote_sdk.serialization.label_mapper import LabelSchemaMapper
from ote_sdk.usecases.adapters.model_adapter import ModelAdapter


def save_model_data(model, folder):
    """
    Saves model data to folder. Folder is created if it does not exist.
    """

    os.makedirs(folder, exist_ok=True)
    for filename, model_adapter in model.model_adapters.items():
        with open(os.path.join(folder, filename), "wb") as write_file:
            write_file.write(model_adapter.data)


def read_binary(path):
    """
    Loads binary data stored at path.

        Args:
            path: A path where to load data from.
    """

    with open(path, "rb") as read_file:
        return read_file.read()


def read_model(model_configuration, path, train_dataset):
    """
    Creates ModelEntity based on model_configuration and data stored at path.
    """

    if path.endswith(".bin") or path.endswith(".xml"):
        model_adapters = {
            "openvino.xml": ModelAdapter(read_binary(path[:-4] + ".xml")),
            "openvino.bin": ModelAdapter(read_binary(path[:-4] + ".bin")),
        }
    else:
        model_adapters = {"weights.pth": ModelAdapter(read_binary(path))}

    for key in [
        "confidence_threshold",
        "image_threshold",
        "image_mean",
        "image_std",
        "pixel_mean",
        "pixel_std",
    ]:
        full_path = os.path.join(os.path.dirname(path), key)
        if os.path.exists(full_path):
            model_adapters[key] = ModelAdapter(read_binary(full_path))

    model = ModelEntity(
        configuration=model_configuration,
        model_adapters=model_adapters,
        train_dataset=train_dataset,
    )

    return model


def read_label_schema(path):
    """
    Reads json file and returns deserialized LabelSchema.
    """

    with open(path, encoding="UTF-8") as read_file:
        serialized_label_schema = json.load(read_file)

    return LabelSchemaMapper().backward(serialized_label_schema)


def generate_label_schema(dataset, task_type):
    """
    Generates label schema depending on task type.
    """

    if task_type == TaskType.CLASSIFICATION and dataset.is_multilabel():
        not_empty_labels = dataset.get_labels()
        assert len(not_empty_labels) > 1
        label_schema = LabelSchemaEntity()
        empty_label = LabelEntity(
            name="Empty label", is_empty=True, domain=Domain.CLASSIFICATION
        )
        empty_group = LabelGroup(
            name="empty", labels=[empty_label], group_type=LabelGroupType.EMPTY_LABEL
        )
        single_groups = []
        for label in not_empty_labels:
            single_groups.append(
                LabelGroup(
                    name=label.name, labels=[label], group_type=LabelGroupType.EXCLUSIVE
                )
            )
            label_schema.add_group(single_groups[-1])
        label_schema.add_group(empty_group, exclusive_with=single_groups)
        return label_schema

    if task_type == TaskType.ANOMALY_CLASSIFICATION:
        return LabelSchemaEntity.from_labels(
            [
                LabelEntity(
                    name="Normal", domain=Domain.ANOMALY_CLASSIFICATION, id=ID(0)
                ),
                LabelEntity(
                    name="Anomalous", domain=Domain.ANOMALY_CLASSIFICATION, id=ID(1)
                ),
            ]
        )

    return LabelSchemaEntity.from_labels(dataset.get_labels())
