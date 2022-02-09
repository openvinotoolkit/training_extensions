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
import re
import struct
import tempfile
from io import BytesIO
from zipfile import ZipFile

from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from ote_sdk.entities.model import ModelEntity, ModelOptimizationType
from ote_sdk.entities.model_template import TaskType
from ote_sdk.serialization.label_mapper import LabelSchemaMapper
from ote_sdk.usecases.adapters.model_adapter import ModelAdapter

from ote_cli.utils.nncf import is_checkpoint_nncf


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
    optimization_type = ModelOptimizationType.NONE

    model_adapter_keys = ("confidence_threshold", "image_threshold", "min", "max")

    if path.endswith(".bin") or path.endswith(".xml"):
        # Openvino IR.
        model_adapters = {
            "openvino.xml": ModelAdapter(read_binary(path[:-4] + ".xml")),
            "openvino.bin": ModelAdapter(read_binary(path[:-4] + ".bin")),
        }
        for key in model_adapter_keys:
            full_path = os.path.join(os.path.dirname(path), key)
            if os.path.exists(full_path):
                model_adapters[key] = ModelAdapter(read_binary(full_path))
    elif path.endswith(".pth"):
        # PyTorch
        model_adapters = {"weights.pth": ModelAdapter(read_binary(path))}

        if is_checkpoint_nncf(path):
            optimization_type = ModelOptimizationType.NNCF

        # Weights of auxiliary models
        for key in os.listdir(os.path.dirname(path)):
            if re.match(r"aux_model_[0-9]+\.pth", key):
                full_path = os.path.join(os.path.dirname(path), key)
                model_adapters[key] = ModelAdapter(read_binary(full_path))

    elif path.endswith(".zip"):
        # Deployed code.
        with tempfile.TemporaryDirectory() as temp_dir:
            with ZipFile(path) as myzip:
                myzip.extractall(temp_dir)
            with ZipFile(
                os.path.join(temp_dir, "python", "demo_package-0.0-py3-none-any.whl")
            ) as myzip:
                myzip.extractall(temp_dir)

            model_path = os.path.join(temp_dir, "model", "model")
            model_adapters = {
                "openvino.xml": ModelAdapter(read_binary(model_path + ".xml")),
                "openvino.bin": ModelAdapter(read_binary(model_path + ".bin")),
            }

            config_path = os.path.join(temp_dir, "demo_package", "config.json")
            with open(config_path) as f:
                model_parameters = json.load(f)["model_parameters"]

            for key in model_adapter_keys:
                if key in model_parameters:
                    model_adapters[key] = ModelAdapter(
                        struct.pack("f", model_parameters[key])
                    )
    else:
        raise ValueError(f"Unknown file type: {path}")

    model = ModelEntity(
        configuration=model_configuration,
        model_adapters=model_adapters,
        train_dataset=train_dataset,
        optimization_type=optimization_type,
    )

    return model


def read_label_schema(path):
    """
    Reads serialized LabelSchema and returns deserialized LabelSchema.
    """

    if any(path.endswith(extension) for extension in (".xml", ".bin", ".pth")):
        with open(
            os.path.join(os.path.dirname(path), "label_schema.json"), encoding="UTF-8"
        ) as read_file:
            serialized_label_schema = json.load(read_file)
    elif path.endswith(".zip"):
        with ZipFile(path) as read_zip_file:
            zfiledata = BytesIO(
                read_zip_file.read(
                    os.path.join("python", "demo_package-0.0-py3-none-any.whl")
                )
            )
            with ZipFile(zfiledata) as read_whl_file:
                with read_whl_file.open(
                    os.path.join("demo_package", "config.json")
                ) as read_file:
                    serialized_label_schema = json.load(read_file)["model_parameters"][
                        "labels"
                    ]
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
        label_schema.add_group(empty_group)
        return label_schema

    return LabelSchemaEntity.from_labels(dataset.get_labels())
