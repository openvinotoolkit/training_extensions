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
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.model_template import TaskType
from ote_sdk.serialization.label_mapper import LabelSchemaMapper
from ote_sdk.usecases.adapters.model_adapter import ModelAdapter
from ote_sdk.usecases.exportable_code.demo.demo_package import create_model
from ote_sdk.usecases.exportable_code.prediction_to_annotation_converter import (
    create_converter,
)


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
        for key in ["confidence_threshold", "image_threshold", "min", "max"]:
            full_path = os.path.join(os.path.dirname(path), key)
            if os.path.exists(full_path):
                model_adapters[key] = ModelAdapter(read_binary(full_path))
    else:
        model_adapters = {"weights.pth": ModelAdapter(read_binary(path))}

    model = ModelEntity(
        configuration=model_configuration,
        model_adapters=model_adapters,
        train_dataset=train_dataset,
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
        label_schema.add_group(empty_group, exclusive_with=single_groups)
        return label_schema

    return LabelSchemaEntity.from_labels(dataset.get_labels())


def create_task_from_deployment(openvino_task_class, deployed_code_zip_path):
    """
    Creates a child class of passed 'openvino_task_class', instance of which is initialized by deployment (zip archive).
    """

    class Task(openvino_task_class):
        """A child class of 'openvino_task_class', instance of which is initialized by deployment (zip archive)."""

        class Inferencer:
            """ModelAPI-based OpenVINO inferencer."""

            def __init__(self, model, converter) -> None:
                self.model = model
                self.converter = converter

            def predict(self, frame):
                """Returns predictions made on a given frame."""

                dict_data, input_meta = self.model.preprocess(frame)
                raw_result = self.model.infer_sync(dict_data)
                predictions = self.model.postprocess(raw_result, input_meta)
                annotation_scene = self.converter.convert_to_annotation(
                    predictions, input_meta
                )
                return annotation_scene

        def __init__(self, task_environment) -> None:
            self.task_environment = task_environment
            with tempfile.TemporaryDirectory() as temp_dir:
                with ZipFile(deployed_code_zip_path) as myzip:
                    myzip.extractall(temp_dir)
                with ZipFile(
                    os.path.join(
                        temp_dir, "python", "demo_package-0.0-py3-none-any.whl"
                    )
                ) as myzip:
                    myzip.extractall(temp_dir)

                model_path = Path(os.path.join(temp_dir, "model", "model.xml"))
                config_path = Path(
                    os.path.join(temp_dir, "demo_package", "config.json")
                )

                with open(config_path, encoding="UTF-8") as read_file:
                    parameters = json.load(read_file)
                converter_type = Domain[parameters["converter_type"]]

                self.inferencer = self.Inferencer(
                    create_model(model_path, config_path),
                    create_converter(
                        converter_type, self.task_environment.label_schema
                    ),
                )

        def infer(
            self,
            dataset: DatasetEntity,
            inference_parameters: Optional[InferenceParameters] = None,
        ) -> DatasetEntity:
            """Inference method."""
            if inference_parameters is not None:
                update_progress_callback = inference_parameters.update_progress
            dataset_size = len(dataset)
            for i, dataset_item in enumerate(dataset, 1):
                predicted_scene = self.inferencer.predict(dataset_item.numpy)
                if (
                    self.task_environment.model_template.task_type
                    == TaskType.CLASSIFICATION
                ):
                    dataset_item.append_labels(
                        predicted_scene.annotations[0].get_labels()
                    )
                else:
                    dataset_item.append_annotations(predicted_scene.annotations)
                update_progress_callback(int(i / dataset_size * 100))
            return dataset

    return Task
