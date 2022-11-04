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
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from ote_sdk.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from ote_sdk.entities.datasets import DatasetEntity, DatasetItemEntity
from ote_sdk.entities.image import Image
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

    model_adapter_keys = (
        "confidence_threshold",
        "image_threshold",
        "pixel_threshold",
        "min",
        "max",
        "config.json",
    )

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

            model_path = os.path.join(temp_dir, "model", "model")
            model_adapters = {
                "openvino.xml": ModelAdapter(read_binary(model_path + ".xml")),
                "openvino.bin": ModelAdapter(read_binary(model_path + ".bin")),
            }

            config_path = os.path.join(temp_dir, "model", "config.json")
            with open(config_path, encoding="UTF-8") as f:
                model_parameters = json.load(f)["model_parameters"]
            model_adapters["config.json"] = ModelAdapter(read_binary(config_path))

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
            with read_zip_file.open(os.path.join("model", "config.json")) as read_file:
                serialized_label_schema = json.load(read_file)["model_parameters"][
                    "labels"
                ]
    return LabelSchemaMapper().backward(serialized_label_schema)


def generate_label_schema(dataset, task_type):
    """
    Generates label schema depending on task type.
    """
    if task_type == TaskType.CLASSIFICATION:
        not_empty_labels = dataset.get_labels()
        assert len(not_empty_labels) > 1
        label_schema = LabelSchemaEntity()
        if dataset.is_multilabel():
            emptylabel = LabelEntity(
                name="Empty label", is_empty=True, domain=Domain.CLASSIFICATION
            )
            empty_group = LabelGroup(
                name="empty", labels=[emptylabel], group_type=LabelGroupType.EMPTY_LABEL
            )
            # pylint: disable=unnecessary-comprehension
            key = [i for i in dataset.annotations.keys()][0]
            for group in dataset.annotations[key][2]:
                group_labels = []
                for label_name in group:
                    group_labels.append(dataset.label_name_to_project_label(label_name))
                labels = group_labels if dataset.is_multilabel() else group_labels[1:]
                label_schema.add_group(
                    LabelGroup(
                        name=group_labels[0].name,
                        labels=labels,
                        group_type=LabelGroupType.EXCLUSIVE,
                    )
                )
            label_schema.add_group(empty_group)
        elif dataset.is_multihead():
            emptylabel = LabelEntity(
                name="Empty label", is_empty=True, domain=Domain.CLASSIFICATION
            )
            empty_group = LabelGroup(
                name="empty", labels=[emptylabel], group_type=LabelGroupType.EMPTY_LABEL
            )
            key = [i for i in dataset.annotations.keys()][0]
            hierarchy_info = dataset.annotations[key][2]

            def add_subtask_labels(dataset, info):
                group = info["group"]
                labels = info["labels"]
                task_type = info["task_type"]
                if (
                    task_type == "single-label"
                ):  # add one label group includes all labels
                    label_entity_list = [
                        dataset.label_name_to_project_label(lbl) for lbl in labels
                    ]
                    label_group = LabelGroup(
                        name=group,
                        labels=label_entity_list,
                        group_type=LabelGroupType.EXCLUSIVE,
                    )
                    label_schema.add_group(label_group)
                elif task_type == "multi-label":  # add label group for each label
                    for label in labels:
                        label_entity_list = [dataset.label_name_to_project_label(label)]
                        label_group = LabelGroup(
                            name=f"{group}____{label}",
                            labels=label_entity_list,
                            group_type=LabelGroupType.EXCLUSIVE,
                        )
                        label_schema.add_group(label_group)
                if "subtask" in info:
                    subtask = info["subtask"]
                    for stask in subtask:  # if has several subtasks
                        add_subtask_labels(dataset, stask)

            for info in hierarchy_info:
                if info[
                    "task_type"
                ] == "multi-label" and emptylabel not in label_schema.get_labels(
                    include_empty=True
                ):
                    label_schema.add_group(empty_group)
                add_subtask_labels(dataset, info)
        else:
            main_group = LabelGroup(
                name="labels",
                labels=dataset.project_labels,
                group_type=LabelGroupType.EXCLUSIVE,
            )
            label_schema.add_group(main_group)
        return label_schema

    return LabelSchemaEntity.from_labels(dataset.get_labels())


def get_image_files(root_dir):
    """
    recursively get all image file paths from given root_dir
    """
    img_data_formats = [
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".png",
    ]
    img_data_formats.extend([x.upper() for x in img_data_formats])
    img_data_formats = tuple(img_data_formats)

    # single image path
    if root_dir.endswith(img_data_formats):
        return [os.path.dirname(root_dir), os.path.basename(root_dir)]

    img_files = []
    for root, _, _ in os.walk(root_dir):
        for format_ in img_data_formats:
            img_files.extend(
                [(root, file.name) for file in Path(root).glob(f"*{format_}")]
            )
    return img_files if img_files else None


def save_saliency_output(
    img: np.array,
    saliency_map: np.array,
    save_dir: str,
    fname: str,
    weight: float = 0.5,
) -> None:
    """
    receive img and saliency map, then convert to colormap image and save images
    """
    heatmap = cv2.applyColorMap(np.uint8(saliency_map), cv2.COLORMAP_JET)
    overlay = (1 - weight) * img + weight * heatmap
    overlay /= np.max(overlay)
    overlay = np.uint8(255 * overlay)
    cv2.imwrite(f"{os.path.join(save_dir, fname)}_saliency_map.jpg", heatmap)
    cv2.imwrite(f"{os.path.join(save_dir, fname)}_overlay_img.jpg", overlay)


def get_explain_dataset_from_filelist(image_files: list):
    """
    get explain dataset with empty annotation
    """
    empty_annotation = AnnotationSceneEntity(
        annotations=[], kind=AnnotationSceneKind.PREDICTION
    )
    items = []
    for root_dir, filename in image_files:
        frame = cv2.imread(os.path.join(root_dir, filename))
        item = DatasetItemEntity(
            media=Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            annotation_scene=empty_annotation,
        )
        items.append(item)
    explain_dataset = DatasetEntity(items=items)
    return explain_dataset
