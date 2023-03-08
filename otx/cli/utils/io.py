"""Utils for model io operations."""

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
from typing import List, Optional, Tuple
from zipfile import ZipFile

import cv2
import numpy as np

from otx.api.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import (
    ModelConfiguration,
    ModelEntity,
    ModelOptimizationType,
)
from otx.api.serialization.label_mapper import LabelSchemaMapper
from otx.api.usecases.adapters.model_adapter import ModelAdapter
from otx.cli.utils.nncf import is_checkpoint_nncf


def save_model_data(model: ModelEntity, folder: str) -> None:
    """Saves model data to folder. Folder is created if it does not exist.

    Args:
        model (ModelEntity): The model to save.
        folder (str): Path to output folder.
    """

    os.makedirs(folder, exist_ok=True)
    for filename, model_adapter in model.model_adapters.items():
        with open(os.path.join(folder, filename), "wb") as write_file:
            write_file.write(model_adapter.data)


def read_binary(path: str) -> bytes:
    """Loads binary data stored at path.

    Args:
        path (str): A path where to load data from.

    Returns:
        bytes: Binary data.
    """

    with open(path, "rb") as read_file:
        return read_file.read()


def read_model(model_configuration: ModelConfiguration, path: str, train_dataset: DatasetEntity) -> ModelEntity:
    """Creates ModelEntity based on model_configuration and data stored at path.

    Args:
        model_configuration (ModelConfiguration): ModelConfiguration object.
        path (str): Path to the model data.
        train_dataset (DatasetEntity): DatasetEntity object.

    Returns:
        ModelEntity: ModelEntity object.
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
            else:
                model_adapters[key] = ModelAdapter(bytes())
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
                    model_adapters[key] = ModelAdapter(struct.pack("f", model_parameters[key]))
    else:
        raise ValueError(f"Unknown file type: {path}")

    model = ModelEntity(
        configuration=model_configuration,
        model_adapters=model_adapters,
        train_dataset=train_dataset,
        optimization_type=optimization_type,
    )

    return model


def read_label_schema(path: str) -> LabelSchemaEntity:
    """Reads serialized LabelSchema and returns deserialized LabelSchema.

    Args:
        path (str): Path to model. It assmues that the `label_schema.json` is at the same location as the model.

    Returns:
        LabelSchemaEntity: Desetialized LabelSchemaEntity.
    """

    if any(path.endswith(extension) for extension in (".xml", ".bin", ".pth")):
        with open(os.path.join(os.path.dirname(path), "label_schema.json"), encoding="UTF-8") as read_file:
            serialized_label_schema = json.load(read_file)
    elif path.endswith(".zip"):
        with ZipFile(path) as read_zip_file:
            with read_zip_file.open(os.path.join("model", "config.json")) as read_file:
                serialized_label_schema = json.load(read_file)["model_parameters"]["labels"]
    return LabelSchemaMapper().backward(serialized_label_schema)


def get_image_files(root_dir: str) -> Optional[List[Tuple[str, str]]]:
    """Recursively get all image file paths from given root_dir."""
    img_data_formats = (
        ".jpg",
        ".JPG",
        ".jpeg",
        ".JPEG",
        ".gif",
        ".GIF",
        ".bmp",
        ".BMP",
        ".tif",
        ".TIF",
        ".tiff",
        ".TIFF",
        ".png",
        ".PNG",
    )
    # single image path
    if root_dir.endswith(img_data_formats):
        return [("./", root_dir)]

    img_files = []
    for root, _, _ in os.walk(root_dir):
        for format_ in img_data_formats:
            img_files.extend([(root, file.name) for file in Path(root).glob(f"*{format_}")])
    return img_files if img_files else None


def save_saliency_output(
    process_saliency_maps: bool,
    img: np.array,
    saliency_map: np.array,
    save_dir: str,
    fname: str,
    weight: float = 0.3,
) -> None:
    """Saves processed saliency map (with image overlay) or raw saliency map."""
    if process_saliency_maps:
        # Saves processed saliency map
        overlay = img * weight + saliency_map * (1 - weight)
        overlay[overlay > 255] = 255
        overlay = overlay.astype(np.uint8)

        cv2.imwrite(f"{os.path.join(save_dir, fname)}_saliency_map.png", saliency_map)
        cv2.imwrite(f"{os.path.join(save_dir, fname)}_overlay_img.png", overlay)
    else:
        # Saves raw, low-resolution saliency map
        cv2.imwrite(f"{os.path.join(save_dir, fname)}_saliency_map.tiff", saliency_map)


def get_explain_dataset_from_filelist(image_files: list):
    """Get explain dataset with empty annotation."""
    empty_annotation = AnnotationSceneEntity(annotations=[], kind=AnnotationSceneKind.PREDICTION)
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
