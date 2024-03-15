"""Collection of helper functions for unit tests of otx.algorithms.action."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict, List

import numpy as np

from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.metadata import MetadataItemEntity, VideoMetadata
from otx.api.entities.model_template import ModelTemplate
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment


class MockImage(Image):
    """Mock class for Image entity."""

    def __init__(self, file_path):
        self.__file_path = file_path
        self.__data = np.ndarray((256, 256, 3))
        super().__init__(self.__data)


class MockPipeline:
    """Mock class for data pipeline.

    It returns its inputs.
    """

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return results


def generate_labels(length: int, domain: Domain) -> List[LabelEntity]:
    """Generate list of LabelEntity given length and domain."""

    output: List[LabelEntity] = []
    for i in range(length):
        output.append(LabelEntity(name=f"{i + 1}", domain=domain, id=ID(i + 1)))
    return output


def generate_action_cls_otx_dataset(video_len: int, frame_len: int, labels: List[LabelEntity]) -> DatasetEntity:
    """Generate otx_dataset for action classification task."""

    items: List[DatasetItemEntity] = []
    for video_id in range(video_len):
        if video_id > 1:
            subset = Subset.VALIDATION
        else:
            subset = Subset.TRAINING
        for frame_idx in range(frame_len):
            item = DatasetItemEntity(
                media=MockImage(f"{video_id}_{frame_idx}.png"),
                annotation_scene=AnnotationSceneEntity(
                    annotations=[Annotation(Rectangle.generate_full_box(), [ScoredLabel(labels[video_id])])],
                    kind=AnnotationSceneKind.ANNOTATION,
                ),
                metadata=[MetadataItemEntity(data=VideoMetadata(video_id, frame_idx, is_empty_frame=False))],
                subset=subset,
            )
            items.append(item)
    dataset = DatasetEntity(items=items)
    return dataset


def generate_action_det_otx_dataset(video_len: int, frame_len: int, labels: List[LabelEntity]) -> DatasetEntity:
    """Generate otx_dataset for action detection task."""

    items: List[DatasetItemEntity] = []
    proposals: Dict[str, List[float]] = {}
    for video_id in range(video_len):
        if video_id > 1:
            subset = Subset.VALIDATION
        else:
            subset = Subset.TRAINING
        for frame_idx in range(frame_len):
            if frame_idx % 2 == 0:
                item = DatasetItemEntity(
                    media=MockImage(f"{video_id}_{frame_idx}.png"),
                    annotation_scene=AnnotationSceneEntity(
                        annotations=[Annotation(Rectangle.generate_full_box(), [ScoredLabel(labels[video_id])])],
                        kind=AnnotationSceneKind.ANNOTATION,
                    ),
                    metadata=[MetadataItemEntity(data=VideoMetadata(str(video_id), frame_idx, is_empty_frame=False))],
                    subset=subset,
                )
                proposals[f"{video_id},{frame_idx:04d}"] = [0.0, 0.0, 1.0, 1.0]
            else:
                item = DatasetItemEntity(
                    media=MockImage(f"{video_id}_{frame_idx}.png"),
                    annotation_scene=AnnotationSceneEntity(
                        annotations=[Annotation(Rectangle.generate_full_box(), [ScoredLabel(labels[video_id])])],
                        kind=AnnotationSceneKind.ANNOTATION,
                    ),
                    metadata=[MetadataItemEntity(data=VideoMetadata(str(video_id), frame_idx, is_empty_frame=True))],
                    subset=subset,
                )
            items.append(item)
    dataset = DatasetEntity(items=items)
    return dataset, proposals


class MockModelTemplate(ModelTemplate):
    """Mock class for ModelTemplate."""

    def __post_init__(self):
        pass


def return_args(*args, **kwargs):
    """This function returns its args."""
    return args, kwargs


def return_inputs(inputs):
    """This function returns its input."""
    return inputs


def init_environment(params, model_template, label_schema):
    """Initialize environment."""
    environment = TaskEnvironment(
        model=None,
        hyper_parameters=params,
        label_schema=label_schema,
        model_template=model_template,
    )
    return environment
