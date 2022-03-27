"""
Module contains SemanticSegmentationDataset
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

from pathlib import Path
from typing import Dict, List, Optional

from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.audio import Audio
from speech_to_text.datasets import parse_librispeech_dataset


class SpeechToTextDataset(DatasetEntity):
    """Class for working with file-system based Speech To Text dataset."""

    def __init__(
        self,
        train_subset=None,
        val_subset=None,
        test_subset=None,
    ):

        items: List[DatasetItemEntity] = []

        if train_subset is not None:
            items.extend(
                self.load_dataset_items_librispeech_format(
                    data_path=train_subset["data_root"],
                    subset=Subset.TRAINING,
                )
            )

        if val_subset is not None:
            items.extend(
                self.load_dataset_items_librispeech_format(
                    data_path=val_subset["data_root"],
                    subset=Subset.VALIDATION,
                )
            )

        if test_subset is not None:
            items.extend(
                self.load_dataset_items_librispeech_format(
                    data_path=train_subset["data_root"],
                    subset=Subset.TESTING,
                )
            )

        super().__init__(items=items)

    @staticmethod
    def load_dataset_items_librispeech_format(
        data_path: List[str], subset: Subset
    ) -> List[DatasetItemEntity]:
        """Loads dataset based on the list or folders.

        Args:
            data_path (List[str]): List of folders of datasets in librispeech format.
            subset (Subset): Subset of the dataset.

        Returns:
            List[DatasetItemEntity]: List containing subset dataset.
        """
        dataset_items = []
        for path in data_path:
            data = parse_librispeech_dataset(path)
            for sample in data:
                audio = Audio(file_path=sample["audio_path"])
                label = LabelEntity(name=sample["text"], domain=Domain.CLASSIFICATION)
                labels = [ScoredLabel(label)]
                annotations = [Annotation(shape=Rectangle(x1=0, y1=0, x2=1, y2=1), labels=labels)]
                annotation_scene = AnnotationSceneEntity(
                    annotations=annotations, kind=AnnotationSceneKind.ANNOTATION
                )
                dataset_item = DatasetItemEntity(
                    media=audio, annotation_scene=annotation_scene, subset=subset
                )
                # Add to dataset items
                dataset_items.append(dataset_item)
        return dataset_items
