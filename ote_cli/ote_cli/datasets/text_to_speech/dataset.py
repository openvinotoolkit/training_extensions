"""
Module contains ImageClassificationDataset
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


import numpy as np
from typing import Dict, List, Optional, Tuple

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
from ote_sdk.entities.media import IMediaEntity


def parse_ljspeech_dataset(csv_path, data_root):
    dataset = []
    with open(csv_path, 'r', encoding="utf8", errors="ignore") as f:
        for line in f:
            datas = line.strip().split('|')
            audio_path = f'{data_root}/wavs/{datas[0]}.wav'
            text = datas[-1]
            dataset.append({"audio_path": audio_path,
                            "text": text})
    return dataset


class Text(IMediaEntity):
    """
    Represents a mel spectrogram.

    The audio must be instantiated with either a NumPy array containing the image data
    or a path to an image file.

    Arguments:
        text (str): text.
        metadata (str): metadata.
    """
    # pylint: disable=too-many-arguments, redefined-builtin
    def __init__(
        self,
        text: Optional[str] = None,
        metadata: Optional[str] = None
    ):
        self.__text = text
        self.__metadata = metadata

    def __str__(self):
        return (
            f"{self.__class__.__name__}"
            f"({self.__text})"
        )

    def __get_size(self) -> int:
        """
        Returns text size.

        Returns:
            text size (int).
        """
        return len(self.__text)

    @property
    def numpy(self) -> Tuple[np.ndarray, int]:
        """
        NumPy representation of the text.

        Returns:
            text (np.array): text.
        """

        return np.array(self.__text)

    @property
    def metadata(self) -> str:
        """
        """
        return self.__metadata

    @numpy.setter
    def numpy(self, text: str):
        self.__text = text
        self.__size = self.__get_size()

    @property
    def size(self) -> int:
        """
        Returns the size of the audio.
        """
        if self.__size is None:
            self.__size = self.__get_size()
        return self.__size


class OTETextToSpeechDataset(DatasetEntity):
    """Dataloader for LJSpeech To Text Task.

    Args:
        csv_path (str): csv file in ljspeech format.
        data_path (str): path to media files.
    """

    def __init__(
        self,
        train_subset: Optional[Dict[str, str]] = None,
        val_subset: Optional[Dict[str, str]] = None,
        test_subset: Optional[Dict[str, str]] = None,
    ):

        items: List[DatasetItemEntity] = []

        if train_subset is not None:
            items.extend(
                self.get_dataset_items(
                    csv_path=train_subset["ann_file"],
                    data_path=train_subset["data_root"],
                    subset=Subset.TRAINING
                )
            )

        if val_subset is not None:
            items.extend(
                self.get_dataset_items(
                    csv_path=val_subset["ann_file"],
                    data_path=val_subset["data_root"],
                    subset=Subset.VALIDATION
                )
            )

        if test_subset is not None:
            items.extend(
                self.get_dataset_items(
                    csv_path=test_subset["ann_file"],
                    data_path=test_subset["data_root"],
                    subset=Subset.TESTING
                )
            )

        super().__init__(items=items)

    @staticmethod
    def get_dataset_items(
        csv_path: str, data_path: str, subset: Subset
    ) -> List[DatasetItemEntity]:
        """Loads dataset based on the csv file.

        Args:
            csv_path (str): csv file in ljspeech format.
            data_path (str): path to media files.
            subset (Subset): Subset of the dataset (train, val, test).

        Returns:
            List[DatasetItemEntity]: List containing subset dataset.
        """
        dataset_items = []

        data = parse_ljspeech_dataset(csv_path, data_path)
        for sample in data:
            input_data = Text(text=sample["text"], metadata=sample["audio_path"])
            label = LabelEntity(name=sample["audio_path"], domain=Domain.TEXT_TO_SPEECH)
            labels = [ScoredLabel(label)]
            annotations = [Annotation(shape=Rectangle(x1=0, y1=0, x2=1, y2=1), labels=labels)]
            annotation_scene = AnnotationSceneEntity(
                annotations=annotations, kind=AnnotationSceneKind.ANNOTATION
            )
            dataset_item = DatasetItemEntity(
                media=input_data, annotation_scene=annotation_scene, subset=subset
            )
            # Add to dataset items
            dataset_items.append(dataset_item)

        return dataset_items
