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

from .text_mel_dataset import parse_ljspeech_dataset



class Text(IMediaEntity):
    """
    Represents a mel spectrogram.

    The audio must be instantiated with either a NumPy array containing the image data
    or a path to an image file.

    Arguments:
        data (np.ndarray): NumPy data.
        sampling_rate (int): Sampling rate.
        file_path (str): Path to audio file in .wav format.
    """
    # pylint: disable=too-many-arguments, redefined-builtin
    def __init__(
        self,
        text: Optional[str] = None,
    ):
        self.__text = text

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
    """Dataloader for Speech To Text Task.

    Example:
        >>> train_subset = ["data/train/audio_0001/", "data/train/audio_0002/"]
        >>> val_subset = ["data/val/audio_0001/", "data/val/audio_0002/"]
        >>> training_dataset = SpeechToTextDataset(
                train_subset=train_subset, val_subset=val_subset
            )
        >>> test_subset = ["data/test/audio_0001/", "data/test/audio_0002/"]
        >>> testing_dataset = SpeechToTextDataset(test_subset=test_subset)

    Args:
        train_subset (Optional[Dict[str, str]], optional): Path to annotation
            and dataset used for training. Defaults to None.
        val_subset (Optional[Dict[str, str]], optional): Path to annotation
            and dataset used for validation. Defaults to None.
        test_subset (Optional[Dict[str, str]], optional): Path to annotation
            and dataset used for testing. Defaults to None.
    """

    def __init__(
        self,
        train_subset: Optional[List[str]] = None,
        val_subset: Optional[List[str]] = None,
        test_subset: Optional[List[str]] = None,
    ):

        items: List[DatasetItemEntity] = []

        if train_subset is not None:
            items.extend(
                self.get_dataset_items(
                    data_path=train_subset,
                    subset=Subset.TRAINING,
                )
            )

        if val_subset is not None:
            items.extend(
                self.get_dataset_items(
                    data_path=val_subset,
                    subset=Subset.VALIDATION,
                )
            )

        if test_subset is not None:
            items.extend(
                self.get_dataset_items(
                    data_path=train_subset,
                    subset=Subset.TESTING,
                )
            )

        super().__init__(items=items)

    def get_dataset_items(
        self, csv_path: str, data_path: str, subset: Subset
    ) -> List[DatasetItemEntity]:
        """Loads dataset based on the list or folders.

        Args:
            csv_path (List[str]): List of folders of datasets in librispeech format.
            subset (Subset): Subset of the dataset.

        Returns:
            List[DatasetItemEntity]: List containing subset dataset.
        """
        dataset_items = []

        data = parse_ljspeech_dataset(csv_path, data_path)
        for sample in data:
            input_data = Text(file_path=sample["text"])
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
