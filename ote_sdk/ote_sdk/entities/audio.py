# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""This module implements the Audio entity."""

from typing import Optional, Tuple

import torchaudio
import numpy as np

from ote_sdk.entities.media import IMediaEntity


def load_audio(file_audio: str) -> Tuple[np.array, int]:
    """
    Load audio file.

    Arguments:
        audio_path (str): Path to audio file in .wav format.

    Returns:
        audio (np.array): Waveform.
        sampling_rate (int): Sampling rate.
    """
    waveform, sampling_rate = torchaudio.load(file_audio)
    return waveform.numpy().flatten(), sampling_rate


class Audio(IMediaEntity):
    """
    Represents an audio.

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
        data: Optional[np.ndarray] = None,
        sampling_rate: Optional[int] = None,
        file_path: Optional[str] = None,
    ):
        if (data is None) == (file_path is None):
            raise ValueError(
                "Either path to image file or audio data should be provided."
            )
        if (data is not None) == (sampling_rate is None):
            raise ValueError(
                "Audio data should be provided with sampling_rate."
            )

        self.__data: Optional[np.ndarray] = None if data is None else data.flatten()
        self.__file_path: Optional[str] = file_path
        self.__size: Optional[int] = None
        self.__sampling_rate: Optional[int] = sampling_rate

    def __str__(self):
        return (
            f"{self.__class__.__name__}"
            f"({self.__file_path if self.__data is None else 'with data'}, "
            f"size={self.size})"
        )

    def __get_size(self) -> int:
        """
        Returns audio size.

        Returns:
            audio_size (int): Audio size as a 'size'.
        """
        if self.__data is not None:
            return self.__data.shape[0]
        audio, sampling_rate = load_audio(self.__file_path)
        return audio.shape[0]

    def __get_sampling_rate(self) -> int:
        """
        Returns sampling rate.

        Returns:
            sampling_rate (int): Audio sampling rate.
        """
        if self.__sampling_rate is not None:
            return self.__sampling_rate
        audio, sampling_rate = load_audio(self.__file_path)
        return sampling_rate

    @property
    def numpy(self) -> Tuple[np.ndarray, int]:
        """
        NumPy representation of the audio.

        Returns:
            audio (np.array): Waveform.
            sampling_rate (int): Audio sampling rate.
        """
        if self.__data is None:
            return load_audio(self.__file_path)
        return self.__data, self.__sampling_rate

    @numpy.setter
    def numpy(self, audio: np.ndarray, sampling_rate: int):
        self.__data = audio.flatten()
        self.__sampling_rate = sampling_rate
        self.__file_path = None
        self.__size = self.__get_size()

    @property
    def size(self) -> int:
        """
        Returns the size of the audio.
        """
        if self.__size is None:
            self.__size = self.__get_size()
        return self.__size

    @property
    def sampling_rate(self) -> int:
        """
        Returns the size of the audio.
        """
        if self.__sampling_rate is None:
            self.__sampling_rate = self.__get_sampling_rate()
        return self.__sampling_rate

    @property
    def file_path(self) -> str:
        """
        Returns file path.
        """
        return self.__file_path
