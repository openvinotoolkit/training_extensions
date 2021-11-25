# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.

import abc
import multiprocessing
import queue
import sys
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, List, NamedTuple, Optional, Tuple, Union

import cv2

from natsort import natsorted

import numpy as np


class MediaType(Enum):
    image = 1
    video = 2
    camera = 3


class MediaExtensions(NamedTuple):
    image: Tuple[str, ...]
    video: Tuple[str, ...]


MEDIA_EXTENSIONS = MediaExtensions(
    image=(".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"),
    video=(".avi", ".mp4"),
)


def get_media_type(path: Optional[Union[str, Path]]) -> MediaType:
    """
    Get Media Type from the input path.
    :param path: Path to file or directory.
                 Could be None, which implies camera media type.
    """
    if isinstance(path, str):
        path = Path(path)

    media_type: MediaType

    if path is None:
        media_type = MediaType.camera

    elif path.is_dir():
        if _get_filenames(path, MediaType.image):
            media_type = MediaType.image

    elif path.is_file():
        if _is_file_with_supported_extensions(path, _get_extensions(MediaType.image)):
            media_type = MediaType.image
        elif _is_file_with_supported_extensions(path, _get_extensions(MediaType.video)):
            media_type = MediaType.video
        else:
            raise ValueError("File extension not supported.")
    else:
        raise ValueError("File or folder does not exist")

    return media_type


def _get_extensions(media_type: MediaType) -> Tuple[str, ...]:
    """
    Get extensions of the input media type.
    :param media_type: Type of the media. Either image or video.
    :return: Supported extensions for the corresponding media type.

    :example:

        >>> _get_extensions(media_type=MediaType.image)
        ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        >>> _get_extensions(media_type=MediaType.video)
        ('.avi', '.mp4')

    """
    return getattr(MEDIA_EXTENSIONS, media_type.name)


def _is_file_with_supported_extensions(path: Path, extensions: Tuple[str, ...]) -> bool:
    """
    Check if the file is supported for the media type
    :param path: File path to check
    :param extensions: Supported extensions for the media type

    :example:

        >>> from pathlib import Path
        >>> path = Path("./demo.mp4")
        >>> extensions = _get_extensions(media_type=MediaType.video)
        >>> _is_file_with_supported_extensions(path, extensions)
        True

        >>> path = Path("demo.jpg")
        >>> extensions = _get_extensions(media_type=MediaType.image)
        >>> _is_file_with_supported_extensions(path, extensions)
        True

        >>> path = Path("demo.mp3")
        >>> extensions = _get_extensions(media_type=MediaType.image)
        >>> _is_file_with_supported_extensions(path, extensions)
        False

    """
    return path.suffix.lower() in extensions


def _get_filenames(path: Union[str, Path], media_type: MediaType) -> List[str]:
    """
    Get filenames from a directory or a path to a file.
    :param path: Path to the file or to the location that contains files.
    :param media_type: Type of the media (image or video)

    :example:
        >>> path = "../images"
        >>> _get_filenames(path, media_type=MediaType.image)
        ['images/4.jpeg', 'images/1.jpeg', 'images/5.jpeg', 'images/3.jpeg', 'images/2.jpeg']

    """
    extensions = _get_extensions(media_type)
    filenames: List[str] = []

    if media_type == MediaType.camera:
        raise ValueError(
            "Cannot get filenames for camera. Only image and video files are supported."
        )

    if isinstance(path, str):
        path = Path(path)

    if path.is_file():
        if _is_file_with_supported_extensions(path, extensions):
            filenames = [path.as_posix()]
        else:
            raise ValueError("Extension not supported for media type")

    if path.is_dir():
        for filename in path.rglob("*"):
            if _is_file_with_supported_extensions(filename, extensions):
                filenames.append(filename.as_posix())

    filenames = natsorted(filenames)  # type: ignore[assignment]

    if len(filenames) == 0:
        raise FileNotFoundError(f"No {media_type.name} file found in {path}!")

    return filenames


def _read_video_stream(stream: cv2.VideoCapture) -> Iterator[np.ndarray]:
    """
    Read video and yield the frame.
    :param stream: Video stream captured via OpenCV's VideoCapture
    :return: Individual frame
    """
    while True:
        frame_available, frame = stream.read()
        if not frame_available:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame

    stream.release()


class BaseStreamer(metaclass=abc.ABCMeta):
    """
    Base Streamer interface to implement Image, Video and Camera streamers.
    """

    @abc.abstractmethod
    def get_stream(self, stream_input):
        """
        Get the streamer object, depending on the media type.
        :param stream_input: Path to the stream or
                             camera device index  in case to capture from camera.
        :return: Streamer object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate through the streamer object that is a Python Generator object.
        :return: Yield the image or video frame.
        """
        raise NotImplementedError


def _process_run(streamer: BaseStreamer, buffer: multiprocessing.Queue):
    """
    Private function that is run by the thread.

    Waits for the buffer to gain space for timeout seconds while it is full.
    If no space was available within this time the function will exit

    :param streamer: The streamer to retrieve frames from
    :param buffer: The buffer to place the retrieved frames in
    """
    for frame in streamer:
        buffer.put(frame)


class ThreadedStreamer(BaseStreamer):
    """
    Runs a BaseStreamer on a seperate thread.

    :param streamer: The streamer to run on a thread
    :param buffer_size: Number of frame to buffer internally

    :example:

        >>> streamer = VideoStreamer(path="../demo.mp4")
        >>> threaded_streamer = ThreadedStreamer(streamer)
        ... for frame in threaded_streamer:
        ...    pass
    """

    def __init__(self, streamer: BaseStreamer, buffer_size: int = 2):
        self.buffer_size = buffer_size
        self.streamer = streamer

    def get_stream(self, _=None) -> BaseStreamer:
        return self.streamer

    def __iter__(self) -> Iterator[np.ndarray]:
        buffer: multiprocessing.Queue = multiprocessing.Queue(maxsize=self.buffer_size)
        process = multiprocessing.Process(
            target=_process_run, args=(self.get_stream(), buffer)
        )
        # Make thread a daemon so that it will exit when the main program exits as well
        process.daemon = True
        process.start()

        try:
            while process.is_alive() or not buffer.empty():
                try:
                    yield buffer.get(timeout=0.1)
                except queue.Empty:
                    pass
        except GeneratorExit:
            process.terminate()
        finally:
            process.join(timeout=0.1)
            # The kill() function is only available in Python 3.7.
            # Skip it if running an older Python version.
            if sys.version_info >= (3, 7) and process.exitcode is None:
                process.kill()


class VideoStreamer(BaseStreamer):
    """
    Video Streamer
    :param path: Path to the video file or directory.

    :example:

        >>> streamer = VideoStreamer(path="../demo.mp4")
        ... for frame in streamer:
        ...    pass
    """

    def __init__(self, path: str) -> None:
        self.media_type = MediaType.video
        self.filenames = _get_filenames(path, media_type=MediaType.video)

    def get_stream(self, stream_input: str) -> cv2.VideoCapture:
        return cv2.VideoCapture(stream_input)

    def __iter__(self) -> Iterator[np.ndarray]:
        for filename in self.filenames:
            stream = self.get_stream(stream_input=filename)
            yield from _read_video_stream(stream)


class CameraStreamer(BaseStreamer):
    """
    Stream video frames from camera
    :param camera_device: Camera device index e.g, 0, 1

    :example:

        >>> streamer = CameraStreamer(camera_device=0)
        ... for frame in streamer:
        ...     cv2.imshow("Window", frame)
        ...     if ord("q") == cv2.waitKey(1):
        ...         break
    """

    def __init__(self, camera_device: Optional[int] = None):
        self.media_type = MediaType.camera
        self.camera_device = 0 if camera_device is None else camera_device

    def get_stream(self, stream_input: int):
        return cv2.VideoCapture(stream_input)

    def __iter__(self) -> Iterator[np.ndarray]:
        stream = self.get_stream(stream_input=self.camera_device)
        yield from _read_video_stream(stream)


class ImageStreamer(BaseStreamer):
    """
    Stream from image file or directory.
    :param path: Path to an image or directory.

    :example:

        >>> streamer = ImageStreamer(path="../images")
        ... for frame in streamer:
        ...     cv2.imshow("Window", frame)
        ...     cv2.waitKey(0)
    """

    def __init__(self, path: str) -> None:
        self.media_type = MediaType.image
        self.filenames = _get_filenames(path=path, media_type=MediaType.image)

    @staticmethod
    def get_stream(stream_input: str) -> Iterable[np.ndarray]:
        image = cv2.imread(stream_input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yield image

    def __iter__(self) -> Iterator[np.ndarray]:
        for filename in self.filenames:
            yield from self.get_stream(stream_input=filename)


def get_streamer(
    path: Optional[str] = None,
    camera_device: Optional[int] = None,
    threaded: bool = False,
) -> BaseStreamer:
    """
    Get streamer object based on the file path or camera device index provided.
    :param path: Path to file or directory.
    :param camera_device: Camera device index.
    :param threaded: Threaded streaming option
    """
    if path is not None and camera_device is not None:
        raise ValueError(
            "Both path and camera device is provided. Choose either camera or path to a image/video file."
        )

    media_type = get_media_type(path)

    streamer: BaseStreamer

    if path is not None and media_type == MediaType.image:
        streamer = ImageStreamer(path)

    elif path is not None and media_type == MediaType.video:
        streamer = VideoStreamer(path)

    elif media_type == MediaType.camera:
        if camera_device is None:
            camera_device = 0
        streamer = CameraStreamer(camera_device)

    else:
        raise ValueError("Unknown media type")

    if threaded:
        streamer = ThreadedStreamer(streamer)

    return streamer
