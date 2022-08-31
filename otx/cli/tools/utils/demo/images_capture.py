"""Images capturing module."""

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

import copy
import os
import sys

import cv2

# Taken from here:
# https://github.com/openvinotoolkit/open_model_zoo/blob/develop/demos/common/python/images_capture.py


class InvalidInput(Exception):
    """Exception for wrong input format."""

    def __init__(self, message):
        super().__init__()
        self.message = message


class OpenError(Exception):
    """Exception for error opening reader."""

    def __init__(self, message):
        super().__init__()
        self.message = message


class ImagesCapture:
    """Images capturing base class."""

    def read(self):
        """Returns captured image."""
        raise NotImplementedError

    def fps(self):
        """Returns a frequency of getting images from source."""
        raise NotImplementedError

    def get_type(self):
        """Returns type of image capture."""
        raise NotImplementedError


class ImreadWrapper(ImagesCapture):
    """Class for reading an image from file."""

    def __init__(self, source, loop):
        self.loop = loop
        if not os.path.isfile(source):
            raise InvalidInput(f"Can't find the image by {source}")
        self.image = cv2.imread(source, cv2.IMREAD_COLOR)
        if self.image is None:
            raise OpenError(f"Can't open the image from {source}")
        self.can_read = True

    def read(self):
        """Returns captured image."""
        if self.loop:
            return copy.deepcopy(self.image)
        if self.can_read:
            self.can_read = False
            return copy.deepcopy(self.image)
        return None

    def fps(self):
        """Returns a frequency of getting images from source."""
        return 1.0

    def get_type(self):
        """Returns type of image capture."""
        return "IMAGE"


class DirReader(ImagesCapture):
    """Class for reading images from directory."""

    def __init__(self, source, loop):
        self.loop = loop
        self.dir = source
        if not os.path.isdir(self.dir):
            raise InvalidInput(f"Can't find the dir by {source}")
        self.names = sorted(os.listdir(self.dir))
        if not self.names:
            raise OpenError(f"The dir {source} is empty")
        self.file_id = 0
        for name in self.names:
            filename = os.path.join(self.dir, name)
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            if image is not None:
                return
        raise OpenError(f"Can't read the first image from {source}")

    def read(self):
        """Returns captured image."""
        while self.file_id < len(self.names):
            filename = os.path.join(self.dir, self.names[self.file_id])
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            self.file_id += 1
            if image is not None:
                return image
        if self.loop:
            self.file_id = 0
            while self.file_id < len(self.names):
                filename = os.path.join(self.dir, self.names[self.file_id])
                image = cv2.imread(filename, cv2.IMREAD_COLOR)
                self.file_id += 1
                if image is not None:
                    return image
        return None

    def fps(self):
        """Returns a frequency of getting images from source."""
        return 1.0

    def get_type(self):
        """Returns type of image capture."""
        return "DIR"


class VideoCapWrapper(ImagesCapture):
    """Class for capturing images from video."""

    def __init__(self, source, loop):
        self.loop = loop
        self.cap = cv2.VideoCapture()
        status = self.cap.open(source)
        if not status:
            raise InvalidInput(f"Can't open the video from {source}")

    def read(self):
        """Returns captured image."""
        status, image = self.cap.read()
        if not status:
            if not self.loop:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            status, image = self.cap.read()
            if not status:
                return None
        return image

    def fps(self):
        """Returns a frequency of getting images from source."""
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_type(self):
        """Returns type of image capture."""
        return "VIDEO"


class CameraCapWrapper(ImagesCapture):
    """Class for capturing images from camera."""

    def __init__(self, source, camera_resolution):
        self.cap = cv2.VideoCapture()
        try:
            status = self.cap.open(int(source))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            if not status:
                raise OpenError(f"Can't open the camera from {source}")
        except ValueError as ex:
            raise InvalidInput(f"Can't find the camera {source}") from ex

    def read(self):
        """Returns captured image."""
        status, image = self.cap.read()
        if not status:
            return None
        return image

    def fps(self):
        """Returns a frequency of getting images from source."""
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_type(self):
        """Returns type of image capture."""
        return "CAMERA"


def open_images_capture(source, loop, camera_resolution=(1280, 720)):
    """Opens images capture."""

    errors = {InvalidInput: [], OpenError: []}
    for reader in (ImreadWrapper, DirReader, VideoCapWrapper):
        try:
            return reader(source, loop)
        except (InvalidInput, OpenError) as ex:
            errors[type(ex)].append(ex.message)
    try:
        return CameraCapWrapper(source, camera_resolution)
    except (InvalidInput, OpenError) as ex:
        errors[type(ex)].append(ex.message)
    if not errors[OpenError]:
        print(*errors[InvalidInput], file=sys.stderr, sep="\n")
    else:
        print(*errors[OpenError], file=sys.stderr, sep="\n")
    sys.exit(1)
