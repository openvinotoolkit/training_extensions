# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of AsyncExecutor in demo_package."""

import tempfile
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest

CameraStreamer = None
DirStreamer = None
ImageStreamer = None
ThreadedStreamer = None
VideoStreamer = None
get_streamer = None


@pytest.fixture(scope="module", autouse=True)
def fxt_import_module():
    global CameraStreamer, DirStreamer, ImageStreamer, ThreadedStreamer, VideoStreamer, get_streamer  # noqa: PLW0603
    from otx.core.exporter.exportable_code.demo.demo_package.streamer.streamer import CameraStreamer as Cls1
    from otx.core.exporter.exportable_code.demo.demo_package.streamer.streamer import (
        DirStreamer as Cls2,
    )
    from otx.core.exporter.exportable_code.demo.demo_package.streamer.streamer import (
        ImageStreamer as Cls3,
    )
    from otx.core.exporter.exportable_code.demo.demo_package.streamer.streamer import (
        ThreadedStreamer as Cls4,
    )
    from otx.core.exporter.exportable_code.demo.demo_package.streamer.streamer import (
        VideoStreamer as Cls5,
    )
    from otx.core.exporter.exportable_code.demo.demo_package.streamer.streamer import (
        get_streamer as func1,
    )

    CameraStreamer = Cls1
    DirStreamer = Cls2
    ImageStreamer = Cls3
    ThreadedStreamer = Cls4
    VideoStreamer = Cls5
    get_streamer = func1


@pytest.fixture()
def random_image_folder(tmp_path, width: int = 480, height: int = 360, number_of_images: int = 10) -> str:
    """
    Generates a folder with random images, cleans up automatically if used in a `with` statement

    Parameters:
        width (int): height of the images. Defaults to 480.
        height (int): width of the images. Defaults to 360.
        number_of_images (int): number of generated images. Defaults to 10.

    Returns:
        Iterator[str]: The temporary directory
    """
    for n in range(number_of_images):
        temp_file = str(tmp_path / f"{n}.jpg")
        _write_random_image(width, height, temp_file)

    return str(tmp_path)


@pytest.fixture()
def random_video_folder(
    tmp_path,
    width: int = 480,
    height: int = 360,
    number_of_videos: int = 10,
    number_of_frames: int = 150,
) -> Iterator[str]:
    """
    Generates a folder with random videos, cleans up automatically if used in a `with` statement

    Parameters:
        width (int): Width of the video. Defaults to 480.
        height (int): Height of the video. Defaults to 360.
        number_of_videos (int): Number of videos to generate. Defaults to 10.
        number_of_frames (int): Number of frames in each video. Defaults to 150.

    Returns:
        Iterator[str]: A temporary directory with videos
    """

    for n in range(number_of_videos):
        temp_file = str(tmp_path / f"{n}.mp4")
        _write_random_video(width, height, number_of_frames, temp_file)

    return str(tmp_path)


@pytest.fixture()
def random_single_image(tmp_path, width: int = 480, height: int = 360) -> str:
    """
    Generates a random image, cleans up automatically if used in a `with` statement

    Parameters:
        width (int): Width of the image. Defaults to 480.
        height (int): Height of the image. Defaults to 360.

    Returns:
        Iterator[str]: Path to an image file
    """

    temp_file = str(tmp_path / "temp_image.jpg")
    _write_random_image(width, height, temp_file)

    return temp_file


@pytest.fixture()
def random_single_video(tmp_path, width: int = 480, height: int = 360, number_of_frames: int = 150) -> str:
    """
    Generates a random video, cleans up automatically if used in a `with` statement

    Parameters:
        width (int): Width of the video. Defaults to 480.
        height (int): Height of the video. Defaults to 360.
        number_of_frames (int): Number of frames in the video. Defaults to 150.

    Returns:
        Iterator[str]: Path to a video file
    """
    temp_file = str(tmp_path / "temp_video.mp4")
    _write_random_video(width, height, number_of_frames, temp_file)

    return temp_file


def _write_random_image(width: int, height: int, filename: str) -> None:
    img = np.uint8(np.random.random((height, width, 3)) * 255)
    cv2.imwrite(filename, img)


def _write_random_video(width: int, height: int, number_of_frames: int, filename: str) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    f = filename
    videowriter = cv2.VideoWriter(f, fourcc, 30, (width, height))

    for _ in range(number_of_frames):
        img = np.uint8(np.random.random((height, width, 3)) * 255)
        videowriter.write(img)

    videowriter.release()


class TestStreamer:
    @staticmethod
    def assert_streamer_element(streamer) -> None:
        for element in streamer:
            assert element.shape == (360, 480, 3)

    def test_image_streamer_with_single_image(self, random_single_image):
        """
        <b>Description:</b>
        Test that ImageStreamer works correctly with a single image as input

        <b>Input data:</b>
        Random image file

        <b>Expected results:</b>
        Test passes if ImageStreamer returns a single image with the correct size

        <b>Steps</b>
        1. Create ImageStreamer
        2. Request image from streamer
        """
        streamer = ImageStreamer(random_single_image)
        self.assert_streamer_element(streamer)

    def test_dir_streamer_with_folder(self, random_image_folder):
        """
        <b>Description:</b>
        Test that DirStreamer works correctly with a folder of images as input

        <b>Input data:</b>
        Folder with 10 random images

        <b>Expected results:</b>
        Test passes if DirStreamer returns ten images with the correct size

        <b>Steps</b>
        1. Create DirStreamer
        2. Request images from streamer
        """
        streamer = DirStreamer(random_image_folder)
        self.assert_streamer_element(streamer)

    def test_video_streamer_with_single_video(self, random_single_video):
        """
        <b>Description:</b>
        Test that VideoStreamer works correctly with a single video as input

        <b>Input data:</b>
        Random Video file

        <b>Expected results:</b>
        Test passes if VideoStreamer can read the Video file frame by frame

        <b>Steps</b>
        1. Create VideoStreamer
        2. Request frame from VideoStreamer
        """
        streamer = VideoStreamer(random_single_video)
        self.assert_streamer_element(streamer)

    def test_video_streamer_with_loop_flag(self, random_single_video):
        """
        <b>Description:</b>
        Test that VideoStreamer works correctly with a loop flag

        <b>Input data:</b>
        Random Video file

        <b>Expected results:</b>
        Test passes if VideoStreamer returns frames with the correct amount of dimensions
        after the end of the video

        <b>Steps</b>
        1. Create VideoStreamer
        2. Request frames from streamer
        """
        streamer = VideoStreamer(random_single_video, loop=True)

        for index, frame in enumerate(streamer):
            assert frame.shape[-1] == 3
            if index > 200:
                break

    def test_video_streamer_with_single_image(self, random_single_video):
        """
        <b>Description:</b>
        Test that VideoStreamer works correctly with a single image as input

        <b>Input data:</b>
        Random image file

        <b>Expected results:</b>
        Test passes if VideoStreamer can read the single frame

        <b>Steps</b>
        1. Create VideoStreamer
        2. Request frame from VideoStreamer
        """
        streamer = VideoStreamer(random_single_video)
        self.assert_streamer_element(streamer)

    def test_invalid_inputs_to_get_streamer(self, random_video_folder):
        """
        <b>Description:</b>
        Test that get_streamer does not allow invalid inputs

        <b>Input data:</b>
        Invalid file
        Empty directory
        Folder with random videos
        Name of file that does not exist

        <b>Expected results:</b>
        Test passes if get_streamer raises a ValueError

        <b>Steps</b>
        1. Create invalid input file with .bin extension
        2. Attempt to call get_streamer with .bin file
        3. Attempt to call get_streamer with empty directory
        4. Attempt to call get_streamer with video folder, this raises because it is not supported
        5. Attempt to call get_streamer with a string that does not point to a file
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_file = Path(temp_dir) / "not_valid.bin"
            invalid_file.touch()

            with pytest.raises(Exception, match="Can't find the camera"):
                get_streamer(str(invalid_file))

        with tempfile.TemporaryDirectory() as empty_dir, pytest.raises(Exception, match="Can't find the camera"):
            get_streamer(empty_dir)

        with pytest.raises(Exception, match="Can't find the camera"):
            get_streamer(random_video_folder)

        with pytest.raises(Exception, match="Can't find the camera"):
            get_streamer("not_a_file")

    def test_valid_inputs_to_get_streamer(
        self,
        random_single_video,
        random_single_image,
        random_image_folder,
    ):
        """
        <b>Description:</b>
        Test that get_streamer return the correct Streamer class for each input

        <b>Input data:</b>
        A Video
        An Image
        Folder with images

        <b>Expected results:</b>
        Test passes if each call to get_streamer return the correct Streamer instance

        <b>Steps</b>
        1. Call get_streamer with a video file
        2. Call get_streamer with an image file
        3. Call get_streamer with a folder of images
        4. Call get_streamer with a camera index
        5. Call get_streamer with the threaded argument
        """
        streamer = get_streamer(random_single_video)
        assert isinstance(streamer, VideoStreamer)

        streamer = get_streamer(random_single_image)
        assert isinstance(streamer, ImageStreamer)

        streamer = get_streamer(random_image_folder)
        assert isinstance(streamer, DirStreamer)

        streamer = get_streamer("0")
        assert isinstance(streamer, CameraStreamer)

        streamer = get_streamer(input_stream="0", threaded=True)
        assert isinstance(streamer, ThreadedStreamer)

    def test_video_file_fails_on_image_streamer(self, random_single_video):
        """
        <b>Description:</b>
        Test that ImageStreamer raises an exception if a video is passed

        <b>Input data:</b>
        Random Video file

        <b>Expected results:</b>
        Test passes if a OpenError is raised

        <b>Steps</b>
        1. Attempt to create ImageStreamer
        """
        with pytest.raises(RuntimeError):
            ImageStreamer(random_single_video)

    def test_camera_streamer(self):
        """
        <b>Description:</b>
        Check that CameraStreamer works correctly

        <b>Input data:</b>
        CameraStreamer

        <b>Expected results:</b>
        Test passes if CameraStreamer can read 10 frames from the camera

        <b>Steps</b>
        1. Create camera streamer
        2. Retrieve frames from camera streamer
        """
        streamer = get_streamer("0")
        n = 10

        for frame in streamer:
            assert frame.shape[-1] == 3
            n -= 1
            if n == 0:
                break
