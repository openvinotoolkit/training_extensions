# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of AsyncExecutor in demo_package."""

import tempfile
from pathlib import Path

import pytest
from otx.core.exporter.exportable_code.demo.demo_package.streamer.streamer import (
    CameraStreamer,
    DirStreamer,
    ImageStreamer,
    ThreadedStreamer,
    VideoStreamer,
    get_streamer,
)

from tests.test_helpers import (
    generate_random_image_folder,
    generate_random_single_image,
    generate_random_single_video,
    generate_random_video_folder,
)


class TestStreamer:
    @staticmethod
    def assert_streamer_element(streamer) -> None:
        for element in streamer:
            assert element.shape == (360, 480, 3)

    def test_image_streamer_with_single_image(self):
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
        with generate_random_single_image(height=360, width=480) as path:
            streamer = ImageStreamer(path)
            self.assert_streamer_element(streamer)

    def test_dir_streamer_with_folder(self):
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
        with generate_random_image_folder(height=360, width=480) as path:
            streamer = DirStreamer(path)
            self.assert_streamer_element(streamer)

    def test_video_streamer_with_single_video(self):
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
        with generate_random_single_video(height=360, width=480) as path:
            streamer = VideoStreamer(path)
            self.assert_streamer_element(streamer)

    def test_video_streamer_with_loop_flag(self):
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
        with generate_random_single_video(height=360, width=480, number_of_frames=100) as path:
            streamer = VideoStreamer(path, loop=True)

            for index, frame in enumerate(streamer):
                assert frame.shape[-1] == 3
                if index > 200:
                    break

    def test_video_streamer_with_single_image(self):
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
        with generate_random_single_video(height=360, width=480) as path:
            streamer = VideoStreamer(path)
            self.assert_streamer_element(streamer)

    def test_invalid_inputs_to_get_streamer(self):
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

        with generate_random_video_folder() as path, pytest.raises(Exception, match="Can't find the camera"):
            get_streamer(path)

        with pytest.raises(Exception, match="Can't find the camera"):
            get_streamer("not_a_file")

    def test_valid_inputs_to_get_streamer(self):
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
        with generate_random_single_video() as path:
            streamer = get_streamer(path)
            assert isinstance(streamer, VideoStreamer)

        with generate_random_single_image() as path:
            streamer = get_streamer(path)
            assert isinstance(streamer, ImageStreamer)

        with generate_random_image_folder() as path:
            streamer = get_streamer(path)
            assert isinstance(streamer, DirStreamer)

        streamer = get_streamer("0")
        assert isinstance(streamer, CameraStreamer)

        streamer = get_streamer(input_stream="0", threaded=True)
        assert isinstance(streamer, ThreadedStreamer)

    def test_video_file_fails_on_image_streamer(self):
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
        with generate_random_single_video() as path, pytest.raises(RuntimeError):
            ImageStreamer(path)

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
