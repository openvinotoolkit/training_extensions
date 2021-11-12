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

import tempfile
from pathlib import Path
from time import sleep

import pytest

from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements
from ote_sdk.tests.test_helpers import (
    generate_random_image_folder,
    generate_random_single_image,
    generate_random_single_video,
    generate_random_video_folder,
)
from ote_sdk.usecases.exportable_code.streamer import (
    CameraStreamer,
    ImageStreamer,
    ThreadedStreamer,
    VideoStreamer,
    get_streamer,
)


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestStreamer:
    @staticmethod
    def assert_streamer_element(streamer):
        for element in streamer:
            assert element.shape == (360, 480, 3)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
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

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_image_streamer_with_folder(self):
        """
        <b>Description:</b>
        Test that ImageStreamer works correctly with a folder of images as input

        <b>Input data:</b>
        Folder with 10 random images

        <b>Expected results:</b>
        Test passes if ImageStreamer returns ten images with the correct size

        <b>Steps</b>
        1. Create ImageStreamer
        2. Request images from streamer
        """
        with generate_random_image_folder(height=360, width=480) as path:
            streamer = ImageStreamer(path)
            self.assert_streamer_element(streamer)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
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

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_video_streamer_with_folder(self):
        """
        <b>Description:</b>
        Test that VideoStreamer works correctly with a a folder of videos as input

        <b>Input data:</b>
        Folder with random videos

        <b>Expected results:</b>
        Test passes if VideoStreamer returns frames with the correct amount of dimensions

        <b>Steps</b>
        1. Create VideoStreamer
        2. Request frames from streamer
        """
        with generate_random_video_folder() as path:
            streamer = VideoStreamer(path)

            for frame in streamer:
                assert frame.shape[-1] == 3

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_image_file_fails_on_video_streamer(self):
        """
        <b>Description:</b>
        Test that VideoStreamer raises an exception if an image is passed

        <b>Input data:</b>
        Random image file

        <b>Expected results:</b>
        Test passes if a ValueError is raised

        <b>Steps</b>
        1. Attempt to create VideoStreamer
        """
        with generate_random_single_image() as path:
            with pytest.raises(ValueError):
                VideoStreamer(path)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
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

            with pytest.raises(ValueError) as context:
                get_streamer(str(invalid_file))

        the_exception = context  # .exception
        assert "not supported" in str(the_exception), str(the_exception)

        with tempfile.TemporaryDirectory() as empty_dir:
            with pytest.raises(FileNotFoundError):
                get_streamer(empty_dir)

        with generate_random_video_folder() as path:
            with pytest.raises(FileNotFoundError):
                get_streamer(path)

        with pytest.raises(ValueError) as context:
            get_streamer("not_a_file")

        the_exception = context  # .exception
        assert "does not exist" in str(the_exception), str(the_exception)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
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
            assert isinstance(streamer, ImageStreamer)

        streamer = get_streamer(camera_device=0)
        assert isinstance(streamer, CameraStreamer)

        streamer = get_streamer(camera_device=0, threaded=True)
        assert isinstance(streamer, ThreadedStreamer)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_video_file_fails_on_image_streamer(self):
        """
        <b>Description:</b>
        Test that ImageStreamer raises an exception if a video is passed

        <b>Input data:</b>
        Random Video file

        <b>Expected results:</b>
        Test passes if a ValueError is raised

        <b>Steps</b>
        1. Attempt to create ImageStreamer
        """
        with generate_random_single_video() as path:
            with pytest.raises(ValueError):
                ImageStreamer(path)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
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
        streamer = get_streamer()
        n = 10

        for frame in streamer:
            assert frame.shape[-1] == 3
            n -= 1
            if n == 0:
                break

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_threaded_streamer(self):
        """
        <b>Description:</b>
        Check that ThreadedStreamer works correctly

        <b>Input data:</b>
        Folder with images

        <b>Expected results:</b>
        Test passes if ThreadedStreamer reads 3 frames from the folder

        <b>Steps</b>
        1. Create ThreadedStreamer
        2. Retrieve frames from ThreadedStreamer
        """
        with generate_random_image_folder() as path:
            streamer = get_streamer(path, threaded=True)
            frame_count = 0

            for frame in streamer:
                assert frame.shape[-1] == 3
                frame_count += 1

                if frame_count == 3:
                    break

            assert frame_count == 3

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_threaded_streamer_timeout(self):
        """
        <b>Description:</b>
        Check that ThreadedStreamer works correctly even if the main thread is very slow

        <b>Input data:</b>
        Folder with images

        <b>Expected results:</b>
        Test passes if ThreadedStreamer returns 5 images

        <b>Steps</b>
        1. Create ThreadedStreamer
        2. Retrieve frames from camera streamer and suspend each thread in between frames
        """
        with generate_random_image_folder() as path:
            streamer = get_streamer(path, threaded=True)

            streamer.buffer_size = 2
            frame_count = 0

            for frame in streamer:
                assert frame.shape[-1] == 3
                sleep(1)
                frame_count += 1
                if frame_count == 5:
                    break

            assert frame_count == 5

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_get_streamer_parses_path(self):
        """
        <b>Description:</b>
        Test that get_streamer raises an error if both camera_device and path are provided

        <b>Input data:</b>
        Path to a folder
        Camera Index

        <b>Expected results:</b>
        Test passes if a ValueError is raised

        <b>Steps</b>
        1. Attempt to call get_streamer with path and camera_device
        """
        with generate_random_image_folder(number_of_images=1) as path:
            with pytest.raises(ValueError):
                get_streamer(path=path, camera_device=0)
