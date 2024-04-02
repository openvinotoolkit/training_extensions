# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Helper functions for tests."""

import contextlib
import tempfile
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def generate_random_bboxes(
    image_width: int,
    image_height: int,
    num_boxes: int,
    min_width: int = 10,
    min_height: int = 10,
) -> np.ndarray:
    """Generate random bounding boxes.
    Parameters:
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        num_boxes (int): Number of bounding boxes to generate.
        min_width (int): Minimum width of the bounding box. Default is 10.
        min_height (int): Minimum height of the bounding box. Default is 10.
    Returns:
        ndarray: A NumPy array of shape (num_boxes, 4) representing bounding boxes in format (x_min, y_min, x_max, y_max).
    """
    max_width = image_width - min_width
    max_height = image_height - min_height

    bg = np.random.MT19937(seed=42)
    rg = np.random.Generator(bg)

    x_min = rg.integers(0, max_width, size=num_boxes)
    y_min = rg.integers(0, max_height, size=num_boxes)
    x_max = x_min + rg.integers(min_width, image_width, size=num_boxes)
    y_max = y_min + rg.integers(min_height, image_height, size=num_boxes)

    x_max[x_max > image_width] = image_width
    y_max[y_max > image_height] = image_height
    areas = (x_max - x_min) * (y_max - y_min)
    bboxes = np.column_stack((x_min, y_min, x_max, y_max))
    return bboxes[areas > 0]


@contextlib.contextmanager
def generate_random_image_folder(width: int = 480, height: int = 360, number_of_images: int = 10) -> Iterator[str]:
    """
    Generates a folder with random images, cleans up automatically if used in a `with` statement

    Parameters:
        width (int): height of the images. Defaults to 480.
        height (int): width of the images. Defaults to 360.
        number_of_images (int): number of generated images. Defaults to 10.

    Returns:
        Iterator[str]: The temporary directory
    """
    temp_dir = tempfile.TemporaryDirectory()

    for n in range(number_of_images):
        temp_file = str(Path(temp_dir.name) / f"{n}.jpg")
        _write_random_image(width, height, temp_file)

    try:
        yield temp_dir.name
    finally:
        temp_dir.cleanup()


@contextlib.contextmanager
def generate_random_video_folder(
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
    temp_dir = tempfile.TemporaryDirectory()

    for n in range(number_of_videos):
        temp_file = str(Path(temp_dir.name) / f"{n}.mp4")
        _write_random_video(width, height, number_of_frames, temp_file)

    try:
        yield temp_dir.name
    finally:
        temp_dir.cleanup()


@contextlib.contextmanager
def generate_random_single_image(width: int = 480, height: int = 360) -> Iterator[str]:
    """
    Generates a random image, cleans up automatically if used in a `with` statement

    Parameters:
        width (int): Width of the image. Defaults to 480.
        height (int): Height of the image. Defaults to 360.

    Returns:
        Iterator[str]: Path to an image file
    """

    temp_dir = tempfile.TemporaryDirectory()
    temp_file = str(Path(temp_dir.name) / "temp_image.jpg")
    _write_random_image(width, height, temp_file)

    try:
        yield temp_file
    finally:
        temp_dir.cleanup()


@contextlib.contextmanager
def generate_random_single_video(width: int = 480, height: int = 360, number_of_frames: int = 150) -> Iterator[str]:
    """
    Generates a random video, cleans up automatically if used in a `with` statement

    Parameters:
        width (int): Width of the video. Defaults to 480.
        height (int): Height of the video. Defaults to 360.
        number_of_frames (int): Number of frames in the video. Defaults to 150.

    Returns:
        Iterator[str]: Path to a video file
    """
    temp_dir = tempfile.TemporaryDirectory()
    temp_file = str(Path(temp_dir.name) / "temp_video.mp4")
    _write_random_video(width, height, number_of_frames, temp_file)

    try:
        yield temp_file
    finally:
        temp_dir.cleanup()


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


def find_folder(base_path: Path, folder_name: str) -> Path:
    """
    Find the folder with the given name within the specified base path.

    Args:
        base_path (Path): The base path to search within.
        folder_name (str): The name of the folder to find.

    Returns:
        Path: The path to the folder.
    """
    for folder_path in base_path.rglob(folder_name):
        if folder_path.is_dir():
            return folder_path
    msg = f"Folder {folder_name} not found in {base_path}."
    raise FileNotFoundError(msg)
