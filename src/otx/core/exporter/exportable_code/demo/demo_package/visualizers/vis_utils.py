# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""This module implements activation map."""

from __future__ import annotations

import colorsys
import random
from pathlib import Path

import cv2
import numpy as np


def get_actmap(
    saliency_map: np.ndarray,
    output_res: tuple | list,
) -> np.ndarray:
    """Get activation map (heatmap)  from saliency map.

    It will return activation map from saliency map

    Args:
        saliency_map (np.ndarray): Saliency map with pixel values from 0-255
        output_res (Union[tuple, list]): Output resolution

    Returns:
        saliency_map (np.ndarray): [H, W, 3] colormap, more red means more salient

    """
    if len(saliency_map.shape) == 3:
        saliency_map = saliency_map[0]

    saliency_map = cv2.resize(saliency_map, output_res)
    return cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)


def get_input_names_list(input_path: str | int, capture: cv2.VideoCapture) -> list[str]:
    """Lists the filenames of all inputs for demo."""
    # Web camera input
    if isinstance(input_path, int):
        return []
    if "DIR" in str(capture.get_type()):
        return [f.name for f in Path(input_path).iterdir() if f.is_file()]
    return [Path(input_path).name]


def dump_frames(saved_frames: list, output: str, input_path: str | int, capture: cv2.VideoCapture) -> None:
    """Saves images/videos with predictions from saved_frames to output folder with proper names."""
    # If no frames are saved, return
    if not saved_frames:
        return

    # Create the output folder if it doesn't exist
    output_path = Path(output)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # Get the list of input names
    filenames = get_input_names_list(input_path, capture)

    # If the input is a video, save it as video
    if "VIDEO" in str(capture.get_type()):
        filename = filenames[0]
        w, h, _ = saved_frames[0].shape
        video_path = str(output_path / filename)
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, codec, capture.fps(), (h, w))
        for frame in saved_frames:
            out.write(frame)
        out.release()
        print(f"Video was saved to {video_path}")
    # If the input is not a video, save each frame as an image
    else:
        if len(filenames) != len(saved_frames):
            filenames = [f"output_{i}.jpeg" for i, _ in enumerate(saved_frames)]
        for filename, frame in zip(filenames, saved_frames):
            image_path = str(output_path / filename)
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, frame)
            print(f"Image was saved to {image_path}")


class ColorPalette:
    """Represents a palette of colors."""

    def __init__(self, num_classes: int, rng: random.Random | None = None) -> None:
        """Initialize the ColorPalette.

        Args:
        - num_classes (int): The number of classes.
        - rng (Optional[random.Random]): The random number generator.

        Returns:
        None
        """
        if num_classes <= 0:
            msg = "ColorPalette accepts only the positive number of colors"
            raise ValueError(msg)
        if rng is None:
            rng = random.Random(0xACE)  # nosec B311  # disable random check

        candidates_num = 100
        hsv_colors = [(1.0, 1.0, 1.0)]
        for _ in range(1, num_classes):
            colors_candidates = [
                (rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0)) for _ in range(candidates_num)
            ]
            min_distances = [self._min_distance(hsv_colors, c) for c in colors_candidates]
            arg_max = np.argmax(min_distances)
            hsv_colors.append(colors_candidates[arg_max])

        self.palette = [self.hsv2rgb(*hsv) for hsv in hsv_colors]

    @staticmethod
    def _dist(c1: tuple[float, float, float], c2: tuple[float, float, float]) -> float:
        """Calculate the distance between two colors in 3D space.

        Args:
        - c1 (Tuple[float, float, float]): Tuple representing the first RGB color.
        - c2 (Tuple[float, float, float]): Tuple representing the second RGB color.

        Returns:
        float: The distance between the two colors.
        """
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    @classmethod
    def _min_distance(
        cls,
        colors_set: list[tuple[float, float, float]],
        color_candidate: tuple[float, float, float],
    ) -> float:
        """Calculate the minimum distance between color_candidate and colors_set.

        Args:
        - colors_set: List of tuples representing RGB colors.
        - color_candidate: Tuple representing an RGB color.

        Returns:
        - float: The minimum distance between color_candidate and colors_set.
        """
        distances = [cls._dist(o, color_candidate) for o in colors_set]
        return min(distances)

    def to_numpy_array(self) -> np.ndarray:
        """Convert the palette to a NumPy array.

        Returns:
        np.ndarray: The palette as a NumPy array.
        """
        return np.array(self.palette)

    @staticmethod
    def hsv2rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
        """Convert HSV color to RGB color.

        Args:
        - h (float): Hue.
        - s (float): Saturation.
        - v (float): Value.

        Returns:
        Tuple[int, int, int]: RGB color.
        """
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return int(r * 255), int(g * 255), int(b * 255)

    def __getitem__(self, n: int) -> tuple[int, int, int]:
        """Get the color at index n.

        Args:
        - n (int): Index.

        Returns:
        Tuple[int, int, int]: RGB color.
        """
        return self.palette[n % len(self.palette)]

    def __len__(self) -> int:
        """Returns the number of colors in the palette.

        Returns:
        int: The number of colors in the palette.
        """
        return len(self.palette)
