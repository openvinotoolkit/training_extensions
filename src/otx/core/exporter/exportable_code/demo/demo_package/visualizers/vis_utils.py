"""This module implements activation map."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import colorsys
import random
from pathlib import Path
from typing import Union

import cv2
import numpy as np


def get_actmap(
    saliency_map: np.ndarray,
    output_res: Union[tuple, list],
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
    saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    return saliency_map


def get_input_names_list(input_path: Union[str, int], capture):
    """Lists the filenames of all inputs for demo."""
    # Web camera input
    if isinstance(input_path, int):
        return []
    if "DIR" in str(capture.get_type()):
        return [f.name for f in Path(input_path).iterdir() if f.is_file()]
    else:
        return [Path(input_path).name]


def dump_frames(saved_frames: list, output: str, input_path: Union[str, int], capture):
    """Saves images/videos with predictions from saved_frames to output folder with proper names."""
    if not saved_frames:
        return

    output_path = Path(output)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    filenames = get_input_names_list(input_path, capture)

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
    else:
        if len(filenames) != len(saved_frames):
            filenames = [f"output_{i}.jpeg" for i, _ in enumerate(saved_frames)]
        for filename, frame in zip(filenames, saved_frames):
            image_path = str(output_path / filename)
            cv2.imwrite(image_path, frame)
            print(f"Image was saved to {image_path}")


class ColorPalette:
    def __init__(self, n, rng=None):
        if n == 0:
            raise ValueError("ColorPalette accepts only the positive number of colors")
        if rng is None:
            rng = random.Random(0xACE)  # nosec B311  # disable random check

        candidates_num = 100
        hsv_colors = [(1.0, 1.0, 1.0)]
        for _ in range(1, n):
            colors_candidates = [
                (rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0)) for _ in range(candidates_num)
            ]
            min_distances = [self.min_distance(hsv_colors, c) for c in colors_candidates]
            arg_max = np.argmax(min_distances)
            hsv_colors.append(colors_candidates[arg_max])

        self.palette = [self.hsv2rgb(*hsv) for hsv in hsv_colors]

    @staticmethod
    def dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    @classmethod
    def min_distance(cls, colors_set, color_candidate):
        distances = [cls.dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    def to_numpy_array(self):
        return np.array(self.palette)

    @staticmethod
    def hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    def __getitem__(self, n):
        return self.palette[n % len(self.palette)]

    def __len__(self):
        return len(self.palette)
