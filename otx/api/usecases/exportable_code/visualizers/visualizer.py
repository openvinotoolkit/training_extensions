"""Visualizer for results of prediction."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
from pathlib import Path

from otx.api.entities.annotation import AnnotationSceneEntity
from otx.api.utils.shape_drawer import ShapeDrawer


class IVisualizer(metaclass=abc.ABCMeta):
    """Interface for converter."""

    @abc.abstractmethod
    def draw(
        self,
        image: np.ndarray,
        annotation: AnnotationSceneEntity,
        meta: dict,
    ) -> np.ndarray:
        """Draw annotations on the image.

        Args:
            image: Input image
            annotation: Annotations to be drawn on the input image
            metadata: Metadata is needed to render

        Returns:
            Output image with annotations.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def show(self, image: np.ndarray) -> None:
        """Show result image."""

        raise NotImplementedError

    @abc.abstractmethod
    def is_quit(self) -> bool:
        """Check if user wishes to quit."""

        raise NotImplementedError


class Visualizer(IVisualizer):
    """Visualize the predicted output by drawing the annotations on the input image.

    Example:
        >>> predictions = inference_model.predict(frame)
        >>> annotation = prediction_converter.convert_to_annotation(predictions)
        >>> output = visualizer.draw(frame, annotation.shape, annotation.get_labels())
        >>> visualizer.show(output)
    """

    def __init__(
        self,
        window_name: Optional[str] = None,
        show_count: bool = False,
        is_one_label: bool = False,
        no_show: bool = False,
        delay: Optional[int] = None,
        save_results_to: Optional[str] = None
    ) -> None:
        self.window_name = "Window" if window_name is None else window_name
        self.shape_drawer = ShapeDrawer(show_count, is_one_label)

        self.delay = delay
        self.no_show = no_show
        if delay is None:
            self.delay = 1
        self.save_results_to = save_results_to
        self.saved_frames = defaultdict(list)


    def draw(
        self,
        image: np.ndarray,
        annotation: AnnotationSceneEntity,
        meta: Optional[dict] = None,
    ) -> np.ndarray:
        """Draw annotations on the image.

        Args:
            image: Input image
            annotation: Annotations to be drawn on the input image

        Returns:
            Output image with annotations.
        """

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return self.shape_drawer.draw(image, annotation, labels=[])

    def show(self, image: np.ndarray) -> None:
        """Show result image.

        Args:
            image (np.ndarray): Image to be shown.
        """

        if not self.no_show:
            cv2.imshow(self.window_name, image)

    def is_quit(self) -> bool:
        """Check user wish to quit."""
        if self.no_show:
            return False

        return ord("q") == cv2.waitKey(self.delay)

    def save_frame(self, image, input_path, streamer_type) -> None:
        """Save result image into dict.

        Args:
            image (np.ndarray): Image to be saved.
            input_path (str): Filename of the image
            streamer_type (str) : The type of the input
        """

        if self.save_results_to and input_path:
            filename = Path(input_path).name
            if "VIDEO" in streamer_type:
                self.saved_frames[filename].append(image)
            else:
                self.saved_frames[filename] = image

    def dump_frames(self, streamer) -> None:
        """Save frames to file system.

        Args:
            streamer (str): The streamer with images to be saved
        """
        if len(self.saved_frames) > 0:
            if not Path(self.save_results_to).exists():
                Path(self.save_results_to).mkdir(parents=True)

            if "VIDEO" in str(streamer.get_type()):
                filename, frames = list(self.saved_frames.items())[0]
                w, h, _ = frames[0].shape
                video_path = str(Path(self.save_results_to) / filename)
                codec = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, codec, streamer.fps(), (h, w))
                for frame in frames:
                    out.write(frame)
                out.release()
                print(f"Video was saved to {video_path}")
            else:
                for filename, frame in self.saved_frames.items():
                    image_path = str(Path(self.save_results_to, filename))
                    cv2.imwrite(image_path, frame)
                    print(f"Image was saved to {image_path}")
