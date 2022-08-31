"""This module implements helpers for drawing shapes."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals
import abc
from typing import (
    Callable,
    Generic,
    List,
    NewType,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import cv2
import numpy as np

from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    NullAnnotationSceneEntity,
)
from otx.api.entities.coordinate import Coordinate
from otx.api.entities.label import LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.polygon import Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.shapes.shape import ShapeEntity

CvTextSize = NewType("CvTextSize", Tuple[Tuple[int, int], int])

_Any = TypeVar("_Any")


class DrawerEntity(Generic[_Any]):
    """An interface to draw a shape of type ``T`` onto an image."""

    supported_types: Sequence[Type[ShapeEntity]] = []

    @abc.abstractmethod
    def draw(self, image: np.ndarray, entity: _Any, labels: List[ScoredLabel]) -> np.ndarray:
        """Draw an entity to a given frame.

        Args:
            image (np.ndarray): The image to draw the entity on.
            entity (T): The entity to draw.
            labels (List[ScoredLabel]): Labels of the shapes to draw

        Returns:
            np.ndarray: frame with shape drawn on it
        """
        raise NotImplementedError


class Helpers:
    """Contains variables which are used by all subclasses.

    Contains functions which help with generating coordinates, text and text scale.
    These functions are use by the DrawerEntity Classes when drawing to an image.
    """

    def __init__(self) -> None:
        # Same alpha value that the UI uses for Labels
        self.alpha_shape = 100 / 256
        self.alpha_labels = 153 / 256
        self.assumed_image_width_for_text_scale = 1280  # constant number for size of classification/counting overlay
        self.top_margin = 0.07  # part of the top screen reserved for top left classification/counting overlay
        self.content_padding = 3
        self.top_left_box_thickness = 1
        self.content_margin = 2
        self.label_offset_box_shape = 0
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.yellow = (255, 255, 0)

        self.cursor_pos = Coordinate(0, 0)
        self.line_height = 0

    @staticmethod
    def draw_transparent_rectangle(
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        alpha: float,
    ) -> np.ndarray:
        """Draw a rectangle on an image.

        Args:
            img (np.ndarray): Image
            x1 (int): Left side
            y1 (int): Top side
            x2 (int): Right side
            y2 (int): Bottom side
            color (Tuple[int, int, int]): Color
            alpha (float): Alpha value between 0 and 1
        """
        x1 = np.clip(x1, 0, img.shape[1] - 1)
        y1 = np.clip(y1, 0, img.shape[0] - 1)
        x2 = np.clip(x2 + 1, 0, img.shape[1] - 1)
        y2 = np.clip(y2 + 1, 0, img.shape[0] - 1)
        rect = img[y1:y2, x1:x2]
        rect[:] = (alpha * np.array(color))[np.newaxis, np.newaxis] + (1 - alpha) * rect
        return img

    def generate_text_scale(self, image: np.ndarray) -> float:
        """Calculates the scale of the text.

        Args:
            image (np.ndarray): Image to calculate the text scale for.

        Returns:
            scale for the text
        """
        return round(image.shape[1] / self.assumed_image_width_for_text_scale, 1)

    @staticmethod
    def generate_text_for_label(
        label: Union[LabelEntity, ScoredLabel], show_labels: bool, show_confidence: bool
    ) -> str:
        """Return a string representing a given label and its associated probability if label is a ScoredLabel.

        The exact format of the string depends on the function parameters described below.

        Args:
            label (Union[LabelEntity, ScoredLabel]): Label
            show_labels (bool): Whether to render the labels above the shape
            show_confidence (bool): Whether to render the confidence above the
                shape

        Returns:
            str: Formatted string (e.g. `"Cat 58%"`)
        """
        text = ""
        if show_labels:
            text += label.name
        if show_confidence and isinstance(label, ScoredLabel):
            if len(text) > 0:
                text += " "
            text += f"{label.probability:.0%}"
        return text

    def generate_draw_command_for_labels(
        self,
        labels: Sequence[Union[LabelEntity, ScoredLabel]],
        image: np.ndarray,
        show_labels: bool,
        show_confidence: bool,
    ) -> Tuple[Callable[[np.ndarray], np.ndarray], int, int]:
        """Generate draw function and content width and height for labels.

        Generates a function which can be called to draw a list of labels onto an image relatively to the
        cursor position.
        The width and height of the content is also returned and can be determined to compute
        the best position for content before actually drawing it.

        Args:
            labels (Sequence[Union[LabelEntity, ScoredLabel]]): List of labels
            image (np.ndarray): Image (used to compute font size)
            show_labels (bool): Whether to show the label name
            show_confidence (bool): Whether to show the confidence probability

        Returns:
            A tuple containing the drawing function, the content width,
            and the content height
        """
        draw_commands = []
        content_width = 0
        content_height = 0

        # Loop through the list of labels and create a function which can be used to draw the label.
        for label in labels:
            text = self.generate_text_for_label(label, show_labels, show_confidence)
            text_scale = self.generate_text_scale(image)
            thickness = int(text_scale / 2)
            color = label.color.bgr_tuple

            item_command, item_width, item_height = self.generate_draw_command_for_text(
                text, text_scale, thickness, color
            )

            draw_commands.append(item_command)

            content_width += item_width
            content_height = max(content_height, item_height)

        def draw_command(img: np.ndarray) -> np.ndarray:
            for command in draw_commands:
                img = command(img)
            return img

        return draw_command, content_width, content_height

    def generate_draw_command_for_text(
        self, text: str, text_scale: float, thickness: int, color: Tuple[int, int, int]
    ) -> Tuple[Callable[[np.ndarray], np.ndarray], int, int]:
        """Generate function to draw text on image relative to cursor position.

        Generate a function which can be called to draw the given text onto an image
        relatively to the cursor position.

        The width and height of the content is also returned and can be determined to compute
        the best position for content before actually drawing it.

        Args:
            text (str): Text to draw
            text_scale (float): Font size
            thickness (int): Thickness of the text
            color (Tuple[int, int, int]): Color of the text

        Returns:
            A tuple containing the drawing function, the content width,
            and the content height
        """

        padding = self.content_padding
        margin = self.content_margin

        label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_scale, thickness=thickness)

        baseline = label_size[1]
        text_width = label_size[0][0]
        text_height = label_size[0][1]

        width = text_width + 2 * padding
        height = text_height + baseline + 2 * padding
        content_width = width + margin

        if (color[0] + color[1] + color[2]) / 3 > 200:
            text_color = self.black
        else:
            text_color = self.white

        def draw_command(img: np.ndarray) -> np.ndarray:
            cursor_pos = Coordinate(int(self.cursor_pos.x), int(self.cursor_pos.y))
            self.draw_transparent_rectangle(
                img,
                int(cursor_pos.x),
                int(cursor_pos.y),
                int(cursor_pos.x + width),
                int(cursor_pos.y + height),
                color,
                self.alpha_labels,
            )

            img = cv2.putText(
                img=img,
                text=text,
                org=(
                    cursor_pos.x + padding,
                    cursor_pos.y + height - padding - baseline,
                ),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=text_scale,
                color=text_color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

            self.cursor_pos.x += content_width
            self.line_height = height

            return img

        return draw_command, content_width, height

    @staticmethod
    def draw_flagpole(
        image: np.ndarray,
        flagpole_start_point: Coordinate,
        flagpole_end_point: Coordinate,
    ):
        """Draw a small flagpole between two points.

        Args:
            image: Image
            flagpole_start_point: Start of the flagpole
            flagpole_end_point: End of the flagpole

        Returns:
            Image
        """
        return cv2.line(
            image,
            flagpole_start_point.as_int_tuple(),
            flagpole_end_point.as_int_tuple(),
            color=[0, 0, 0],
            thickness=2,
        )

    def newline(self):
        """Move the cursor to the next line."""
        self.cursor_pos.x = 0
        self.cursor_pos.y += self.line_height + self.content_margin

    def set_cursor_pos(self, cursor_pos: Optional[Coordinate] = None):
        """Move the cursor to a new position.

        Args:
            cursor_pos (Optional[Coordinate]): New position of the cursor; (0,0) if not specified.
        """
        if cursor_pos is None:
            cursor_pos = Coordinate(0, 0)

        self.cursor_pos = cursor_pos


class ShapeDrawer(DrawerEntity[AnnotationSceneEntity]):
    """ShapeDrawer to draw any shape on a numpy array. Will overlay the shapes in the same way that the UI does.

    Args:
        show_count: Whether or not to render the amount of objects on
            screen in the top left.
        is_one_label: Whether there is only one label present in the
            project.
    """

    # TODO Connect show_count,is_is_one_label to the UI for toggling.
    def __init__(self, show_count, is_one_label):
        super().__init__()
        self.show_labels = True
        self.show_confidence = True
        self.show_count = show_count
        self.is_one_label = is_one_label

        if self.is_one_label and not self.show_count:
            self.show_labels = False

        self.shape_drawers = [
            self.RectangleDrawer(self.show_labels, self.show_confidence),
            self.PolygonDrawer(self.show_labels, self.show_confidence),
            self.EllipseDrawer(self.show_labels, self.show_confidence),
        ]

        # Always show global labels, especially if shape labels are disabled (because of is_one_label).
        self.top_left_drawer = self.TopLeftDrawer(True, self.show_confidence, self.is_one_label)

    def draw(
        self,
        image: np.ndarray,
        entity: AnnotationSceneEntity,
        labels: List[ScoredLabel],
    ) -> np.ndarray:
        """Use a compatible drawer to draw all shapes of an annotation to the corresponding image.

        Also render a label in the top left if we need to.

        Args:
            image: Numpy image, one frame of a video on which to draw
                something
            entity: AnnotationSceneEntity entity corresponding to this
                particular frame of the video
            labels: Can be passed as an empty list since they are
                already present in annotation_scene

        Returns:
            Modified image.
        """

        num_annotations = 0

        self.top_left_drawer.set_cursor_pos()

        if not isinstance(entity, NullAnnotationSceneEntity):
            for annotation in entity.annotations:
                if (
                    isinstance(annotation.shape, Rectangle)
                    and annotation.shape.x1 == 0
                    and annotation.shape.y1 == 0
                    and annotation.shape.x2 == 1
                    and annotation.shape.y2 == 1
                ):
                    # If is_one_label is activated, don't draw the labels here
                    # because we will draw them again outside the loop.
                    if not self.is_one_label:
                        image = self.top_left_drawer.draw(image, annotation, labels=[])
                else:
                    num_annotations += 1
                    for drawer in self.shape_drawers:
                        if type(annotation.shape) in drawer.supported_types and len(annotation.get_labels()) > 0:
                            image = drawer.draw(image, annotation.shape, labels=annotation.get_labels())
            if self.is_one_label:
                image = self.top_left_drawer.draw_labels(image, entity.get_labels())
            if self.show_count:
                image = self.top_left_drawer.draw_annotation_count(image, num_annotations)
        return image

    class TopLeftDrawer(Helpers, DrawerEntity[Annotation]):
        """Draws labels in an image's top left corner."""

        def __init__(self, show_labels, show_confidence, is_one_label):
            super().__init__()
            self.show_labels = show_labels
            self.show_confidence = show_confidence
            self.is_one_label = is_one_label

        def draw(self, image: np.ndarray, entity: Annotation, labels: List[ScoredLabel]) -> np.ndarray:
            """Draw the labels of a shape in the image top left corner.

            Args:
                image (np.ndarray): Image
                entity (Annotation): Annotation
                labels (List[ScoredLabels]): (Unused) labels to be drawn on the image

            Returns:
                np.ndarray: Image with label on top.
            """
            return self.draw_labels(image, entity.get_labels())

        def draw_labels(self, image: np.ndarray, labels: Sequence[Union[LabelEntity, ScoredLabel]]) -> np.ndarray:
            """Draw the labels in the image top left corner.

            Args:
                image (np.ndarray): Image
                labels (Sequence[Union[LabelEntity, ScoredLabel]]): Sequence of labels

            Returns:
                np.ndarray: Image with label on top.
            """
            show_confidence = self.show_confidence if not self.is_one_label else False

            draw_command, _, _ = self.generate_draw_command_for_labels(labels, image, self.show_labels, show_confidence)

            image = draw_command(image)

            if len(labels) > 0:
                self.newline()

            return image

        def draw_annotation_count(self, image: np.ndarray, num_annotations: int) -> np.ndarray:
            """Draw the number of annotations to the top left corner of the image.

            Args:
                image (np.ndarray): Image
                num_annotations (int): Number of annotations

            Returns:
                np.ndarray: Image with annotation count on top.
            """
            text = f"Count: {num_annotations}"
            color = self.yellow

            text_scale = self.generate_text_scale(image)
            draw_command, _, _ = self.generate_draw_command_for_text(
                text, text_scale, self.top_left_box_thickness, color
            )
            image = draw_command(image)

            self.newline()

            return image

    class RectangleDrawer(Helpers, DrawerEntity[Rectangle]):
        """Draws rectangles."""

        supported_types = [Rectangle]

        def __init__(self, show_labels, show_confidence):
            super().__init__()
            self.show_labels = show_labels
            self.show_confidence = show_confidence

        def draw(self, image: np.ndarray, entity: Rectangle, labels: List[ScoredLabel]) -> np.ndarray:
            """Draws a rectangle on the image along with labels.

            Args:
                image (np.ndarray): Image to draw on.
                entity (Rectangle): Rectangle to draw.
                labels (List[ScoredLabel]): List of labels.

            Returns:
                np.ndarray: Image with rectangle drawn on it.
            """
            base_color = labels[0].color.bgr_tuple

            # Draw the rectangle on the image
            x1, y1 = int(entity.x1 * image.shape[1]), int(entity.y1 * image.shape[0])
            x2, y2 = int(entity.x2 * image.shape[1]), int(entity.y2 * image.shape[0])
            image = self.draw_transparent_rectangle(image, x1, y1, x2, y2, base_color, self.alpha_shape)
            image = cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=base_color, thickness=2)

            (
                draw_command,
                content_width,
                content_height,
            ) = self.generate_draw_command_for_labels(labels, image, self.show_labels, self.show_confidence)

            # Generate a command to draw the list of labels
            # and compute the actual size of the list of labels.
            y_coord = y1 - self.label_offset_box_shape - content_height
            x_coord = x1

            # put label inside if it is out of bounds at the top of the shape, and shift label to left if needed
            if y_coord < self.top_margin * image.shape[0]:
                y_coord = y1 + self.label_offset_box_shape
            if x_coord + content_width > image.shape[1]:
                x_coord = x2 - content_width

            # Draw the list of labels.
            self.set_cursor_pos(Coordinate(x_coord, y_coord))
            image = draw_command(image)
            return image

    class EllipseDrawer(Helpers, DrawerEntity[Ellipse]):
        """Draws ellipses."""

        supported_types = [Ellipse]

        def __init__(self, show_labels, show_confidence):
            super().__init__()
            self.show_labels = show_labels
            self.show_confidence = show_confidence

        def draw(self, image: np.ndarray, entity: Ellipse, labels: List[ScoredLabel]) -> np.ndarray:
            """Draw the ellipse on the image.

            Args:
                image (np.ndarray): Image to draw on.
                entity (Ellipse): Ellipse to draw.
                labels (List[ScoredLabel]): Labels to draw.

            Returns:
                np.ndarray: Image with the ellipse drawn on it.
            """
            base_color = labels[0].color.bgr_tuple
            if entity.width > entity.height:
                axes = (
                    int(entity.major_axis * image.shape[1]),
                    int(entity.minor_axis * image.shape[0]),
                )
            else:
                axes = (
                    int(entity.major_axis * image.shape[0]),
                    int(entity.minor_axis * image.shape[1]),
                )
            center = (
                int(entity.x_center * image.shape[1]),
                int(entity.y_center * image.shape[0]),
            )
            # Draw the shape on the image
            alpha = self.alpha_shape
            overlay = cv2.ellipse(
                img=image.copy(),
                center=center,
                axes=axes,
                angle=0,
                startAngle=0,
                endAngle=360,
                color=base_color,
                thickness=cv2.FILLED,
            )
            result_without_border = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            result_with_border = cv2.ellipse(
                img=result_without_border,
                center=center,
                axes=axes,
                angle=0,
                startAngle=0,
                endAngle=360,
                color=base_color,
                lineType=cv2.LINE_AA,
            )

            # Generate a command to draw the list of labels
            # and compute the actual size of the list of labels.
            (
                draw_command,
                content_width,
                content_height,
            ) = self.generate_draw_command_for_labels(labels, image, self.show_labels, self.show_confidence)

            # get top left corner of imaginary bbox around circle
            offset = self.label_offset_box_shape
            x_coord = entity.x1 * image.shape[1]
            y_coord = entity.y1 * image.shape[0] - offset - content_height

            flagpole_end_point = Coordinate(int(x_coord + 1), int(entity.y_center * image.shape[0]))

            # put label at bottom if it is out of bounds at the top of the shape, and shift label to left if needed
            if y_coord < self.top_margin * image.shape[0]:
                y_coord = (entity.y1 * image.shape[0]) + (entity.y2 * image.shape[0]) + offset
                flagpole_start_point = Coordinate(x_coord + 1, y_coord)
            else:
                flagpole_start_point = Coordinate(x_coord + 1, y_coord + content_height)

            if x_coord + content_width > result_with_border.shape[1]:
                # The list of labels is too close to the right side of the image.
                # Move it slightly to the left.
                x_coord = result_with_border.shape[1] - content_width

            # Draw the list of labels and a small flagpole.
            self.set_cursor_pos(Coordinate(x_coord, y_coord))
            image = draw_command(result_with_border)
            image = self.draw_flagpole(image, flagpole_start_point, flagpole_end_point)

            return image

    class PolygonDrawer(Helpers, DrawerEntity[Polygon]):
        """Draws polygons."""

        supported_types = [Polygon]

        def __init__(self, show_labels, show_confidence):
            super().__init__()
            self.show_labels = show_labels
            self.show_confidence = show_confidence

        def draw(self, image: np.ndarray, entity: Polygon, labels: List[ScoredLabel]) -> np.ndarray:
            """Draw polygon and labels on image.

            Args:
                image (np.ndarray): Image to draw on.
                entity (Polygon): Polygon to draw.
                labels (List[ScoredLabel]): List of labels to draw.

            Returns:
                np.ndarray: Image with polygon drawn on it.
            """
            base_color = labels[0].color.bgr_tuple

            # Draw the shape on the image
            alpha = self.alpha_shape
            contours = np.array(
                [[point.x * image.shape[1], point.y * image.shape[0]] for point in entity.points],
                dtype=np.int32,
            )
            overlay = cv2.drawContours(
                image=image.copy(),
                contours=[contours],
                contourIdx=-1,
                color=base_color,
                thickness=cv2.FILLED,
            )
            result_without_border = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            result_with_border = cv2.drawContours(
                image=result_without_border,
                contours=[contours],
                contourIdx=-1,
                color=base_color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            # Generate a command to draw the list of labels
            # and compute the actual size of the list of labels.
            (
                draw_command,
                content_width,
                content_height,
            ) = self.generate_draw_command_for_labels(labels, image, self.show_labels, self.show_confidence)

            # get top left corner of imaginary bbox around polygon
            x_coord = min(point[0] for point in contours)
            y_coord = min(point[1] for point in contours) - self.label_offset_box_shape - content_height

            # end point = Y in polygon where X is lowest, x offset to make line flush with text rectangle
            _, idx = min((val, idx) for (idx, val) in enumerate([point[0] for point in contours]))
            flagpole_end_point = Coordinate(x_coord + 1, [point[1] for point in contours][idx])

            if y_coord < self.top_margin * image.shape[0]:
                # The polygon is too close to the top of the image.
                # Draw the labels underneath the polygon instead.
                y_coord = max(point[1] for point in contours) + self.label_offset_box_shape
                flagpole_start_point = Coordinate(x_coord + 1, y_coord)
            else:
                flagpole_start_point = Coordinate(x_coord + 1, y_coord + content_height)

            if x_coord + content_width > result_with_border.shape[1]:
                # The list of labels is too close to the right side of the image.
                # Move it slightly to the left.
                x_coord = result_with_border.shape[1] - content_width

            # Draw the list of labels and a small flagpole.
            self.set_cursor_pos(Coordinate(x_coord, y_coord))
            image = draw_command(result_with_border)
            image = self.draw_flagpole(image, flagpole_start_point, flagpole_end_point)

            return image
