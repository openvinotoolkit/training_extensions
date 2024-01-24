"""Visualizer for results of prediction."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import time
from typing import Optional
import logging as log

import cv2
import numpy as np
from ..streamer import BaseStreamer
from openvino.model_api.performance_metrics import put_highlighted_text
from pathlib import Path
from .vis_utils import ColorPalette


class BaseVisualizer:
    """Interface for converter."""

    def __init__(
        self,
        window_name: Optional[str] = None,
        no_show: bool = False,
        delay: Optional[int] = None,
        output: Optional[str] = "./outputs/model_visualization",
    ) -> None:
        self.window_name = "Window" if window_name is None else window_name

        self.delay = delay
        self.no_show = no_show
        if delay is None:
            self.delay = 1
        self.output = output

    def draw(
        self,
        image: np.ndarray,
        predictions: list,
        meta: dict,
        output_transform: Optional[list] = None
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

    def video_delay(self, elapsed_time: float, streamer: BaseStreamer):
        """Check if video frames were inferenced faster than the original video FPS and delay visualizer if so.

        Args:
            elapsed_time (float): Time spent on frame inference
            streamer (BaseStreamer): Streamer object
        """
        if self.no_show:
            return
        if "VIDEO" in str(streamer.get_type()):
            orig_frame_time = 1 / streamer.fps()
            if elapsed_time < orig_frame_time:
                time.sleep(orig_frame_time - elapsed_time)


class FakeVisualizer(BaseVisualizer):
    def draw(
        self,
        frame: np.ndarray,
        predictions: list,
        meta: dict,
        output_transform: Optional[list] = None
    ) -> np.ndarray:
        """Immitate drawing annotations.

        Args:
            image: Input image
            annotation: Annotations to be drawn on the input image
            metadata: Metadata is needed to render

        Returns:
            Output same image without any annotations.
        """
        return frame


class ClassificationVisualizer(BaseVisualizer):
    """Visualize the predicted classification labels by drawing the annotations on the input image.

    Example:
        >>> predictions = inference_model.predict(frame)
        >>> output = visualizer.draw(frame, predictions)
        >>> visualizer.show(output)
    """

    def draw(
        self,
        frame: np.ndarray,
        predictions: list,
        meta: Optional[dict] = None,
        output_transform: Optional[list] = None
    ) -> np.ndarray:
        """Draw annotations on the image.

        Args:
            image: Input image
            annotation: Annotations to be drawn on the input image

        Returns:
            Output image with annotations.
        """
        if output_transform is not None:
            frame = output_transform.resize(frame)
        class_label = ""
        if predictions:
            class_label = predictions[0][1]
        font_scale = 0.7
        label_height = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][1]
        initial_labels_pos =  frame.shape[0] - label_height * (int(1.5 * len(predictions)) + 1)

        if (initial_labels_pos < 0):
            initial_labels_pos = label_height
            log.warning('Too much labels to display on this frame, some will be omitted')
        offset_y = initial_labels_pos

        header = "Label:     Score:"
        label_width = cv2.getTextSize(header, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][0]
        put_highlighted_text(frame, header, (frame.shape[1] - label_width, offset_y),
            cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 0, 0), 2)

        for idx, class_label, score in predictions:
            label = '{}. {}    {:.2f}'.format(idx, class_label, score)
            label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][0]
            offset_y += int(label_height * 1.5)
            predictions(frame, label, (frame.shape[1] - label_width, offset_y),
                cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 0, 0), 2)
        return frame


class SemanticSegmentationVisualizer(BaseVisualizer):
    def __init__(self, *args, colors_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if colors_path:
            self.color_palette = self.get_palette_from_file(colors_path)
            log.debug('The palette is loaded from {}'.format(colors_path))
        else:
            pascal_palette_path = Path(__file__).resolve().parents[3] /\
                'data/palettes/pascal_voc_21cl_colors.txt'
            self.color_palette = self.get_palette_from_file(pascal_palette_path)
            log.debug('The PASCAL VOC palette is used')
        log.debug('Get {} colors'.format(len(self.color_palette)))
        self.color_map = self.create_color_map()

    def get_palette_from_file(self, colors_path):
        with open(colors_path, 'r') as file:
            colors = []
            for line in file.readlines():
                values = line[line.index('(')+1:line.index(')')].split(',')
                colors.append([int(v.strip()) for v in values])
            return colors

    def create_color_map(self):
        classes = np.array(self.color_palette, dtype=np.uint8)[:, ::-1] # RGB to BGR
        color_map = np.zeros((256, 1, 3), dtype=np.uint8)
        classes_num = len(classes)
        color_map[:classes_num, 0, :] = classes
        color_map[classes_num:, 0, :] = np.random.uniform(0, 255, size=(256-classes_num, 3))
        return color_map

    def apply_color_map(self, input):
        input_3d = cv2.merge([input, input, input])
        return cv2.LUT(input_3d, self.color_map)

    def draw(self, frame, masks, only_masks=False,  meta: Optional[dict] = None,
        output_transform: Optional[list] = None):
        output = self.apply_color_map(masks)
        if not only_masks:
            output = cv2.addWeighted(frame, 0.5, output, 0.5, 0)
        return output


class ObjectDetectionVisualizer(BaseVisualizer):
    def draw(
        self,
        frame: np.ndarray,
        predictions: list,
        meta: Optional[dict] = None,
        output_transform: Optional[list] = None
    ) -> np.ndarray:
        if output_transform is not None:
            frame = output_transform.resize(frame)
        labels = meta["labels"]
        palette = meta["palette"]
        palette = ColorPalette(len(labels))

        for detection in predictions:
            class_id = int(detection.id)
            color = palette[class_id]
            det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
            xmin, ymin, xmax, ymax = detection.get_coords()
            if output_transform:
                xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                        (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

        return frame


class InstanceSegmentationVisualizer:
    def __init__(self, *args, labels=None, show_boxes=False, show_scores=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = labels
        colors_num = len(labels) if labels else 80
        self.show_boxes = show_boxes
        self.show_scores = show_scores
        self.palette = ColorPalette(colors_num)

    def draw(self, image, boxes, classes, scores, masks=None, ids=None, texts=None):
        result = image.copy()

        if masks is not None:
            result = self.overlay_masks(result, masks, ids)
        if self.show_boxes:
            result = self.overlay_boxes(result, boxes, classes)

        result = self.overlay_labels(result, boxes, classes, scores, texts)
        return result

    def overlay_masks(self, image, masks, ids=None):
        segments_image = image.copy()
        aggregated_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        aggregated_colored_mask = np.zeros(image.shape, dtype=np.uint8)
        all_contours = []

        for i, mask in enumerate(masks):
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if contours:
                all_contours.append(contours[0])

            mask_color = self.palette[i if ids is None else ids[i]]
            cv2.bitwise_or(aggregated_mask, mask, dst=aggregated_mask)
            cv2.bitwise_or(aggregated_colored_mask, mask_color, dst=aggregated_colored_mask, mask=mask)

        # Fill the area occupied by all instances with a colored instances mask image
        cv2.bitwise_and(segments_image, (0, 0, 0), dst=segments_image, mask=aggregated_mask)
        cv2.bitwise_or(segments_image, aggregated_colored_mask, dst=segments_image, mask=aggregated_mask)

        cv2.addWeighted(image, 0.5, segments_image, 0.5, 0, dst=image)
        cv2.drawContours(image, all_contours, -1, (0, 0, 0))
        return image

    def overlay_boxes(self, image, boxes, classes):
        for box, class_id in zip(boxes, classes):
            color = self.palette[class_id]
            box = box.astype(int)
            top_left, bottom_right = box[:2], box[2:]
            image = cv2.rectangle(image, top_left, bottom_right, color, 2)
        return image

    def overlay_labels(self, image, boxes, classes, scores, texts=None):
        if texts:
            labels = texts
        elif self.labels:
            labels = (self.labels[class_id] for class_id in classes)
        else:
            raise RuntimeError('InstanceSegmentationVisualizer must contain either labels or texts to display')
        template = '{}: {:.2f}' if self.show_scores else '{}'

        for box, score, label in zip(boxes, scores, labels):
            text = template.format(label, score)
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            position = ((box[:2] + box[2:] - textsize) / 2).astype(np.int32)
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image
