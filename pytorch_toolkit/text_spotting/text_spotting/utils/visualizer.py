import numpy as np
import cv2

import torch

from segmentoly.utils.visualizer import Visualizer
from text_spotting.data.alphabet import AlphabetDecoder

class TextVisualizer(Visualizer):

    def __init__(self, text_confidence_threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alphabet_decoder = AlphabetDecoder()
        self.text_confidence_threshold = text_confidence_threshold

    def __call__(self, image, boxes, classes, scores, segms=None, ids=None, text_log_softmax=None):
        result = image.copy()

        # Filter out detections with low confidence.
        filter_mask = scores > self.confidence_threshold
        scores = scores[filter_mask]
        classes = classes[filter_mask]
        boxes = boxes[filter_mask]
        texts = []
        if len(text_log_softmax):
            text_log_softmax = text_log_softmax[filter_mask]
            texts = np.argmax(text_log_softmax, 2)
            texts_confs = np.exp(np.max(text_log_softmax, 2))
            texts_confs = np.mean(texts_confs, axis=1)
            texts = [self.alphabet_decoder.decode(t) if texts_confs[i] > self.text_confidence_threshold else '' for i, t in enumerate(texts) ]

        if self.show_masks and segms is not None:
            segms = list(segm for segm, show in zip(segms, filter_mask) if show)
            result = self.overlay_masks(result, segms, classes, texts, ids)

        if self.show_boxes:
            result = self.overlay_boxes(result, boxes, classes, texts)

        result = self.overlay_texts(result, boxes, scores, texts, show_score=self.show_scores)

        return result

    def overlay_boxes(self, image, boxes, classes, show):
        colors = self.compute_colors_for_labels(classes).tolist()
        for box, color, shoud_show in zip(boxes, colors, show):
            if shoud_show:
                box = box.astype(int)
                top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
                image = cv2.rectangle(
                    image, tuple(top_left), tuple(bottom_right), tuple(color), 1
                )
        return image

    def overlay_masks(self, image, masks, classes, show, ids=None):
        colors = self.compute_colors_for_labels(classes).tolist()

        segments_image = image.copy()
        aggregated_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        aggregated_colored_mask = np.zeros(image.shape, dtype=np.uint8)
        black = np.zeros(3, dtype=np.uint8)

        for i, (mask, color, shoud_show) in enumerate(zip(masks, colors, show)):
            if shoud_show:
                mask = mask.astype(np.uint8)
                color_idx = i if ids is None else ids[i]
                mask_color = self.instance_color_palette[color_idx % len(self.instance_color_palette)].tolist()
                cv2.bitwise_or(aggregated_mask, mask, dst=aggregated_mask)
                cv2.bitwise_or(aggregated_colored_mask, np.asarray(mask_color, dtype=np.uint8),
                               dst=aggregated_colored_mask, mask=mask)

        # Fill the area occupied by all instances with a colored instances mask image.
        cv2.bitwise_and(segments_image, black, dst=segments_image, mask=aggregated_mask)
        cv2.bitwise_or(segments_image, aggregated_colored_mask, dst=segments_image, mask=aggregated_mask)
        # Blend original image with the one, where instances are colored.
        # As a result instances masks become transparent.
        cv2.addWeighted(image, 0.5, segments_image, 0.5, 0, dst=image)

        return image


    def overlay_texts(self, image, boxes, scores, texts, show_score=True):
        template = '{}: {:.2f}' if show_score else '{}'
        white = (255, 255, 255)

        for box, score, label in zip(boxes, scores, texts):
            if label:
                s = template.format(label, score)
                textsize = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                position = ((box[:2] + box[2:] - textsize) / 2).astype(int)
                cv2.putText(image, s, tuple(position), cv2.FONT_HERSHEY_SIMPLEX, .5, white, 1)

        return image