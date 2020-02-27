"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import copy
import random

import cv2
import numpy as np


class RandomRotate90:
    """ Class for training sample random rotation by 90, 180, 270 degrees. """

    @staticmethod
    def rotate_image(image, angle):
        """ Rotates an image (angle in degrees) and expands image to avoid cropping. """

        height, width = image.shape[:2]
        image_center = (width / 2, height / 2)

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        image = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))

        return image

    @staticmethod
    def rotate_point_by_90(x_coord, y_coord, width, height, rotate_by_90_k_times):
        """ Rotate point by 90 degrees (clockwise). """

        cos = [1.0, 0.0, -1.0, 0.0]
        sin = [0.0, -1.0, 0.0, 1.0]

        x1_coord = x_coord - 0.5 * width
        y1_coord = y_coord - 0.5 * height

        if rotate_by_90_k_times % 2 == 1:
            width, height = height, width

        x_coord = x1_coord * cos[rotate_by_90_k_times] - y1_coord * sin[
            rotate_by_90_k_times] + 0.5 * width
        y_coord = x1_coord * sin[rotate_by_90_k_times] + y1_coord * cos[
            rotate_by_90_k_times] + 0.5 * height

        return x_coord, y_coord

    def __call__(self, sample):
        angle = random.choice([0, 90, 270])
        if angle:
            height, width = sample['image'].shape[:2]
            sample['image'] = self.rotate_image(sample['image'], angle)

            # Rotate boxes.
            if 'gt_boxes' in sample:
                boxes = sample['gt_boxes']
                boxes[:, 0], boxes[:, 1] = self.rotate_point_by_90(boxes[:, 0], boxes[:, 1],
                                                                   width, height, angle // 90)
                boxes[:, 2], boxes[:, 3] = self.rotate_point_by_90(boxes[:, 2], boxes[:, 3],
                                                                   width, height, angle // 90)

                boxes[:, 0], boxes[:, 2] = np.minimum(boxes[:, 0], boxes[:, 2]), np.maximum(
                    boxes[:, 0], boxes[:, 2])
                boxes[:, 1], boxes[:, 3] = np.minimum(boxes[:, 1], boxes[:, 3]), np.maximum(
                    boxes[:, 1], boxes[:, 3])

                sample['gt_boxes'] = boxes

            # Rotate masks.
            if 'gt_masks' in sample:
                polygons = sample['gt_masks']
                for i, obj in enumerate(polygons):
                    for j, _ in enumerate(obj):
                        polygons[i][j][:, 0], polygons[i][j][:, 1] = self.rotate_point_by_90(
                            polygons[i][j][:, 0], polygons[i][j][:, 1], width, height, angle // 90
                        )

                sample['gt_masks'] = polygons

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string


class ColorJitter:
    """ Class for training sample color jittering. """

    @staticmethod
    def adjust_brightness(image, value):
        ''' Adjusts brightness of input image by value. '''

        return np.clip(image + value, 0.0, 1.0)

    def random_brightness(self, image):
        ''' Adjusts brightness of input image by random value. '''

        max_delta = 32. / 255.
        value = random.uniform(-max_delta, max_delta)
        return self.adjust_brightness(image, value)

    @staticmethod
    def adjust_contrast(image, value):
        ''' Adjusts contrast of input image by value. '''

        return np.clip(image * value, 0.0, 1.0)

    def random_contrast(self, image):
        ''' Adjusts contrast of input image by random value. '''

        lower = 0.5
        upper = 1.5
        value = random.uniform(lower, upper)
        return self.adjust_contrast(image, value)

    @staticmethod
    def adjust_hue(image, value):
        ''' Adjusts hue of input image by value. '''

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 0] += value * 255.0
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return np.clip(image, 0.0, 1.0)

    def random_hue(self, image):
        ''' Adjusts hue of input image by random value. '''

        max_delta = 0.2
        value = random.uniform(-max_delta, max_delta)
        return self.adjust_hue(image, value)

    @staticmethod
    def adjust_saturation(image, value):
        ''' Adjusts saturation of input image by value. '''

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 1] *= value
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return np.clip(image, 0.0, 1.0)

    def random_saturation(self, image):
        ''' Adjusts saturation of input image by random value. '''

        lower = 0.5
        upper = 1.5
        value = random.uniform(lower, upper)
        return self.adjust_saturation(image, value)

    def __call__(self, sample):
        assert sample['image'].dtype == np.uint8, sample['image'].dtype

        sample['image'] = sample['image'].astype(np.float32) / 255.0

        transforms = [
            self.random_brightness,
            self.random_contrast,
            self.random_saturation,
            self.random_hue
        ]

        random.shuffle(transforms)

        for transform in transforms:
            sample['image'] = transform(sample['image'])

        sample['image'] = (sample['image'] * 255.0).astype(np.uint8)

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string


class AlphabetDecodeTransform:
    """ Class for applying alphabet encoding. """

    def __init__(self, alphabet_decoder):
        self.decoder = alphabet_decoder

    def __call__(self, sample):
        if 'gt_texts' not in sample:
            return sample
        gt_texts = sample['gt_texts']
        assert isinstance(gt_texts, list)
        sample['gt_texts_str'] = copy.deepcopy(gt_texts)
        sample['gt_texts'] = self.decoder.encode_batch(gt_texts)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += f"[alphabet = '{self.decoder.alphabet}',sos = '{self.decoder.sos}']"
        return format_string


class FilterTextByArea:
    """ Class for filtering out too small text instances. """

    def __init__(self, min_area):
        self.min_area = min_area

    def __call__(self, sample):
        if 'gt_texts' not in sample:
            return sample
        gt_texts = sample['gt_texts']
        assert isinstance(gt_texts, list)

        for i, _ in enumerate(zip(sample['gt_texts'])):
            if cv2.contourArea(sample['gt_masks'][i][0].astype(np.int32)) < self.min_area:
                sample['gt_texts'][i] = None

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += f"[min_area = '{self.min_area}']"
        return format_string


class FilterTextByLength:
    """ Class for filtering out too short text instances. """

    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, sample):
        if 'gt_texts' not in sample:
            return sample
        gt_texts = sample['gt_texts']
        assert isinstance(gt_texts, list)

        for i, _ in enumerate(zip(sample['gt_texts'])):
            if sample['gt_texts'][i] is not None and len(sample['gt_texts'][i]) > self.max_length:
                sample['gt_texts'][i] = None

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += f"[max_length = '{self.max_length}']"
        return format_string


class RandomCrop:
    """ Class for random cropping. """

    OUTSIDE = 0
    INSIDE = 1
    ON_BORDER = 2

    def __init__(self, min_relative_size, attempts):
        self.min_relative_size = min_relative_size
        self.attempts = attempts

    @staticmethod
    def position(box, crop_height, crop_width):
        if box[0] <= 0 and box[1] <= 0 and box[2] <= 0 and box[3] <= 0:
            return RandomCrop.OUTSIDE
        elif box[0] >= crop_width and box[1] >= crop_height and box[2] >= crop_width and box[
            3] >= crop_height:
            return RandomCrop.OUTSIDE
        elif box[0] >= 0 and box[1] >= 0 and box[2] < crop_width and box[3] < crop_height:
            return RandomCrop.INSIDE
        else:
            return RandomCrop.ON_BORDER

    def __call__(self, sample):
        height, width = sample['image'].shape[:2]

        for attempt in range(self.attempts):
            crop_height = random.randint(int(self.min_relative_size * height), height)
            crop_width = random.randint(int(self.min_relative_size * width), width)

            crop_min_y = random.randint(0, height - crop_height)
            crop_min_x = random.randint(0, width - crop_width)

            if 'gt_boxes' in sample:
                boxes = np.copy(sample['gt_boxes'])
                boxes[:, 0] -= crop_min_x
                boxes[:, 1] -= crop_min_y
                boxes[:, 2] -= crop_min_x
                boxes[:, 3] -= crop_min_y

                positions = np.array(
                    [RandomCrop.position(box, crop_height, crop_width) for box in boxes])
                if RandomCrop.ON_BORDER in positions:
                    continue
                if RandomCrop.INSIDE not in positions:
                    continue

                sample['gt_texts'] = np.array(sample['gt_texts'])[positions == RandomCrop.INSIDE]
                sample['gt_texts'] = list(sample['gt_texts'])
                sample['gt_boxes'] = boxes[positions == RandomCrop.INSIDE]
                sample['gt_masks'] = np.array(sample['gt_masks'])[positions == RandomCrop.INSIDE]
                sample['gt_classes'] = sample['gt_classes'][positions == RandomCrop.INSIDE]

                polygons = list(sample['gt_masks'])
                for i, obj in enumerate(polygons):
                    for j, _ in enumerate(obj):
                        polygons[i][j][:, 0] -= crop_min_x
                        polygons[i][j][:, 1] -= crop_min_y

                sample['gt_masks'] = polygons

                sample['image'] = sample['image'][crop_min_y:crop_min_y + crop_height,
                                  crop_min_x:crop_min_x + crop_width]

                break

        return sample


class Visualize:
    """ Visualizes image with rectangles, masks, texts using cv2.imshow. """

    def __init__(self, delay):
        self.delay = delay

    def __call__(self, sample):
        image = sample['image'].copy()
        for box, mask, text in zip(sample['gt_boxes'], sample['gt_masks'], sample['gt_texts']):
            xmin, ymin, xmax, ymax = [int(x) for x in box]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
            cv2.drawContours(image, np.array(mask).astype(np.int32), -1, (255, 255, 255), 2)
            cv2.putText(image, text, (xmin, ymin), 1, 1, (255, 255, 255), 2)
        cv2.imshow('image', image)
        cv2.waitKey(self.delay)

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += f"[delay = '{self.delay}']"
        return format_string
