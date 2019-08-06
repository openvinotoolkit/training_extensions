# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from functools import partial

import tensorflow as tf
import numpy as np


def _clip_to_unit(value):
    """Clips input value into [0,1] range.

    :param value: Input value
    :return: Clipped value
    """

    return tf.maximum(0.0, tf.minimum(value, 1.0))


class AugmentFactory(object):
    """Class to define the set of augmentors with specified probability.
    """

    def __init__(self, independent_probs=True, name='IndependentAugmentFactory'):
        """Constructor.

        :param independent_probs: Whether to normalize input probabilities
        :param name: Class name
        """

        self.name = name
        self.independent_probs = independent_probs
        self.augmentors = []
        self.probs = []

    def _augment_images(self, images):
        """Carry out augmentation of image batch

        :param images: Batch of images
        :return: Augmented images
        """

        augm_images = images

        if self.independent_probs:
            for i in xrange(len(self.augmentors)):
                augmentor = self.augmentors[i]
                prob = self.probs[i]

                if prob is None:
                    augm_images = augmentor(augm_images)
                else:
                    uniform = tf.random_uniform([], 0., 1., dtype=tf.float32)
                    augm_images = tf.cond(tf.less(uniform, prob),
                                          lambda: augmentor(augm_images),  # pylint: disable=cell-var-from-loop
                                          lambda: augm_images)
        else:
            sum_probs = float(sum(self.probs))
            probs = [0.] + [float(p) / sum_probs for p in self.probs]
            prob_sums = np.cumsum(probs)

            uniform = tf.random_uniform([], 0., 1., dtype=tf.float32)
            for i in xrange(len(self.augmentors)):
                condition = tf.logical_and(tf.greater(uniform, prob_sums[i]),
                                           tf.less(uniform, prob_sums[i + 1]))
                augm_images = tf.cond(condition,
                                      lambda: self.augmentors[i](augm_images),
                                      lambda: augm_images)

        return augm_images

    def add(self, augmentor, prob=None):
        """Adds new augmentor function into set of augmentors.

        :param augmentor: New augmentor
        :param prob: Probability of augmentor call
        :return: New factory
        """

        if self.independent_probs:
            if prob is not None:
                assert 0. < prob < 1., 'prob must be in range (0; 1).'
        else:
            assert prob is not None, 'prob must be specified.'
            assert prob > 0., 'prob must be positive.'

        self.augmentors.append(augmentor)
        self.probs.append(prob)

        return self

    def __call__(self, float_images):
        """Carry out augmentation.

        :param float_images: Batch of images
        :return: Augmented images
        """

        assert float_images.dtype == tf.float32

        with tf.name_scope(self.name):
            augm_images = self._augment_images(float_images)
            augm_images = tf.clip_by_value(augm_images, 0.0, 1.0)
            augm_images = tf.stop_gradient(augm_images)
            return augm_images


class ContrastAugmentor(object):
    """Carry out augmentation of image contrast.
    """

    def __init__(self, lower, upper):
        """Constructor.

        :param lower: Lower border >= 0
        :param upper: Upper border
        """

        assert lower >= 0., 'lower must be non-negative.'
        assert upper > lower, 'upper must be > lower.'

        self.lower = lower
        self.upper = upper

    def __call__(self, float_images):
        """Carry out augmentation.

        :param float_images: Batch of images
        :return: Augmented images
        """

        return tf.image.random_contrast(float_images, self.lower, self.upper)


class BrightnessAugmentor(object):
    """Carry out augmentation of image brightness.
    """

    def __init__(self, delta):
        """Constructor.

        :param delta: Brightness delta parameter (>= 0)
        """

        assert delta >= 0., 'delta must be non-negative.'

        self.delta = delta

    def __call__(self, float_images):
        """Carry out augmentation.

        :param float_images: Batch of images
        :return: Augmented images
        """

        return tf.image.random_brightness(float_images, self.delta)


class SaturationAugmentor(object):
    """Carry out augmentation of image saturation.
    """

    def __init__(self, limits):
        """Constructor.

        :param limits: List of limits in format [lower, upper]
        """

        assert len(limits) == 2
        assert limits[0] >= 0., 'lower must be non-negative.'
        assert limits[1] > limits[0], 'upper must be > lower.'

        self.lower = limits[0]
        self.upper = limits[1]

    def __call__(self, float_images):
        """Carry out augmentation.

        :param float_images: Batch of images
        :return: Augmented images
        """

        return tf.image.random_saturation(float_images, self.lower, self.upper)


class GammaAugmentor(object):
    """Carry out Gamma augmentation of input image.
    """

    def __init__(self, delta, name='GammaAugmentor'):
        """Constructor.

        :param delta: Positive parameter
        :param name: Name of augmentor
        """

        assert delta > 0., 'delta must be positive.'

        self.delta = delta
        self.name = name

    def __call__(self, float_images):
        """Carry out augmentation.

        :param float_images: Batch of images
        :return: Augmented images
        """

        with tf.name_scope(self.name):
            uniform = tf.random_uniform([], -self.delta, self.delta)
            gamma = tf.log(0.5 + (2 ** (-0.5)) * uniform) / tf.log(0.5 - (2 ** (-0.5)) * uniform)
            return tf.pow(float_images, gamma)


class ClassifierAugmentation(object):
    """Classification-specified set of augmentations.
    """

    def __init__(self, prob, scale_limits, var_limits, brightness_delta, saturation_limits, trg_aspect_ratio):
        """Constructor.

        :param prob: Crop probability
        :param scale_limits: Limits of crop augmentation
        :param var_limits: Limits to shake scale limits of crop augmentation
        :param brightness_delta: Parameter for brightness augmentation
        :param saturation_limits: Parameters for saturation augmentation
        :param trg_aspect_ratio: Target image aspect ratio
        """

        self._prob = prob
        self._scale_limits = scale_limits
        self._var_limits = var_limits
        self._brightness_delta = brightness_delta
        self._saturation_limits = saturation_limits
        self._trg_aspect_ratio = trg_aspect_ratio

    @staticmethod
    def _crop_augmentation(input_image, prob, scale_limits, var_limits, trg_aspect_ratio):
        """Carry out crop augmentation.

        :param input_image: Image to crop
        :param prob: Probability to run crop augmnetation
        :param scale_limits: Scale limits
        :param var_limits: Variance limits
        :param trg_aspect_ratio: Target image aspect ratio
        :return: Augmented image
        """

        def _expand_to_aspect_ratio(in_height, in_width):
            src_aspect_ratio = tf.divide(in_height, in_width)
            out_height, out_width = tf.cond(tf.greater(src_aspect_ratio, trg_aspect_ratio),
                                            lambda: (in_height, tf.divide(in_height, trg_aspect_ratio)),
                                            lambda: (in_width * trg_aspect_ratio, in_width))
            return out_height, out_width

        def _crop_image():
            im_float_height = tf.cast(tf.shape(input_image)[0], tf.float32)
            im_float_width = tf.cast(tf.shape(input_image)[1], tf.float32)

            crop_scale = tf.random_uniform([], scale_limits[0], scale_limits[1], dtype=tf.float32)
            crop_height_var = tf.random_uniform([], var_limits[0], var_limits[1], dtype=tf.float32)
            crop_width_var = tf.random_uniform([], var_limits[0], var_limits[1], dtype=tf.float32)

            crop_height = im_float_height * tf.minimum(tf.maximum(0.0, crop_scale + crop_height_var), 1.0)
            crop_width = im_float_width * tf.minimum(tf.maximum(0.0, crop_scale + crop_width_var), 1.0)

            crop_height, crop_width = _expand_to_aspect_ratio(crop_height, crop_width)

            crop_int_height = tf.minimum(tf.cast(crop_height, tf.int32), tf.shape(input_image)[0])
            crop_int_width = tf.minimum(tf.cast(crop_width, tf.int32), tf.shape(input_image)[1])

            cropped_im = tf.image.random_crop(input_image, [crop_int_height, crop_int_width, 3])

            return cropped_im

        uniform = tf.random_uniform([], 0., 1., dtype=tf.float32)
        cropped_image = tf.cond(tf.less(uniform, prob), lambda: _crop_image(), lambda: input_image)

        return cropped_image

    @staticmethod
    def _image_augmentation(input_image, brightness_delta, saturation_limits):
        """Carry out set of image augmentations.

        :param input_image: Input image
        :param brightness_delta: Parameter for brightness augmentation
        :param saturation_limits: Parameters for saturation augmentation
        :return: Augmented image
        """

        blob = tf.image.random_flip_left_right(input_image)
        blob = tf.image.random_brightness(blob, max_delta=brightness_delta)
        blob = tf.image.random_saturation(blob, lower=saturation_limits[0], upper=saturation_limits[1])
        blob = tf.clip_by_value(blob, 0.0, 1.0)
        return blob

    def __call__(self, src_image):
        """Carry out augmentation

        :param src_image: Input image
        :return: Augmented image
        """

        image = self._crop_augmentation(src_image, self._prob, self._scale_limits,
                                        self._var_limits, self._trg_aspect_ratio)
        image = self._image_augmentation(image, self._brightness_delta, self._saturation_limits)
        return image


class DetectionAugmentation(object):
    """Detection-specified set of augmentations.
    """

    def __init__(self, free_prob, expand_prob, crop_prob, max_expand_ratio,
                 crop_scale_delta, crop_scale_limits, crop_shift_delta, crop_aspect_ratio):
        """Constructor.

        :param free_prob: Probability to preserve original image
        :param expand_prob: Probability to apply expand augmentation
        :param crop_prob: Probability to apply crop augmentation
        :param max_expand_ratio: Max ratio for expand augmentation
        :param crop_scale_delta: Delta parameter for crop augmentation
        :param crop_scale_limits: Scale limits for crop augmentation
        :param crop_shift_delta: Shift parameter for crop augmentation
        :param crop_aspect_ratio: Target aspect ratio for crop augmentation
        """

        assert 0.0 <= free_prob <= 1
        assert 0.0 <= expand_prob <= 1
        assert 0.0 <= crop_prob <= 1

        self.free_prob = free_prob
        self.expand_prob = expand_prob
        self.crop_prob = crop_prob

        assert max_expand_ratio > 1.0
        assert len(crop_scale_limits) == 2
        assert crop_scale_limits[0] < crop_scale_limits[1]
        assert crop_shift_delta > 0.0

        self.min_crop_size = 10

        self.expand_augmentor = partial(self._expand, max_ratio=max_expand_ratio)
        self.crop_augmentor = partial(self._crop, scale_delta=crop_scale_delta, scale_limits=crop_scale_limits,
                                      shift_delta=crop_shift_delta, trg_aspect_ratio=crop_aspect_ratio,
                                      min_crop_size=self.min_crop_size)

    @staticmethod
    def _left_right_flip(in_tuple, prob=0.5):
        """Carry out horizontal flip image augmentation.

        :param in_tuple: Image and annotation tuple
        :param prob: Probability to apply augmentation
        :return: Augmented image
        """

        def _process():
            flipped_image = tf.image.flip_left_right(in_tuple[0])

            bboxes = in_tuple[2]
            flipped_bboxes = tf.stack([bboxes[:, 0],
                                       1.0 - bboxes[:, 3],
                                       bboxes[:, 2],
                                       1.0 - bboxes[:, 1]], axis=1)

            return flipped_image, in_tuple[1], flipped_bboxes

        flipped_tuple = tf.cond(tf.less(tf.random_uniform([], 0., 1., dtype=tf.float32), prob),
                                lambda: _process(),
                                lambda: in_tuple)

        return flipped_tuple

    @staticmethod
    def _crop(in_tuple, scale_limits, shift_delta, scale_delta, trg_aspect_ratio, min_crop_size):
        """Carry out crop augmentation.

        :param in_tuple: Image and annotation tuple
        :param scale_limits: Crop limits
        :param shift_delta: Delta to shift crop
        :param scale_delta: Delta to shake crop
        :param trg_aspect_ratio: Target aspect ratio of image
        :param min_crop_size: Minimal size of cropped image
        :return: Augmented image
        """

        def _estimate_similarity(anchor_bbox, list_bboxes):
            anchor_center_y = 0.5 * (anchor_bbox[0] + anchor_bbox[2])
            anchor_center_x = 0.5 * (anchor_bbox[1] + anchor_bbox[3])

            rest_center_y = 0.5 * (list_bboxes[:, 0] + list_bboxes[:, 2])
            rest_center_x = 0.5 * (list_bboxes[:, 1] + list_bboxes[:, 3])

            distances = tf.squared_difference(anchor_center_y, rest_center_y) + \
                        tf.squared_difference(anchor_center_x, rest_center_x)

            return tf.negative(distances)

        def _estimate_box_limits(list_bboxes):
            return tf.stack([tf.reduce_min(list_bboxes[:, 0]),
                             tf.reduce_min(list_bboxes[:, 1]),
                             tf.reduce_max(list_bboxes[:, 2]),
                             tf.reduce_max(list_bboxes[:, 3])])

        def _get_support_bbox(max_num_factor=5):
            def _estimate_support_bbox(list_limits):
                num_limits = tf.shape(list_limits)[0]
                anchor_obj_id = tf.random_uniform([], 1, num_limits, tf.int32)
                similarity_to_all = _estimate_similarity(list_limits[anchor_obj_id], list_limits)

                max_support_size = tf.maximum(2, num_limits / max_num_factor)
                support_size = tf.random_uniform([], 1, max_support_size, dtype=tf.int32)
                _, support_obj_ids = tf.nn.top_k(similarity_to_all, k=support_size, sorted=False)

                support_obj = tf.gather(list_limits, support_obj_ids)
                support_limits = _estimate_box_limits(support_obj)

                return support_limits

            valid_obj_limits = tf.boolean_mask(in_tuple[2], tf.greater_equal(in_tuple[1], 0))
            list_size = tf.shape(valid_obj_limits)[0]
            return tf.cond(tf.equal(list_size, 1),
                           lambda: valid_obj_limits[0],
                           lambda: _estimate_support_bbox(valid_obj_limits))

        def _expand_to_aspect_ratio(ymin, xmin, ymax, xmax):
            height = ymax - ymin
            width = xmax - xmin
            src_aspect_ratio = tf.divide(height, width)

            center_y = 0.5 * (ymin + ymax)
            center_x = 0.5 * (xmin + xmax)

            out_h, out_w = tf.cond(tf.greater(src_aspect_ratio, trg_aspect_ratio),
                                   lambda: (height, tf.divide(height, trg_aspect_ratio)),
                                   lambda: (width * trg_aspect_ratio, width))

            out_ymin = _clip_to_unit(center_y - 0.5 * out_h)
            out_xmin = _clip_to_unit(center_x - 0.5 * out_w)
            out_ymax = _clip_to_unit(center_y + 0.5 * out_h)
            out_xmax = _clip_to_unit(center_x + 0.5 * out_w)

            return out_ymin, out_xmin, out_ymax, out_xmax

        def _is_valid_box(ymin, xmin, ymax, xmax):
            return tf.logical_and(tf.less(ymin, ymax), tf.less(xmin, xmax))

        def _process(roi_ymin, roi_xmin, roi_height, roi_width):
            src_image_height = tf.cast(tf.shape(in_tuple[0])[0], tf.float32)
            src_image_width = tf.cast(tf.shape(in_tuple[0])[1], tf.float32)
            cropped_image = tf.image.crop_to_bounding_box(in_tuple[0],
                                                          tf.cast(roi_ymin * src_image_height, tf.int32),
                                                          tf.cast(roi_xmin * src_image_width, tf.int32),
                                                          tf.cast(roi_height * src_image_height, tf.int32),
                                                          tf.cast(roi_width * src_image_width, tf.int32))

            obj_bboxes = in_tuple[2]
            cropped_obj_ymin = _clip_to_unit((obj_bboxes[:, 0] - roi_ymin) / roi_height)
            cropped_obj_xmin = _clip_to_unit((obj_bboxes[:, 1] - roi_xmin) / roi_width)
            cropped_obj_ymax = _clip_to_unit((obj_bboxes[:, 2] - roi_ymin) / roi_height)
            cropped_obj_xmax = _clip_to_unit((obj_bboxes[:, 3] - roi_xmin) / roi_width)

            valid_mask = tf.logical_and(_is_valid_box(cropped_obj_ymin, cropped_obj_xmin,
                                                      cropped_obj_ymax, cropped_obj_xmax),
                                        tf.greater_equal(in_tuple[1], 0))
            valid_labels = tf.where(valid_mask, in_tuple[1], tf.fill(tf.shape(valid_mask), -1))
            valid_cropped_obj_bboxes = tf.stack(
                [tf.where(valid_mask, cropped_obj_ymin, tf.zeros_like(cropped_obj_ymin)),
                 tf.where(valid_mask, cropped_obj_xmin, tf.zeros_like(cropped_obj_xmin)),
                 tf.where(valid_mask, cropped_obj_ymax, tf.zeros_like(cropped_obj_ymax)),
                 tf.where(valid_mask, cropped_obj_xmax, tf.zeros_like(cropped_obj_xmax))],
                axis=1)

            return cropped_image, valid_labels, valid_cropped_obj_bboxes

        support_bbox = _get_support_bbox()
        support_height = support_bbox[2] - support_bbox[0]
        support_width = support_bbox[3] - support_bbox[1]
        support_center_y = 0.5 * (support_bbox[0] + support_bbox[2])
        support_center_x = 0.5 * (support_bbox[1] + support_bbox[3])

        min_scale = tf.maximum(scale_limits[0] / support_height, scale_limits[0] / support_width)
        max_scale = tf.minimum(scale_limits[1] / support_height, scale_limits[1] / support_width)
        scale = tf.random_uniform([], min_scale, max_scale, dtype=tf.float32)
        scale_y = scale * tf.random_uniform([], 1.0 - scale_delta, 1.0 + scale_delta, dtype=tf.float32)
        scale_x = scale * tf.random_uniform([], 1.0 - scale_delta, 1.0 + scale_delta, dtype=tf.float32)

        crop_candidate_height = scale_y * support_height
        crop_candidate_width = scale_x * support_width

        shift_delta_y = shift_delta * crop_candidate_height
        shift_delta_x = shift_delta * crop_candidate_width

        shift_y = tf.random_uniform([], -shift_delta_y, shift_delta_y, dtype=tf.float32)
        shift_x = tf.random_uniform([], -shift_delta_x, shift_delta_x, dtype=tf.float32)

        crop_ymin = _clip_to_unit(support_center_y + shift_y - 0.5 * crop_candidate_height)
        crop_xmin = _clip_to_unit(support_center_x + shift_x - 0.5 * crop_candidate_width)
        crop_ymax = _clip_to_unit(support_center_y + shift_y + 0.5 * crop_candidate_height)
        crop_xmax = _clip_to_unit(support_center_x + shift_x + 0.5 * crop_candidate_width)

        crop_ymin, crop_xmin, crop_ymax, crop_xmax = \
            _expand_to_aspect_ratio(crop_ymin, crop_xmin, crop_ymax, crop_xmax)

        crop_height = _clip_to_unit(crop_ymax - crop_ymin)
        crop_width = _clip_to_unit(crop_xmax - crop_xmin)

        int_crop_height = tf.cast(crop_height * tf.cast(tf.shape(in_tuple[0])[0], tf.float32), tf.int32)
        int_crop_width = tf.cast(crop_width * tf.cast(tf.shape(in_tuple[0])[1], tf.float32), tf.int32)
        is_valid_crop = tf.logical_and(tf.greater(int_crop_height, min_crop_size),
                                       tf.greater(int_crop_width, min_crop_size))
        out_image, out_labels, out_objects = tf.cond(is_valid_crop,
                                                     lambda: _process(crop_ymin, crop_xmin, crop_height, crop_width),
                                                     lambda: in_tuple)

        return out_image, out_labels, out_objects

    @staticmethod
    def _expand(in_tuple, max_ratio):
        """Carry out expand augmentation.

        :param in_tuple: Image and annotation tuple
        :param max_ratio: Max ratio to expand image
        :return: Augmented image
        """

        src_image_shape = tf.shape(in_tuple[0])
        src_height = tf.cast(src_image_shape[0], tf.float32)
        src_width = tf.cast(src_image_shape[1], tf.float32)

        ratio = tf.random_uniform([], 1., float(max_ratio), dtype=tf.float32)
        trg_height = src_height * ratio
        trg_width = src_width * ratio

        offset_height = tf.floor(tf.random.uniform([], 0., trg_height - src_height, tf.float32))
        offset_width = tf.floor(tf.random.uniform([], 0., trg_width - src_width, tf.float32))

        shift_y = offset_height / trg_height
        shift_x = offset_width / trg_width
        shift = tf.reshape([shift_y, shift_x, shift_y, shift_x], [1, 4])
        scale = tf.reciprocal(ratio)

        expanded_image = tf.image.pad_to_bounding_box(in_tuple[0],
                                                      tf.cast(offset_height, tf.int32),
                                                      tf.cast(offset_width, tf.int32),
                                                      tf.cast(trg_height, tf.int32),
                                                      tf.cast(trg_width, tf.int32))
        expanded_bboxes = shift + scale * in_tuple[2]

        return expanded_image, in_tuple[1], expanded_bboxes

    def _spatial_transform(self, in_tuple, branch_prob):
        """Carry out expand and crop augmentation according probability.

        :param in_tuple: Image and annotation tuple
        :param branch_prob: Overall probability to apply at least one augmentation
        :return: Augmented image
        """

        expand_prob = self.expand_prob / branch_prob
        expanded_tuple = tf.cond(tf.less(tf.random_uniform([], 0., 1., dtype=tf.float32), expand_prob),
                                 lambda: self.expand_augmentor(in_tuple),
                                 lambda: in_tuple)

        crop_prob = self.crop_prob / branch_prob
        cropped_tuple = tf.cond(tf.less(tf.random_uniform([], 0., 1., dtype=tf.float32), crop_prob),
                                lambda: self.crop_augmentor(expanded_tuple),
                                lambda: expanded_tuple)

        return cropped_tuple

    def __call__(self, src_images, src_labels, src_bboxes):
        """Carry out image augmentation with according annotation.

        :param src_images: Input images
        :param src_labels: Input detection classes
        :param src_bboxes: Input detection boxes
        :return: Augmented images
        """

        augmented_tuple = src_images, src_labels, src_bboxes

        augmented_tuple = tf.cond(tf.less(tf.random_uniform([], 0., 1., dtype=tf.float32), self.free_prob),
                                  lambda: augmented_tuple,
                                  lambda: self._spatial_transform(augmented_tuple, 1.0 - self.free_prob))

        augmented_tuple = self._left_right_flip(augmented_tuple)

        return augmented_tuple
