import random
import copy

import cv2
import numpy as np


class ConvertKeypoints(object):
    def __call__(self, sample):
        label = sample['label']
        h, w, _ = sample['image'].shape
        keypoints = label['keypoints']
        for keypoint in keypoints:  # keypoint[2] == 0: occluded, == 1: visible, == 2: not in image
            if keypoint[0] == keypoint[1] == 0:
                keypoint[2] = 2
            if (keypoint[0] < 0
                    or keypoint[0] >= w
                    or keypoint[1] < 0
                    or keypoint[1] >= h):
                keypoint[2] = 2
        for other_label in label['processed_other_annotations']:
            keypoints = other_label['keypoints']
            for keypoint in keypoints:
                if keypoint[0] == keypoint[1] == 0:
                    keypoint[2] = 2
                if (keypoint[0] < 0
                        or keypoint[0] >= w
                        or keypoint[1] < 0
                        or keypoint[1] >= h):
                    keypoint[2] = 2
        label['keypoints'] = self._convert(label['keypoints'], w, h)

        for other_label in label['processed_other_annotations']:
            other_label['keypoints'] = self._convert(other_label['keypoints'], w, h)
        return sample

    def _convert(self, keypoints, w, h):
        # Nose, Neck, R hand, L hand, R leg, L leg, Eyes, Ears
        reorder_map = [1, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
        converted_keypoints = list(keypoints[i - 1] for i in reorder_map)
        converted_keypoints.insert(1, [(keypoints[5][0] + keypoints[6][0]) / 2,
                                       (keypoints[5][1] + keypoints[6][1]) / 2, 0])  # Add neck as a mean of shoulders
        if keypoints[5][2] == 2 and keypoints[6][2] == 2:
            converted_keypoints[1][2] = 2
        elif keypoints[5][2] == 3 and keypoints[6][2] == 3:
            converted_keypoints[1][2] = 3
        elif keypoints[5][2] == 1 and keypoints[6][2] == 1:
            converted_keypoints[1][2] = 1
        if (converted_keypoints[1][0] < 0
                or converted_keypoints[1][0] >= w
                or converted_keypoints[1][1] < 0
                or converted_keypoints[1][1] >= h):
            converted_keypoints[1][2] = 2
        return converted_keypoints


class Scale(object):
    def __init__(self, prob=1, min_scale=0.5, max_scale=1.1, target_dist=0.6):
        self._prob = prob
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._target_dist = target_dist

    def __call__(self, sample):
        prob = random.random()
        scale_multiplier = 1
        if prob <= self._prob:
            prob = random.random()
            scale_multiplier = (self._max_scale - self._min_scale) * prob + self._min_scale
        label = sample['label']
        scale_abs = self._target_dist / label['scale_provided']
        scale = scale_abs * scale_multiplier
        sample['image'] = cv2.resize(sample['image'], dsize=(0, 0), fx=scale, fy=scale)
        label['img_height'], label['img_width'], _ = sample['image'].shape
        sample['mask'] = cv2.resize(sample['mask'], dsize=(0, 0), fx=scale, fy=scale)

        label['objpos'][0] *= scale
        label['objpos'][1] *= scale
        for keypoint in sample['label']['keypoints']:
            keypoint[0] *= scale
            keypoint[1] *= scale
        for other_annotation in sample['label']['processed_other_annotations']:
            other_annotation['objpos'][0] *= scale
            other_annotation['objpos'][1] *= scale
            for keypoint in other_annotation['keypoints']:
                keypoint[0] *= scale
                keypoint[1] *= scale
        return sample


class Rotate(object):
    def __init__(self, pad, max_rotate_degree=40):
        self._pad = pad
        self._max_rotate_degree = max_rotate_degree

    def __call__(self, sample):
        prob = random.random()
        degree = (prob - 0.5) * 2 * self._max_rotate_degree
        h, w, _ = sample['image'].shape
        img_center = (w / 2, h / 2)
        R = cv2.getRotationMatrix2D(img_center, degree, 1)

        abs_cos = abs(R[0, 0])
        abs_sin = abs(R[0, 1])

        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)
        dsize = (bound_w, bound_h)

        R[0, 2] += dsize[0] / 2 - img_center[0]
        R[1, 2] += dsize[1] / 2 - img_center[1]
        sample['image'] = cv2.warpAffine(sample['image'], R, dsize=dsize,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=self._pad)
        sample['label']['img_height'], sample['label']['img_width'], _ = sample['image'].shape
        sample['mask'] = cv2.warpAffine(sample['mask'], R, dsize=dsize,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))  # border is ok
        label = sample['label']
        label['objpos'] = self._rotate(label['objpos'], R)
        for keypoint in label['keypoints']:
            point = [keypoint[0], keypoint[1]]
            point = self._rotate(point, R)
            keypoint[0], keypoint[1] = point[0], point[1]
        for other_annotation in label['processed_other_annotations']:
            for keypoint in other_annotation['keypoints']:
                point = [keypoint[0], keypoint[1]]
                point = self._rotate(point, R)
                keypoint[0], keypoint[1] = point[0], point[1]
        return sample

    def _rotate(self, point, R):
        return [R[0, 0] * point[0] + R[0, 1] * point[1] + R[0, 2],
                R[1, 0] * point[0] + R[1, 1] * point[1] + R[1, 2]]


class CropPad(object):
    def __init__(self, pad, center_perterb_max=40, crop_x=368, crop_y=368):
        self._pad = pad
        self._center_perterb_max = center_perterb_max
        self._crop_x = crop_x
        self._crop_y = crop_y

    def __call__(self, sample):
        prob_x = random.random()
        prob_y = random.random()

        offset_x = int((prob_x - 0.5) * 2 * self._center_perterb_max)
        offset_y = int((prob_y - 0.5) * 2 * self._center_perterb_max)
        label = sample['label']
        shifted_center = (label['objpos'][0] + offset_x, label['objpos'][1] + offset_y)
        offset_left = -int(shifted_center[0] - self._crop_x / 2)
        offset_up = -int(shifted_center[1] - self._crop_y / 2)

        cropped_image = np.empty(shape=(self._crop_y, self._crop_x, 3), dtype=np.uint8)
        for i in range(3):
            cropped_image[:, :, i].fill(self._pad[i])
        cropped_mask = np.empty(shape=(self._crop_y, self._crop_x), dtype=np.uint8)
        cropped_mask.fill(1)

        image_x_start = int(shifted_center[0] - self._crop_x / 2)
        image_y_start = int(shifted_center[1] - self._crop_y / 2)
        image_x_finish = image_x_start + self._crop_x
        image_y_finish = image_y_start + self._crop_y
        crop_x_start = 0
        crop_y_start = 0
        crop_x_finish = self._crop_x
        crop_y_finish = self._crop_y

        w, h = label['img_width'], label['img_height']
        should_crop = True
        if image_x_start < 0:  # Adjust crop area
            crop_x_start -= image_x_start
            image_x_start = 0
        if image_x_start >= w:
            should_crop = False

        if image_y_start < 0:
            crop_y_start -= image_y_start
            image_y_start = 0
        if image_y_start >= w:
            should_crop = False

        if image_x_finish > w:
            diff = image_x_finish - w
            image_x_finish -= diff
            crop_x_finish -= diff
        if image_x_finish < 0:
            should_crop = False

        if image_y_finish > h:
            diff = image_y_finish - h
            image_y_finish -= diff
            crop_y_finish -= diff
        if image_y_finish < 0:
            should_crop = False

        if should_crop:
            cropped_image[crop_y_start:crop_y_finish, crop_x_start:crop_x_finish, :] =\
                sample['image'][image_y_start:image_y_finish, image_x_start:image_x_finish, :]
            cropped_mask[crop_y_start:crop_y_finish, crop_x_start:crop_x_finish] =\
                sample['mask'][image_y_start:image_y_finish, image_x_start:image_x_finish]

        sample['image'] = cropped_image
        sample['mask'] = cropped_mask
        label['img_width'] = self._crop_x
        label['img_height'] = self._crop_y

        label['objpos'][0] += offset_left
        label['objpos'][1] += offset_up
        for keypoint in label['keypoints']:
            keypoint[0] += offset_left
            keypoint[1] += offset_up
        for other_annotation in label['processed_other_annotations']:
            for keypoint in other_annotation['keypoints']:
                keypoint[0] += offset_left
                keypoint[1] += offset_up

        return sample

    def _inside(self, point, width, height):
        if point[0] < 0 or point[1] < 0:
            return False
        if point[0] >= width or point[1] >= height:
            return False
        return True


class Flip(object):
    def __init__(self, prob=0.5):
        self._prob = prob

    def __call__(self, sample):
        prob = random.random()
        do_flip = prob <= self._prob
        if not do_flip:
            return sample

        sample['image'] = cv2.flip(sample['image'], 1)
        sample['mask'] = cv2.flip(sample['mask'], 1)

        label = sample['label']
        w, h = label['img_width'], label['img_height']
        label['objpos'][0] = w - 1 - label['objpos'][0]
        for keypoint in label['keypoints']:
            keypoint[0] = w - 1 - keypoint[0]
        label['keypoints'] = self._swap_left_right(label['keypoints'])

        for other_annotation in label['processed_other_annotations']:
            other_annotation['objpos'][0] = w - 1 - other_annotation['objpos'][0]
            for keypoint in other_annotation['keypoints']:
                keypoint[0] = w - 1 - keypoint[0]
            other_annotation['keypoints'] = self._swap_left_right(other_annotation['keypoints'])

        return sample

    def _swap_left_right(self, keypoints):
        right = [2, 3, 4, 8, 9, 10, 14, 16]
        left = [5, 6, 7, 11, 12, 13, 15, 17]
        for r, l in zip(right, left):
            keypoints[r], keypoints[l] = keypoints[l], keypoints[r]
        return keypoints


class SinglePersonBodyMasking(object):
    def __init__(self, prob=0.5, percentage=0.3, mask_color=(128, 128, 128)):
        self._prob = prob
        self._percentage = percentage
        self._mask_color = mask_color

    def __call__(self, sample):
        image = sample['image']
        h, w, c = image.shape
        if random.random() > self._prob:
            center_x = random.randint(w // 3, w - 1 - w // 3)
            center_y = random.randint(h // 3, h - 1 - h // 3)
            kpt = [
                [center_x - random.randint(1, int(w * self._percentage)), center_y - random.randint(1, int(h * self._percentage))],
                [center_x + random.randint(1, int(w * self._percentage)), center_y - random.randint(1, int(h * self._percentage))],
                [center_x + random.randint(1, int(w * self._percentage)), center_y + random.randint(1, int(h * self._percentage))],
                [center_x - random.randint(1, int(w * self._percentage)), center_y + random.randint(1, int(h * self._percentage))]
            ]
            cv2.fillConvexPoly(image, np.array(kpt, dtype=np.int32), (128, 128, 128))
        return sample


class SinglePersonFlip(object):
    def __init__(self, left_keypoints_indice, right_keypoints_indice, prob=0.5):
        self._left_keypoints_indice = left_keypoints_indice
        self._right_keypoints_indice = right_keypoints_indice
        self._prob = prob

    def __call__(self, sample):
        rand = random.random()
        do_flip = rand <= self._prob
        if not do_flip:
            return sample

        sample['image'] = cv2.flip(sample['image'], 1)

        w, h = sample['image'].shape[1], sample['image'].shape[0]
        for id in range(len(sample['keypoints']) // 3):
            if sample['keypoints'][id * 3] == -1:
                continue
            sample['keypoints'][id * 3] = w - 1 - sample['keypoints'][id * 3]
        self._swap_left_right(sample['keypoints'])

        return sample

    def _swap_left_right(self, keypoints):
        keypoints[self._right_keypoints_indice + self._left_keypoints_indice] =\
            keypoints[self._left_keypoints_indice + self._right_keypoints_indice]


class ChannelPermutation(object):
    def __init__(self, prob=0.5):
        self._prob = prob

    def __call__(self, sample):
        rand = random.random()
        do_cp = rand < self._prob
        if not do_cp:
            return sample

        new_order = np.random.permutation(3)
        image = sample['image']
        image[:, :, 0], image[:, :, 1], image[:, :, 2] =\
            image[:, :, new_order[0]], image[:, :, new_order[1]], image[:, :, new_order[2]]
        sample['image'] = image

        return sample


class SinglePersonRotate(object):
    def __init__(self, pad=(128, 128, 128), max_rotate_degree=40):
        self._pad = pad
        self._max_rotate_degree = max_rotate_degree

    def __call__(self, sample):
        prob = random.random()
        degree = (prob - 0.5) * 2 * self._max_rotate_degree
        h, w, _ = sample['image'].shape
        img_center = (w / 2, h / 2)
        R = cv2.getRotationMatrix2D(img_center, degree, 1)

        abs_cos = abs(R[0, 0])
        abs_sin = abs(R[0, 1])

        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)
        dsize = (bound_w, bound_h)

        R[0, 2] += dsize[0] / 2 - img_center[0]
        R[1, 2] += dsize[1] / 2 - img_center[1]
        sample['image'] = cv2.warpAffine(sample['image'], R, dsize=dsize,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=self._pad)

        for id in range(len(sample['keypoints']) // 3):
            if sample['keypoints'][id * 3] == -1:
                continue
            point = (sample['keypoints'][id * 3], sample['keypoints'][id * 3 + 1])
            point = self._rotate(point, R)
            sample['keypoints'][id * 3], sample['keypoints'][id * 3 + 1] = point
        return sample

    def _rotate(self, point, R):
        return (R[0, 0] * point[0] + R[0, 1] * point[1] + R[0, 2],
                R[1, 0] * point[0] + R[1, 1] * point[1] + R[1, 2])


class SinglePersonCropPad(object):
    def __init__(self, pad, crop_x=256, crop_y=256):
        self._pad = pad
        self._crop_x = crop_x
        self._crop_y = crop_y

    def __call__(self, sample):
        img = sample['image']
        rnd_scale = 1
        rnd_offset_x = 0
        rnd_offset_y = 0

        if random.random() > 0.5:
            rnd_scale = random.random() * 0.7 + 0.8
            h, w, _ = img.shape
            scaled_img = cv2.resize(img, dsize=None, fx=rnd_scale, fy=rnd_scale, interpolation=cv2.INTER_CUBIC)
            sh, sw, _ = scaled_img.shape
            if rnd_scale >= 1:  # random crop from upsampled image
                rnd_offset_x = (sw - w) // 2
                rnd_offset_y = (sh - h) // 2
                img = scaled_img[rnd_offset_y:rnd_offset_y + h, rnd_offset_x:rnd_offset_x + w]
                rnd_offset_x *= -1
                rnd_offset_y *= -1
            else:  # pad to original size
                rnd_offset_x = (w - sw) // 2
                rnd_offset_y = (h - sh) // 2
                b_border = h - sh - rnd_offset_y
                r_border = w - sw - rnd_offset_x
                img = cv2.copyMakeBorder(scaled_img, rnd_offset_y, b_border, rnd_offset_x, r_border,
                                         borderType=cv2.BORDER_CONSTANT, value=self._pad)

        scale = self._crop_x / max(img.shape[0], img.shape[1])
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        offset_x = (self._crop_x - img.shape[1]) // 2
        offset_y = (self._crop_y - img.shape[0]) // 2

        padded_img = np.ones((self._crop_y, self._crop_x, 3), dtype=np.uint8) * self._pad
        padded_img[offset_y:offset_y+img.shape[0], offset_x:offset_x+img.shape[1], :] = img
        sample['image'] = padded_img

        for id in range(len(sample['keypoints']) // 3):
            if sample['keypoints'][id * 3] == -1:
                continue
            sample['keypoints'][id * 3] = (sample['keypoints'][id * 3] * rnd_scale + rnd_offset_x) * scale + offset_x
            sample['keypoints'][id * 3 + 1] = (sample['keypoints'][id * 3 + 1] * rnd_scale + rnd_offset_y) * scale + offset_y

        return sample


class RandomScaleRotate(object):
    def __init__(self, scale=0.35, rotate=45):
        self._scale = scale
        self._rotate = rotate

    def __call__(self, sample):
        scale = sample['scale']
        rotate = sample['rotate']

        scale = scale * np.clip(np.random.normal(loc=1, scale=scale) + 1, 1 - scale, 1 + scale)
        rotate = np.clip(np.random.normal(loc=1, scale=rotate), -rotate * 2, rotate * 2)\
              if np.random.uniform() <= 0.6 else 0

        sample['scale'] = scale
        sample['rotate'] = rotate

        return sample


class SinglePersonRandomAffineTransform(object):
    def __init__(self, scale=0.35, rotate=45, mode='train', input_width=288, input_height=384, stride=8, num_keypoints=17):
        self._mode = mode
        self._scale = scale
        self._rotate = rotate
        self._width = input_width
        self._height = input_height
        self._stride = stride
        self._num_keypoints = num_keypoints

    def __call__(self, sample):
        scale = sample['scale']
        center = sample['center']
        rotate = sample['rotate']

        trans, _ = self._get_transformation_matrix(center, scale, rotate, [self._width, self._height])
        input = cv2.warpAffine(sample['image'], trans, (self._width, self._height), flags=cv2.INTER_LINEAR)
        sample['trans'] = trans
        if self._mode == 'train':
            for id in range(self._num_keypoints):
                sample['keypoints'][3 * id: 3 * id + 2] = self._affine_transform(sample['keypoints'][3 * id: 3 * id + 2], trans)
        else:
            sample['rev_trans'] = self._get_transformation_matrix(center, scale, rotate,
                                                                  [self._width // self._stride, self._height // self._stride])[1]
        sample['image'] = input

        return sample

    def _get_transformation_matrix(self, center, scale, rotate, output_size):
        w, _ = scale * 200
        shift_y = self._rotation([0, -w * 0.5], rotate)
        shift_x = self._rotation([-w * 0.5, 0], rotate)

        points = np.array([center, center + shift_x, center + shift_y], dtype=np.float32)
        transformed_points = np.array([[output_size[0] * 0.5, output_size[1] * 0.5],
                                       [0, output_size[1] * 0.5],
                                       [output_size[0] * 0.5, output_size[1] * 0.5 - output_size[0] * 0.5]],
                                       dtype=np.float32)

        rev_trans = cv2.getAffineTransform(np.float32(transformed_points), np.float32(points))

        trans = cv2.getAffineTransform(np.float32(points), np.float32(transformed_points))

        return trans, rev_trans

    @staticmethod
    def _affine_transform(pt, t):
        new_pt = np.array([pt[0], pt[1], 1.])
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    @staticmethod
    def _rotation(point, r):
        r = np.pi * r / 180
        return [point[0] * np.cos(r) - point[1] * np.sin(r), point[0] * np.sin(r) + point[1] * np.cos(r)]


class HalfBodyTransform(object):
    def __init__(self, aspect_ratio=0.75, prob=0.3, num_keypoints=17):
        self._prob = prob
        self.aspect_ratio = aspect_ratio
        self._num_keypoints = num_keypoints
    def __call__(self, sample):
        rand = np.random.uniform()
        do_body_transform = rand <= self._prob and np.sum(sample['keypoints'][2:][::3]) > self._num_keypoints
        if not do_body_transform:
            return sample

        upper_points = []
        lower_points = []
        for idx in range(len(sample['keypoints']) // 3):
            if sample['keypoints'][idx * 3 + 2] > 0:
                if idx < 11:
                    upper_points.append([sample['keypoints'][idx * 3], sample['keypoints'][idx * 3 + 1]])
                else:
                    lower_points.append([sample['keypoints'][idx * 3], sample['keypoints'][idx * 3 + 1]])

        if np.random.uniform() < 0.5 and len(upper_points) > 2 or len(lower_points) < 2:
            target_points = upper_points
        else:
            target_points = lower_points

        if len(target_points) < 2:
            center, scale = None, None
        else:
            target_points = np.array(target_points, dtype=np.float32)
            center = np.array([target_points[:, 0].sum() / len(target_points),
                               target_points[:, 1].sum() / len(target_points)], dtype=np.float32)

            w = target_points[:, 0].max() - target_points[:, 0].min()
            h = target_points[:, 1].max() - target_points[:, 1].min()

            if w > self.aspect_ratio * h:
                h = w / self.aspect_ratio
            elif w < self.aspect_ratio * h:
                w = h * self.aspect_ratio

            scale = np.array([w / 200, h / 200], dtype=np.float32)
            scale *= 1.5

        if center is not None:
            sample['center'] = center
            sample['scale'] = scale

        return sample


class Normalization(object):
    def __init__(self, mean=0.5, std=1):
        self._mean = mean
        self._std = std

    def __call__(self, sample):
        sample['image'] = sample['image'] / 255
        sample['image'] = (sample['image'] - self._mean) / self._std

        return sample
