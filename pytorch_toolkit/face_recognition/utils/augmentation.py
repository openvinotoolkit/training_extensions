"""
 Copyright (c) 2018 Intel Corporation
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

import math
import torch
import numpy as np
import cv2 as cv

try:
    from .face_align import FivePointsAligner
except (ImportError, SystemError) as exp:
    from face_align import FivePointsAligner


class HorizontalFlipNumpy:
    """Horizontal flip augmentation with probability p"""
    def __init__(self, p=.5):
        assert 0 <= p <= 1.
        self.p = p

    def __call__(self, img):
        if float(torch.FloatTensor(1).uniform_()) < self.p:
            return cv.flip(img, 1)
        return img


class ShowTransform:
    """Show image using opencv"""
    def __call__(self, sample):
        img = np.array(sample)
        cv.imshow('image', img)
        cv.waitKey()
        return sample


class NumpyToTensor:
    """Converts a numpy array to torch.Tensor with optionally swapping R and B channels"""
    def __init__(self, switch_rb=False):
        self.switch_rb = switch_rb

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if self.switch_rb:
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).type(torch.FloatTensor) / 255.


class RandomShiftNumpy:
    """Shifts an image by a randomly generated offset along x and y axes"""
    def __init__(self, max_rel_shift, p=.5):
        self.p = p
        self.max_rel_shift = max_rel_shift

    def __call__(self, image):
        if float(torch.FloatTensor(1).uniform_()) < self.p:
            rel_shift = 2 * (torch.FloatTensor(1).uniform_() - .5) * self.max_rel_shift
            h, w = image.shape[:2]
            shift_w = w * rel_shift
            shift_h = h * rel_shift
            transl_mat = np.array([[1., 0., shift_w], [0., 1., shift_h]])
            image = cv.warpAffine(image, transl_mat, (w, h))

        return image


class RandomRotationNumpy:
    """Rotates an image around it's center by a randomly generated angle"""
    def __init__(self, max_angle, p=.5):
        self.max_angle = max_angle
        self.p = p

    def __call__(self, image):
        if float(torch.FloatTensor(1).uniform_()) < self.p:
            angle = 2 * (torch.FloatTensor(1).uniform_() - .5) * self.max_angle
            h, w = image.shape[:2]
            rot_mat = cv.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.)
            image = cv.warpAffine(image, rot_mat, (w, h), flags=cv.INTER_LANCZOS4)

        return image


class ResizeNumpy:
    """Resizes an image in numpy format"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = cv.resize(image, (new_h, new_w))
        return img


class CenterCropNumpy:
    """Performs a center crop of an images"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            new_h, new_w = self.output_size, self.output_size
        else:
            new_h, new_w = self.output_size

        s_h = int(h / 2 - new_h / 2)
        s_w = int(w / 2 - new_w / 2)
        image = image[s_h: s_h + new_h, s_w: s_w + new_w]
        return image


class BlurNumpy:
    """Blurs an image with the given sigma and probability"""
    def __init__(self, p, k):
        self.p = p
        assert k % 2 == 1
        self.k = k

    def __call__(self, img):
        if float(torch.FloatTensor(1).uniform_()) < self.p:
            img = cv.blur(img, (self.k, self.k))
        return img


class CutOutWithPrior:
    """Cuts rectangular patches from an image around pre-defined landmark locations"""
    def __init__(self, p, max_area):
        self.p = p
        self.max_area = max_area

    # use after resize transform
    def __call__(self, img):
        height, width = img.shape[:2]
        keypoints_ref = np.zeros((5, 2), dtype=np.float32)
        keypoints_ref[:, 0] = FivePointsAligner.ref_landmarks[:, 0] * width
        keypoints_ref[:, 1] = FivePointsAligner.ref_landmarks[:, 1] * height

        if float(torch.FloatTensor(1).uniform_()) < self.p:
            erase_num = torch.LongTensor(1).random_(1, 4)
            erase_ratio = torch.FloatTensor(1).uniform_(self.max_area / 2, self.max_area)
            erase_h = math.sqrt(erase_ratio) / float(erase_num) * height
            erase_w = math.sqrt(erase_ratio) / float(erase_num) * width

            erased_idx = []
            for _ in range(erase_num):
                erase_pos = int(torch.LongTensor(1).random_(0, 5))
                while erase_pos in erased_idx:
                    erase_pos = int(torch.LongTensor(1).random_(0, 5))

                left_corner = (
                    int(keypoints_ref[erase_pos][0] - erase_h / 2), int(keypoints_ref[erase_pos][1] - erase_w / 2))
                right_corner = (
                    int(keypoints_ref[erase_pos][0] + erase_h / 2), int(keypoints_ref[erase_pos][1] + erase_w / 2))

                cv.rectangle(img, tuple(left_corner), tuple(right_corner), (0, 0, 0), thickness=-1)
                erased_idx.append(erase_pos)

        return img
