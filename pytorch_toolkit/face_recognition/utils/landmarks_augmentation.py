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

import cv2 as cv
import numpy as np
import torch


class Rescale:
    """Resizes an image and corresponding landmarks"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['img'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if w > h:
                new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size * h / w, self.output_size
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv.resize(image, (new_h, new_w))
        return {'img': img, 'landmarks': landmarks}


class RandomCrop:
    """Makes a random crop from the source image with corresponding transformation of landmarks"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['img'], sample['landmarks'].reshape(-1, 2)

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left / float(w), top / float(h)]
        for point in landmarks:
            point[0] *= float(h) / new_h
            point[1] *= float(w) / new_w

        return {'img': image, 'landmarks': landmarks}


class HorizontalFlip:
    """Flips an input image and landmarks horizontally with a given probability"""
    def __init__(self, p=.5):
        self.p = p

    def __call__(self, sample):
        image, landmarks = sample['img'], sample['landmarks'].reshape(-1, 2)

        if float(torch.FloatTensor(1).uniform_()) < self.p:
            image = cv.flip(image, 1)
            landmarks = landmarks.reshape(5, 2)
            landmarks[:, 0] = 1. - landmarks[:, 0]
            tmp = np.copy(landmarks[0])
            landmarks[0] = landmarks[1]
            landmarks[1] = tmp

            tmp = np.copy(landmarks[3])
            landmarks[3] = landmarks[4]
            landmarks[4] = tmp

        return {'img': image, 'landmarks': landmarks}


class Blur:
    """Blurs an image with the given sigma and probability"""
    def __init__(self, p, k):
        self.p = p
        assert k % 2 == 1
        self.k = k

    def __call__(self, sample):
        image, landmarks = sample['img'], sample['landmarks']

        if float(torch.FloatTensor(1).uniform_()) < self.p:
            image = cv.blur(image, (self.k, self.k))

        return {'img': image, 'landmarks': landmarks}


class Show:
    """Show image using opencv"""
    def __call__(self, sample):
        image, landmarks = sample['img'].copy(), sample['landmarks'].reshape(-1, 2)
        h, w = image.shape[:2]
        for point in landmarks:
            cv.circle(image, (int(point[0]*w), int(point[1]*h)), 3, (255, 0, 0), -1)
        cv.imshow('image', image)
        cv.waitKey()
        return sample


class RandomRotate:
    """
        Rotates an image around it's center by a randomly generated angle.
        Also performs the same transformation with landmark points.
    """
    def __init__(self, max_angle, p=.5):
        self.max_angle = max_angle
        self.p = p

    def __call__(self, sample):
        image, landmarks = sample['img'], sample['landmarks']

        if float(torch.FloatTensor(1).uniform_()) < self.p:
            angle = 2*(torch.FloatTensor(1).uniform_() - .5)*self.max_angle
            h, w = image.shape[:2]
            rot_mat = cv.getRotationMatrix2D((w*0.5, h*0.5), angle, 1.)
            image = cv.warpAffine(image, rot_mat, (w, h), flags=cv.INTER_LANCZOS4)
            rot_mat_l = cv.getRotationMatrix2D((0.5, 0.5), angle, 1.)
            landmarks = cv.transform(landmarks.reshape(1, 5, 2), rot_mat_l).reshape(5, 2)

        return {'img': image, 'landmarks': landmarks}


class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, switch_rb=False):
        self.switch_rb = switch_rb

    def __call__(self, sample):
        image, landmarks = sample['img'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if self.switch_rb:
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image = image.transpose((2, 0, 1))
        return {'img': torch.from_numpy(image).type(torch.FloatTensor) / 255,
                'landmarks': torch.from_numpy(landmarks).type(torch.FloatTensor).view(-1, 1, 1)}


class RandomScale:
    """Performs uniform scale with a random magnitude"""
    def __init__(self, max_scale, min_scale, p=.5):
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.p = p

    def __call__(self, sample):
        image, landmarks = sample['img'], sample['landmarks']

        if float(torch.FloatTensor(1).uniform_()) < self.p:
            scale = self.min_scale + torch.FloatTensor(1).uniform_()*(self.max_scale - self.min_scale)
            h, w = image.shape[:2]
            rot_mat = cv.getRotationMatrix2D((w*0.5, h*0.5), 0, scale)
            image = cv.warpAffine(image, rot_mat, (w, h), flags=cv.INTER_LANCZOS4)
            rot_mat_l = cv.getRotationMatrix2D((0.5, 0.5), 0, scale)
            landmarks = cv.transform(landmarks.reshape(1, 5, 2), rot_mat_l).reshape(5, 2)

        return {'img': image, 'landmarks': landmarks}
