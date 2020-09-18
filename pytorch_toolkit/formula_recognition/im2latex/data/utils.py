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

import datetime
import pickle as pkl
import os

import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import ToTensor, Compose
import operator


from .vocab import END_TOKEN, PAD_TOKEN, START_TOKEN, UNK_TOKEN, Vocab
import torch.nn.functional as F

COLOR_WHITE = (255, 255, 255)


class BatchResizePadToTGTShape():
    """This class helps to resize image to fit the target shape
    and save original aspect ratio and pad
    (if resized image's shape is not equal to target shape)
    """

    def __init__(self, target_shape):
        self.target_shape = target_shape

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        res = []
        target_height, target_width = self.target_shape
        for image_raw in imgs:

            img_h, img_w = image_raw.shape[0:2]
            if (img_h, img_w) != (target_height, target_width):
                if img_h >= target_height and img_w >= target_width:
                    # dim = (target_width, target_height)
                    rescale_h = img_h / target_height
                    rescale_w = img_w / target_width
                    if rescale_h > rescale_w:
                        new_h = int(img_h / rescale_h)
                        new_w = int(img_w / rescale_h)
                    else:
                        new_h = int(img_h / rescale_w)
                        new_w = int(img_w / rescale_w)

                    image_raw = cv.resize(image_raw, (new_w, new_h))
                    img_h, img_w = image_raw.shape[0:2]
                    if (img_h, img_w != target_height, target_width):
                        image_raw = cv.copyMakeBorder(image_raw, 0, target_height - img_h,
                                                      0, target_width - img_w, cv.BORDER_CONSTANT,
                                                      None, COLOR_WHITE)
                elif img_h < target_height and img_w < target_width:
                    rescale_h = img_h / target_height
                    rescale_w = img_w / target_width
                    if rescale_h > rescale_w:
                        new_h = int(img_h / rescale_h)
                        new_w = int(img_w / rescale_h)
                    else:
                        new_h = int(img_h / rescale_w)
                        new_w = int(img_w / rescale_w)
                    image_raw = cv.resize(image_raw, (new_w, new_h))
                    img_h, img_w = image_raw.shape[0:2]
                    if (img_h, img_w != target_height, target_width):
                        image_raw = cv.copyMakeBorder(image_raw, 0, target_height - img_h,
                                                      0, target_width - img_w, cv.BORDER_CONSTANT,
                                                      None, COLOR_WHITE)
                elif img_h < target_height and img_w >= target_width:
                    dim = (target_width, int(target_width * img_h / img_w))
                    image_raw = cv.resize(image_raw, dim)
                    image_raw = cv.copyMakeBorder(image_raw, 0, target_height - image_raw.shape[0],
                                                  0, 0, cv.BORDER_CONSTANT, None,
                                                  COLOR_WHITE)
                elif img_h >= target_height and img_w < target_width:
                    dim = (int(target_height * img_w / img_h), target_height)
                    image_raw = cv.resize(image_raw, dim)
                    img_h, img_w = image_raw.shape[0:2]
                    image_raw = cv.copyMakeBorder(image_raw, 0, 0,
                                                  0, target_width - img_w, cv.BORDER_CONSTANT,
                                                  None, COLOR_WHITE)
            res.append(image_raw)
        return res


class BatchCropPadToTGTShape():
    """This class helps to crop image to fit the target shape
    and save original aspect ratio and pad
    (if resized image's shape is not equal to target shape)
    """

    def __init__(self, target_shape):
        self.target_shape = target_shape

    def __repr__(self):
        return "CropPadToTGTShape object w target_shape {}".format(self.target_shape)

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        res = []
        target_height, target_width = self.target_shape
        for image_raw in imgs:

            img_h, img_w = image_raw.shape[0:2]
            if (img_h, img_w) != (target_height, target_width):
                if img_h >= target_height and img_w >= target_width:
                    if len(image_raw.shape) > 2:
                        image_raw = image_raw[:target_height, :target_width, :]
                    else:
                        image_raw = image_raw[:target_height, :target_width]
                elif img_h < target_height and img_w < target_width:

                    image_raw = cv.copyMakeBorder(image_raw, 0, target_height - img_h,
                                                  0, target_width - img_w, cv.BORDER_CONSTANT,
                                                  None, COLOR_WHITE)
                elif img_h < target_height and img_w >= target_width:
                    if len(image_raw.shape) > 2:
                        image_raw = image_raw[:, :target_width, :]
                    else:
                        image_raw = image_raw[:, :target_width]
                    image_raw = cv.copyMakeBorder(image_raw, 0, target_height - image_raw.shape[0],
                                                  0, 0, cv.BORDER_CONSTANT, None,
                                                  COLOR_WHITE)
                elif img_h >= target_height and img_w < target_width:
                    if len(image_raw.shape) > 2:
                        image_raw = image_raw[:target_height, :, :]
                    else:
                        image_raw = image_raw[:target_height, :]
                    img_h, img_w = image_raw.shape[0:2]
                    image_raw = cv.copyMakeBorder(image_raw, 0, 0,
                                                  0, target_width - img_w, cv.BORDER_CONSTANT,
                                                  None, COLOR_WHITE)
            res.append(image_raw)
        return res


class ImitateWebTransform():
    """
    Transformation imitates grey formulas
    like they are capture by the web-camera
    """

    def __init__(self, a=None, b=None, noise_var=10):
        default_a = 60
        default_b = 170
        self.a = a if a is not None else default_a
        self.b = b if b is not None else default_b
        assert 0 <= self.a < self.b
        assert self.b <= 255
        self.noise_var = noise_var

    def __call__(self, _imgs):

        cur_b = np.random.uniform(self.b, 255)
        cur_a = np.random.uniform(0, self.a)
        if not isinstance(_imgs, list):
            _imgs = [_imgs]
        imgs = np.stack(_imgs, axis=0)
        noise = np.random.normal(0, self.noise_var, imgs.shape)
        imgs = imgs.astype(float)
        imgs += noise
        imgs = (cur_b - cur_a) / 255 * imgs + cur_a
        imgs = np.maximum(0, imgs)
        imgs = np.minimum(255, imgs)
        imgs = imgs.astype(dtype=np.uint8)
        if imgs.shape[-1] == 3:
            imgs = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in imgs]
            imgs = [cv.cvtColor(img, cv.COLOR_GRAY2BGR) for img in imgs]
        return imgs


class BatchTransformPad():
    """Random pad batch of images
    """

    def __init__(self, shift_l=50, shift_r=50, shift_b=20, shift_t=20):
        self.shift_l = shift_l
        self.shift_r = shift_r
        self.shift_b = shift_b
        self.shift_t = shift_t

    def __call__(self, imgs):
        if self.shift_l > 0:
            left_p = np.random.randint(0, self.shift_l)
        else:
            left_p = None
        if self.shift_r > 0:
            right_p = np.random.randint(0, self.shift_r)
        else:
            right_p = None
        if self.shift_b > 0:
            bottom_p = np.random.randint(0, self.shift_b)
        else:
            bottom_p = None
        if self.shift_t > 0:
            top_p = np.random.randint(0, self.shift_t)
        else:
            top_p = None
        if not isinstance(imgs, list):
            imgs = [imgs]
        padded_imgs = [cv.copyMakeBorder(img, top_p, bottom_p, left_p, right_p,
                                         borderType=cv.BORDER_CONSTANT, value=COLOR_WHITE)
                       for img in imgs]
        return padded_imgs


class BatchToTensor():
    def __init__(self):
        ...

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        return [ToTensor()(img) for img in imgs]


class BatchTransformBlur():
    def __init__(self, sigmaX=1.15):
        self.sigmaX = sigmaX

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        return [cv.GaussianBlur(img, (3, 3), self.sigmaX) for img in imgs]


class TransformShift():
    """Shift formula randomly on x and y from a set range
    """

    def __init__(self, shift_x, shift_y):
        self.shift_x = shift_x
        self.shift_y = shift_y

    def __call__(self, img):
        tx = np.random.randint(0, self.shift_x)
        ty = np.random.randint(0, self.shift_y)
        shift_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        res = cv.warpAffine(
            img, shift_matrix, (img.shape[1], img.shape[0]), borderMode=cv.BORDER_CONSTANT, borderValue=COLOR_WHITE)
        return res


class BatchTransformRandomNoise():
    """Add random noise to batch of images
    """

    def __init__(self, intensity):
        self.intensity = intensity

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        res = []
        for img in imgs:
            mean = np.mean(img)
            variance = np.std(img)
            gauss = np.zeros_like(img, dtype=np.uint8)
            noise = np.array(np.random.normal(
                mean, variance, img.shape), dtype=np.uint8)
            gauss += noise * self.intensity
            res.append(img + gauss)
        return res


class TransformResize():
    def __init__(self, target_shape: tuple):
        self.target_shape = target_shape

    def __call__(self, img):
        if not isinstance(img, list):
            img = [img]
        res = [cv.resize(im, self.target_shape[::-1]) for im in img]
        return res


class TransformErosion():
    """Morphologic erosion
    """

    def __init__(self, kernel_size=3, iterations=1):
        self.iterations = iterations
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def __call__(self, img):
        if not isinstance(img, list):
            img = [img]
        return [cv.erode(image, self.kernel, iterations=self.iterations) for image in img]


class TransformRandomBolding():
    """
    This class helps to imitate images from scaner \ camera
    after applying binarization on them
    """

    def __init__(self, kernel_size=3, iterations=1, threshold=160, res_threshold=190, sigmaX=0.8, distr=0.7):
        self.iterations = iterations
        self.threshold = threshold
        self.res_threshold = res_threshold
        self.sigmaX = sigmaX
        self.kernel = (kernel_size, kernel_size)
        self.distr = distr

    def __call__(self, img):
        if not isinstance(img, list):
            img = [img]
        n = len(img)
        first_eroded = [
            cv.erode(image, self.kernel, iterations=self.iterations) for image in img]
        blurred = [None] * n
        res = [None] * n
        for i in range(n):
            first_eroded[i] = (first_eroded[i] >= self.threshold).astype(np.uint8) * 255
            blurred[i] = cv.GaussianBlur(first_eroded[i], (3, 3), self.sigmaX)

            blurred[i] = (blurred[i] >= self.threshold).astype(np.uint8) * 255

            # randint returns random integers in range [low, high)
            # so 0 here stands for low, 2 - high -> random integers in range [0, 1]

            total_dim = 1
            for dim in img[i].shape:
                total_dim *= dim
            mask1 = np.zeros(total_dim)
            ones_len = int(total_dim * self.distr)
            mask1[0:ones_len] = 1
            np.random.shuffle(mask1)
            mask1 = mask1.reshape(img[i].shape)
            mask2 = 1 - mask1

            masked_source = np.multiply(img[i], mask1)
            masked_blur = np.multiply(blurred[i], mask2)

            res[i] = masked_blur + masked_source
            res[i] = cv.GaussianBlur(res[i].astype(np.uint8), (3, 3), self.sigmaX)
            res[i] = (res[i] >= self.res_threshold).astype(np.uint8) * 255

        return res


class TransformDilation():
    """Morphologic dilation
    """

    def __init__(self, kernel_size=3, iterations=3):
        self.iterations = iterations
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def __call__(self, img):
        if not isinstance(img, list):
            img = [img]
        return [cv.dilate(image, self.kernel, iterations=self.iterations) for image in img]


class BatchTransformBin():
    def __init__(self, threshold):

        self.transform = cv.threshold
        self.thresh_type = cv.THRESH_BINARY
        self.threshold = threshold
        self.max_val = 255

    def __call__(self, img):
        if not isinstance(img, list):
            img = [img]
        return [cv.threshold(im, self.threshold, self.max_val, self.thresh_type)[1] for im in img]


class BatchTransfromAdaptiveBin():
    def __init__(self, threshold, block_size, method=cv.ADAPTIVE_THRESH_MEAN_C, mean_c=10):
        self.method = method
        self.block_size = block_size
        self.C = mean_c
        self.thresh_type = cv.THRESH_BINARY
        self.threshold = threshold
        self.max_val = 255

    def __call__(self, img):
        if not isinstance(img, list):
            img = [img]
        img = [cv.cvtColor(im, cv.COLOR_BGR2GRAY) for im in img]
        return [cv.adaptiveThreshold(
            im, self.max_val, self.method, self.thresh_type,
            self.block_size, self.C) for im in img]


class BatchTransformRescale():
    def __init__(self, scale_min, scale_max):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, imgs):
        fx = np.random.uniform(self.scale_min, self.scale_max)
        fy = fx  # np.random.uniform(fx, self.scale_max)
        if not isinstance(imgs, list):
            imgs = [imgs]
        imgs = [cv.resize(img, dsize=None, fx=fx, fy=fy) for img in imgs]
        return imgs


class BatchTransformRotate():
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, imgs):
        bound = self.angle
        angle = np.random.uniform(low=0 - bound, high=bound)
        if not isinstance(imgs, list):
            imgs = [imgs]

        img = imgs[0]
        h, w = img.shape[0:2]
        center = int(w / 2), int(h / 2)

        M = cv.getRotationMatrix2D(center, angle=angle, scale=1)
        src = np.array([[[0, 0]], [[w, 0]], [[0, h]], [[w, h]]])

        new_pts = cv.transform(src, M)
        top_left, top_right, bottom_left, bottom_right = new_pts.astype(int)

        bb = cv.boundingRect(new_pts)
        empty_img_shape = (bb[3], bb[2])
        empty_img = 255 * np.ones(empty_img_shape)
        shift_matrix = np.array([[0, 0, 0-bb[0]], [0, 0, 0 - bb[1]]])
        M = M + shift_matrix
        rotated = [cv.warpAffine(
            img, M, empty_img_shape[::-1], borderMode=cv.BORDER_CONSTANT, borderValue=COLOR_WHITE) for img in imgs]

        return rotated


def get_timestamp():
    return "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())


def get_num_lines_in_file(path):
    total = 0
    with open(path) as f:
        for line in f:
            total += 1
    return total


def cal_loss(logits, targets, should_cut_by_min=False):
    """args:
        logits: probability distribution return by model
                [B, MAX_LEN, voc_size]
        targets: target formulas
                [B, MAX_LEN]
    """

    if should_cut_by_min:
        required_len = min(logits.size(1), targets.size(1))
        logits = logits.narrow(1, 0, required_len)
        targets = targets.narrow(1, 0, required_len)
        if required_len < targets.size(1):
            warn("Cutting tensor leads to tensor sized less than target")
    else:
        # narrows on 1st dim from 'start_pos' 'length' symbols
        logits = logits.narrow(1, 0, targets.size(1))

    padding = torch.ones_like(targets) * PAD_TOKEN
    mask = (targets != padding)
    targets = targets.masked_select(mask)
    logits = logits.masked_select(mask.unsqueeze(2).expand(-1, -1, logits.size(2))
                                  ).contiguous().view(-1, logits.size(2))
    logits = torch.log(logits)

    assert logits.size(0) == targets.size(0)
    pred = torch.max(logits.data, 1)[1]

    accuracy = (pred == targets)
    accuracy = accuracy.cpu().numpy().astype(np.uint32)
    accuracy = np.sum(accuracy) / len(accuracy)

    loss = F.nll_loss(logits, targets)
    return loss, accuracy.item()


def collate_fn(sign2id, batch, *, batch_transform=None):
    # filter the pictures that have different width or height
    size = batch[0]['img'].shape
    batch = [img_formula for img_formula in batch
             if img_formula['img'].shape == size]
    # sort by the length of formula
    batch.sort(key=lambda img_formula: len(img_formula['formula'].split()),
               reverse=True)

    imgs = [item['img'] for item in batch]
    formulas = [item['formula'] for item in batch]
    img_names = [item['img_name'] for item in batch]
    formulas_tensor = formulas2tensor(formulas, sign2id)

    if batch_transform:
        imgs = batch_transform(imgs)
    imgs = torch.stack(imgs, dim=0)

    bsize = len(batch)
    tgt4training = torch.cat([torch.ones(bsize, 1).long()*START_TOKEN, formulas_tensor],
                             dim=1
                             )  # targets for training, begin with START_TOKEN
    tgt4cal_loss = torch.cat([formulas_tensor, torch.ones(bsize, 1).long()*END_TOKEN],
                             dim=1)  # targets for calculating loss, end with END_TOKEN
    return img_names, imgs, tgt4training, tgt4cal_loss


def create_list_of_transforms(transforms_list):
    transforms = []
    if transforms_list:
        for transform in transforms_list:
            if transform['name'] == 'BatchResizePadToTGTShape':
                transforms.append(BatchResizePadToTGTShape(transform['target_shape']))
            elif transform['name'] == 'BatchCropPadToTGTShape':
                transforms.append(BatchCropPadToTGTShape(transform['target_shape']))
            elif transform['name'] == 'BatchTransformBin':
                transforms.append(BatchTransformBin(transform['threshold']))
            elif transform['name'] == 'BatchTransformBlur':
                transforms.append(BatchTransformBlur())
            elif transform['name'] == 'BatchTransformPad':
                transforms.append(BatchTransformPad(*transform['shifts']))
            elif transform['name'] == 'BatchTransformRandomNoise':
                transforms.append(BatchTransformRandomNoise(transform['intensivity']))
            elif transform['name'] == 'BatchTransformRescale':
                transforms.append(BatchTransformRescale(*transform['scales']))
            elif transform['name'] == 'BatchTransformRotate':
                transforms.append(BatchTransformRotate(transform['angle']))
            elif transform['name'] == 'BatchTransfromAdaptiveBin':
                transforms.append(BatchTransfromAdaptiveBin(transform['threshold'], transform['block_size']))
            elif transform['name'] == 'TransformDilation':
                transforms.append(TransformDilation())
            elif transform['name'] == 'TransformErosion':
                transforms.append(TransformErosion())
            elif transform['name'] == 'TransformResize':
                transforms.append(TransformResize(transform['target_shape']))
            elif transform['name'] == 'TransformShift':
                transforms.append(TransformShift(*transform['shifts']))
            elif transform['name'] == TransformRandomBolding:
                transforms.append(TransformRandomBolding(transform['kernel_size'], transform['iterations'],
                                                         transform['threshold'], transform['res_threshold'], transform['sigmaX'], transform['distr']))
    transforms.append(BatchToTensor())
    return Compose(transforms)


def formulas2tensor(formulas, sign2id):
    """convert formula to tensor"""
    formulas = [formula.split() for formula in formulas]
    batch_size = len(formulas)
    max_len = len(formulas[0])
    tensors = torch.ones(batch_size, max_len, dtype=torch.long) * PAD_TOKEN
    for i, formula in enumerate(formulas):
        for j, sign in enumerate(formula):
            tensors[i][j] = sign2id.get(sign, UNK_TOKEN)
    return tensors
