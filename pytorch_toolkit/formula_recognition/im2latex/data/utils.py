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

import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor

from .vocab import END_TOKEN, PAD_TOKEN, START_TOKEN, UNK_TOKEN

COLOR_WHITE = (255, 255, 255)


class TransformResizePad:
    """This class helps to resize image to fit the target shape
    and save original aspect ratio and pad
    (if resized image's shape is not equal to target shape)
    """

    def __init__(self, target_shape):
        self.target_shape = target_shape

    def __repr__(self):
        return "ResizePad(target_shape={})".format(self.target_shape)

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        res = []
        target_height, target_width = self.target_shape
        for image_raw in imgs:
            img_h, img_w = image_raw.shape[0:2]
            scale = min(target_height / img_h, target_width / img_w)
            image_raw = cv.resize(image_raw, None, fx=scale, fy=scale)
            img_h, img_w = image_raw.shape[0:2]
            image_raw = cv.copyMakeBorder(image_raw, 0, target_height - img_h,
                                          0, target_width - img_w, cv.BORDER_CONSTANT,
                                          None, COLOR_WHITE)
            assert tuple(image_raw.shape[0:2]) == tuple(self.target_shape[0:2]
                                                        ), f"image_raw shape {image_raw.shape[0:2]}, tgt_shape: {self.target_shape[0:2]}"
            res.append(image_raw)
        return res


class TransformCropPad:
    """This class helps to make top-left crop of the image to fit the target shape
    and save original aspect ratio and pad (if resized image's shape is not equal to target shape)
    """

    def __init__(self, target_shape):
        self.target_shape = target_shape

    def __repr__(self):
        return "CropPad(target_shape={})".format(self.target_shape)

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        res = []
        target_height, target_width = self.target_shape
        for image_raw in imgs:
            img_h, img_w = image_raw.shape[0:2]
            new_w = min(target_width, img_w)
            new_h = min(target_height, img_h)
            image_raw = image_raw[:new_h, :new_w, :]
            img_h, img_w = image_raw.shape[0:2]
            image_raw = cv.copyMakeBorder(image_raw, 0, target_height - img_h,
                                          0, target_width - img_w, cv.BORDER_CONSTANT,
                                          None, COLOR_WHITE)
            res.append(image_raw)
        return res


class TransformPad:
    """Random pad batch of images
    """

    def __init__(self, pad_l=50, pad_r=50, pad_b=20, pad_t=20):
        self.pad_l = pad_l
        self.pad_r = pad_r
        self.pad_b = pad_b
        self.pad_t = pad_t

    def __repr__(self):
        return "TransformPad(pad_l={}, pad_r={}, pad_b={}, pad_t={})".format(
            self.pad_l, self.pad_r, self.pad_b, self.pad_t)

    def __call__(self, imgs):
        if self.pad_l > 0:
            left_p = np.random.randint(0, self.pad_l)
        else:
            left_p = None
        if self.pad_r > 0:
            right_p = np.random.randint(0, self.pad_r)
        else:
            right_p = None
        if self.pad_b > 0:
            bottom_p = np.random.randint(0, self.pad_b)
        else:
            bottom_p = None
        if self.pad_t > 0:
            top_p = np.random.randint(0, self.pad_t)
        else:
            top_p = None
        if not isinstance(imgs, list):
            imgs = [imgs]
        padded_imgs = [cv.copyMakeBorder(img, top_p, bottom_p, left_p, right_p,
                                         borderType=cv.BORDER_CONSTANT, value=COLOR_WHITE)
                       for img in imgs]
        return padded_imgs


class TransformToTensor:
    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        return [ToTensor()(img) for img in imgs]


class TransformOvinoIR:
    """The same transform as above with the exception that it does not
    cast input array to [0, 1] range
    """

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        imgs = [np.transpose(img, [2, 0, 1]) for img in imgs]
        return [torch.Tensor(img) for img in imgs]


class TransformBlur:
    def __init__(self, sigmaX=1.15):
        self.sigmaX = sigmaX

    def __repr__(self):
        return "TransformBlur(sigmaX={})".format(self.sigmaX)

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        return [cv.GaussianBlur(img, (3, 3), self.sigmaX) for img in imgs]


class TransformShift:
    """Shift formula randomly on x and y from a set range
    """

    def __init__(self, shift_x, shift_y):
        self.shift_x = shift_x
        self.shift_y = shift_y

    def __repr__(self):
        return "TransformShift(shift_x={}, shift_y={})".format(self.shift_x, self.shift_y)

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        tx = np.random.randint(0, self.shift_x)
        ty = np.random.randint(0, self.shift_y)
        shift_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        res = [cv.warpAffine(img, shift_matrix, (img.shape[1], img.shape[0]),
                             borderMode=cv.BORDER_CONSTANT, borderValue=COLOR_WHITE) for img in imgs]
        return res


class TransformRandomNoise:
    """Add random noise to batch of images
    """

    def __init__(self, intensity):
        self.intensity = intensity

    def __repr__(self):
        return "TransformRandomNoise(intensity={})".format(self.intensity)

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        res = []
        for img in imgs:
            mean = np.mean(img)
            variance = np.std(img)
            noise = np.array(np.random.normal(
                mean, variance, img.shape), dtype=np.uint8)
            res.append(img + noise * self.intensity)
        return res


class TransformResize:
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def __repr__(self):
        return "TransformResize(target_shape={})".format(self.target_shape)

    def __call__(self, img):
        if not isinstance(img, list):
            img = [img]
        res = [cv.resize(im, tuple(self.target_shape[::-1])) for im in img]
        return res


class TransformErosion:
    """Morphologic erosion
    """

    def __init__(self, kernel_size=3, iterations=1):
        self.iterations = iterations
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def __repr__(self):
        return "TransformErosion(iterations={}, kernel_size={})".format(self.iterations, self.kernel.shape[0])

    def __call__(self, img):
        if not isinstance(img, list):
            img = [img]
        return [cv.erode(image, self.kernel, iterations=self.iterations) for image in img]


class TransformRandomBolding:
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

    def __repr__(self):
        return """TransformRandomBolding(iterations={}, threshold={}, res_threshold={},
                sigmaX={}, kernel_size={}, distr={})""".format(
            self.iterations, self.threshold, self.res_threshold, self.sigmaX, self.kernel[0], self.distr)

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


class TransformDilation:
    """Morphologic dilation
    """

    def __init__(self, kernel_size=3, iterations=3):
        self.iterations = iterations
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def __repr__(self):
        return "TransformDilation(iterations={}, kernel={})".format(self.iterations, self.kernel)

    def __call__(self, img):
        if not isinstance(img, list):
            img = [img]
        return [cv.dilate(image, self.kernel, iterations=self.iterations) for image in img]


class TransformBin:
    def __init__(self, threshold):
        self.transform = cv.threshold
        self.thresh_type = cv.THRESH_BINARY
        self.threshold = threshold
        self.max_val = 255

    def __repr__(self):
        return "TransformBin(threshold={})".format(self.threshold)

    def __call__(self, img):
        if not isinstance(img, list):
            img = [img]
        return [cv.threshold(im, self.threshold, self.max_val, self.thresh_type)[1] for im in img]


class TransformAdaptiveBin:
    def __init__(self, threshold, block_size, method=cv.ADAPTIVE_THRESH_MEAN_C, mean_c=10):
        self.method = method
        self.block_size = block_size
        self.C = mean_c
        self.thresh_type = cv.THRESH_BINARY
        self.threshold = threshold
        self.max_val = 255

    def __repr__(self):
        return "TransfromAdaptiveBin(threshold={}, block_size={})".format(self.threshold, self.block_size)

    def __call__(self, img):
        if not isinstance(img, list):
            img = [img]
        img = [cv.cvtColor(im, cv.COLOR_BGR2GRAY) for im in img]
        return [cv.adaptiveThreshold(
            im, self.max_val, self.method, self.thresh_type,
            self.block_size, self.C) for im in img]


class TransformRescale:
    def __init__(self, scale_min, scale_max):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __repr__(self):
        return "TransformRescale(scale_min={}, scale_max={})".format(self.scale_min, self.scale_max)

    def __call__(self, imgs):
        fx = np.random.uniform(self.scale_min, self.scale_max)
        fy = fx
        if not isinstance(imgs, list):
            imgs = [imgs]
        imgs = [cv.resize(img, dsize=None, fx=fx, fy=fy) for img in imgs]
        return imgs


class TransformRotate:
    def __init__(self, angle):
        self.angle = angle

    def __repr__(self):
        return "TransformRotate(angle={})".format(self.angle)

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

        bb = cv.boundingRect(new_pts)
        empty_img_shape = (bb[3], bb[2])
        shift_matrix = np.array([[0, 0, 0-bb[0]], [0, 0, 0 - bb[1]]])
        M = M + shift_matrix
        rotated = [cv.warpAffine(
            img, M, empty_img_shape[::-1], borderMode=cv.BORDER_CONSTANT, borderValue=COLOR_WHITE) for img in imgs]

        return rotated


TRANSFORMS = {
    'TransformResizePad': TransformResizePad,
    'TransformCropPad': TransformCropPad,
    'TransformBin': TransformBin,
    'TransformBlur': TransformBlur,
    'TransformPad': TransformPad,
    'TransformRandomNoise': TransformRandomNoise,
    'TransformRescale': TransformRescale,
    'TransformRotate': TransformRotate,
    'TransfromAdaptiveBin': TransformAdaptiveBin,
    'TransformDilation': TransformDilation,
    'TransformErosion': TransformErosion,
    'TransformResize': TransformResize,
    'TransformShift': TransformShift,
    'TransformRandomBolding': TransformRandomBolding,
}


def get_timestamp():
    return "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())


def get_num_lines_in_file(path):
    total = 0
    with open(path) as f:
        for line in f:
            total += 1
    return total


def collate_fn(sign2id, batch, *, batch_transform=None):
    # filter the pictures that have different width or height
    size = batch[0]['img'].shape
    batch = [img_formula for img_formula in batch
             if img_formula['img'].shape == size]
    # sort by the length of formula
    # the purpose of the sort is to put the longest formula on the first place
    # to get correct size of the tensor in the formulas2tensor function
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
    # Ground truth symbols that are used as a decoding step input for the next symbol prediction during training.
    training_gt = torch.cat([torch.ones(bsize, 1).long()*START_TOKEN, formulas_tensor], dim=1)
    # Ground truth values for the outputs of decoder. Used for loss computation.
    loss_computation_gt = torch.cat([formulas_tensor, torch.ones(bsize, 1).long()*END_TOKEN], dim=1)
    return img_names, imgs, training_gt, loss_computation_gt


def create_list_of_transforms(transforms_list, ovino_ir=False):
    transforms = []
    if transforms_list:
        for transform in transforms_list:
            transform_name = transform.pop('name')
            transforms.append(TRANSFORMS[transform_name](**transform))
    if ovino_ir:
        transforms.append(TransformOvinoIR())
    else:
        transforms.append(TransformToTensor())
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
