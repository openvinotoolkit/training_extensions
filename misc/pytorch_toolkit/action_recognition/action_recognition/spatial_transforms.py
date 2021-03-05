import collections
import math
import numbers
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance

try:
    import accimage
except ImportError:
    accimage = None

MEAN_STATISTICS = {
    'imagenet': [0.485, 0.456, 0.406],
    'kinetics': [0.434, 0.405, 0.378],
    'activitynet': [0.450, 0.422, 0.390],
    'none': [0.0, 0.0, 0.0]
}

STD_STATISTICS = {
    'imagenet': [0.229, 0.224, 0.225],
    'kinetics': [0.152, 0.148, 0.157],
    'none': [1.0, 1.0, 1.0]
}


def resize(img, size):
    if not isinstance(size, (list, tuple)):
        size = (size, size)
    if isinstance(img, np.ndarray):
        return cv2.resize(img, size)
    return img.resize(size, Image.LINEAR)


def crop(img, position):
    x1, y1, x2, y2 = position
    if isinstance(img, np.ndarray):
        return img[y1:y2, x1:x2]
    return img.crop(position)


def flip(img, horizontal=False):
    if isinstance(img, np.ndarray):
        if horizontal:
            return np.ascontiguousarray(img[:, ::-1, :])
        return np.ascontiguousarray(img[::-1, ...])
    if horizontal:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def size(img):
    if isinstance(img, np.ndarray):
        h, w, c = img.shape
        return w, h
    w, h = img.size
    return w, h


def pad(img, p, value):
    top, bottom, left, right = p
    if isinstance(img, np.ndarray):
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)
    else:
        width, height = img.size
        new_width = width + right + left
        new_height = height + top + bottom
        new_img = Image.new(img.mode, (new_width, new_height), value)
        new_img.paste(img, (left, top))
    return new_img


def _repr_params(**kwargs):
    params = ['{}={}'.format(k, str(v)) for k, v in kwargs.items()]
    return '({})'.format(', '.join(params))


class VideoSpatialTransform:
    def randomize_parameters(self):
        pass

    def __repr__(self):
        visible_params = {k: getattr(self, k) for k in dir(self) if
                          not k.startswith('_') and k != 'randomize_parameters'}
        return self.__class__.__name__ + _repr_params(**visible_params)


class Compose(VideoSpatialTransform):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(VideoSpatialTransform):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img


class Normalize(VideoSpatialTransform):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


class Scale(VideoSpatialTransform):
    """Rescale the input image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
    """

    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and
                                         len(size) == 2)
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image or np.ndarray): Image to be scaled.
        Returns:
            (PIL.Image or np.ndarray): Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = size(img)
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return resize(img, (ow, oh))
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return resize(img, (ow, oh))
        else:
            return resize(img, self.size)


class CenterCrop(VideoSpatialTransform):
    """Crops the given image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = size(img)
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return crop(img, (x1, y1, x1 + tw, y1 + th))


class CornerCrop(VideoSpatialTransform):
    """Crops the given image at the corners"""

    def __init__(self, size, crop_position=None):
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, img):
        image_width = size(img)[0]
        image_height = size(img)[1]

        if self.crop_position == 'c':
            th, tw = (self.size, self.size)
            x1 = int(round((image_width - tw) / 2.))
            y1 = int(round((image_height - th) / 2.))
            x2 = x1 + tw
            y2 = y1 + th
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = self.size
            y2 = self.size
        elif self.crop_position == 'tr':
            x1 = image_width - self.size
            y1 = 0
            x2 = image_width
            y2 = self.size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - self.size
            x2 = self.size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - self.size
            y1 = image_height - self.size
            x2 = image_width
            y2 = image_height

        img = crop(img, (x1, y1, x2, y2))

        return img

    def __repr__(self):
        return self.__class__.__name__ + _repr_params(size=self.size, crop_position=self.crop_position)

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]


class PadIfNeeded(VideoSpatialTransform):
    def __init__(self, size, value=(0, 0, 0), mode='center'):
        self.size = size
        self.value = value
        assert mode in {'center', 'topleft'}
        self.mode = mode

    def __call__(self, img):
        w, h = size(img)
        w_pad = max(self.size[0] - w, 0)
        h_pad = max(self.size[1] - h, 0)
        if w_pad > 0 or h_pad > 0:
            if self.mode == 'center':
                dh = h_pad // 2
                dw = w_pad // 2
                img = pad(img, (dh, h_pad - dh, dw, w_pad - dw), self.value)
            else:
                img = pad(img, (0, h_pad, 0, w_pad), self.value)
        w, h = size(img)
        return img


class RandomCrop(VideoSpatialTransform):
    """Crop image at the random point with the standard normal distribution from the center"""

    def __init__(self, size, mode='norm'):
        self.size = size
        assert mode in {'norm', 'uniform'}
        self.mode = mode

    def __call__(self, img):
        w, h = size(img)
        x_c = w // 2
        sigma_x = max(0, x_c - self.size // 2)
        y_c = h // 2
        sigma_y = max(0, y_c - self.size // 2)
        dw = (w - self.size) // 2
        dh = (h - self.size) // 2
        if self.mode == 'norm':
            x_c = max(min(round(random.gauss(x_c, 3 * math.sqrt(sigma_x))), x_c + dw), x_c - dw)
            y_c = max(min(round(random.gauss(y_c, 3 * math.sqrt(sigma_y))), y_c + dh), y_c - dh)
        else:
            if dw > 0:
                x_c = random.randrange(x_c - dw, x_c + dw)
            if dh > 0:
                y_c = random.randrange(y_c - dh, y_c + dh)
        x1 = max(0, x_c - self.size // 2)
        x2 = x1 + self.size
        y1 = max(0, y_c - self.size // 2)
        y2 = y1 + self.size
        img = crop(img, (x1, y1, x2, y2))
        w, h = size(img)
        return img


class GaussCrop(RandomCrop):
    def __init__(self, size):
        super().__init__(size, mode='norm')


class RandomFlip(VideoSpatialTransform):
    """Flip the given PIL.Image randomly with a probability of 0.5."""
    def __init__(self, horizontal=True):
        self.horizontal = horizontal

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self._rand < 0.5:
            return flip(img, horizontal=self.horizontal)
        return img

    def randomize_parameters(self):
        self._rand = random.random()


class RandomHorizontalFlip(RandomFlip):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""
    def __init__(self):
        super().__init__(horizontal=True)


class RandomVerticalFlip(RandomFlip):
    """Vertically flip the given PIL.Image randomly with a probability of 0.5."""
    def __init__(self):
        super().__init__(horizontal=False)


class HorizontalFlip(VideoSpatialTransform):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Flipped image.
        """
        return flip(img)


class RandomScale(VideoSpatialTransform):
    def __init__(self, scale_ratios=None, scale_range=None):
        self.scale_ratios = scale_ratios
        self.scale_range = scale_range
        assert (scale_ratios is None) != (scale_range is None)

    def __call__(self, image):
        w, h = size(image)
        if self.scale_ratios is not None:
            scale = random.choice(self.scale_ratios)
        else:
            scale = random.uniform(*self.scale_range)
        w = int(math.ceil(w * scale))
        h = int(math.ceil(h * scale))
        assert w > 0 and h > 0
        image = resize(image, (w, h))
        w, h = size(image)
        return image


class MultiScaleCrop(VideoSpatialTransform):
    """
    Description: Corner cropping and multi-scale cropping. Two data augmentation techniques introduced in:
        Towards Good Practices for Very Deep Two-Stream ConvNets,
        http://arxiv.org/abs/1507.02159
        Limin Wang, Yuanjun Xiong, Zhe Wang and Yu Qiao

    Parameters:
        size: height and width required by network input, e.g., (224, 224)
        scale_ratios: efficient scale jittering, e.g., [1.0, 0.875, 0.75, 0.66]
        fix_crop: use corner cropping or not. Default: True
        more_fix_crop: use more corners or not. Default: True
        max_distort: maximum distortion. Default: 1
        interpolation: Default: cv2.INTER_LINEAR
    """

    def __init__(self, size, scale_ratios, fix_crop=True, more_fix_crop=True, max_distort=1,
                 interpolation=Image.LINEAR):
        self.height = size[0]
        self.width = size[1]
        self.scale_ratios = scale_ratios
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.max_distort = max_distort
        self.interpolation = interpolation

        self._crop_scale = None
        self._crop_offset = None
        self._num_scales = len(scale_ratios)
        self._num_offsets = 5 if not more_fix_crop else 13

    def fillFixOffset(self, datum_height, datum_width):
        h_off = int((datum_height - self.height) / 4)
        w_off = int((datum_width - self.width) / 4)

        offsets = []
        offsets.append((0, 0))  # upper left
        offsets.append((0, 4 * w_off))  # upper right
        offsets.append((4 * h_off, 0))  # lower left
        offsets.append((4 * h_off, 4 * w_off))  # lower right
        offsets.append((2 * h_off, 2 * w_off))  # center

        if self.more_fix_crop:
            offsets.append((0, 2 * w_off))  # top center
            offsets.append((4 * h_off, 2 * w_off))  # bottom center
            offsets.append((2 * h_off, 0))  # left center
            offsets.append((2 * h_off, 4 * w_off))  # right center

            offsets.append((1 * h_off, 1 * w_off))  # upper left quarter
            offsets.append((1 * h_off, 3 * w_off))  # upper right quarter
            offsets.append((3 * h_off, 1 * w_off))  # lower left quarter
            offsets.append((3 * h_off, 3 * w_off))  # lower right quarter

        return offsets

    def __repr__(self):
        return self.__class__.__name__ + _repr_params(h=self.height, w=self.width, scales=self.scale_ratios,
                                                      fix_crop=self.fix_crop, max_distort=self.max_distort)

    def fillCropSize(self, input_height, input_width):
        crop_sizes = []
        base_size = np.min((input_height, input_width))
        for scale in self.scale_ratios:
            crop_h = int(base_size * scale)
            crop_w = int(base_size * scale)
            # append this cropping size into the list
            crop_sizes.append((crop_h, crop_w))
        return crop_sizes

    def __call__(self, image):
        w, h = size(image)
        crop_size_pairs = self.fillCropSize(h, w)
        crop_height, crop_width = crop_size_pairs[self._crop_scale]

        if self.fix_crop:
            # Randomly choosen crop position.
            offsets = self.fillFixOffset(h, w)
            h_off, w_off = offsets[self._crop_offset]
        else:
            # Center crop.
            h_off = (h - crop_height) // 2
            w_off = (w - crop_width) // 2

        x1, y1, x2, y2 = w_off, h_off, w_off + crop_width, h_off + crop_height

        image = crop(image, (x1, y1, x2, y2))
        return resize(image, (self.width, self.height))

    def randomize_parameters(self):
        self._crop_scale = np.random.choice(self._num_scales)
        self._crop_offset = np.random.choice(self._num_offsets)


class RandomSaturation(VideoSpatialTransform):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        if self.rnd:
            im2arr = np.array(image).astype(np.float32)
            im2arr[:, :, 1] *= self.coef
            im2arr[:, :, 1][im2arr[:, :, 1] > 100] = 100
            image = Image.fromarray(im2arr.astype(np.uint8), mode='HSV')
        return image

    def __repr__(self):
        return self.__class__.__name__ + _repr_params(lower=self.lower, upper=self.upper)

    def randomize_parameters(self):
        self.rnd = random.randint(0, 1)
        self.scale = random.uniform(self.lower, self.upper)
        self.coef = random.uniform(self.lower, self.upper)


class RandomHue(VideoSpatialTransform):
    def __init__(self, delta=14.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image):
        if self.rnd:
            im2arr = np.array(image).astype(np.float32)  # im2arr.shape: height x width x channel
            im2arr[:, :, 0] += self.delta_rnd
            im2arr[:, :, 0][im2arr[:, :, 0] > 360.0] = 360.0
            im2arr[:, :, 0][im2arr[:, :, 0] < 0.0] = 0
            image = Image.fromarray(im2arr.astype(np.uint8), mode='HSV')
        return image

    def __repr__(self):
        return self.__class__.__name__ + _repr_params(delta=self.delta)

    def randomize_parameters(self):
        self.rnd = random.randint(0, 1)
        self.delta_rnd = random.randint(-self.delta, self.delta)


class RandomLightingNoise(VideoSpatialTransform):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image):
        if self.rnd:
            shuffle = SwapChannels(self.perm)  # shuffle channels
            image = shuffle(image)
        return image

    def __repr__(self):
        return self.__class__.__name__ + _repr_params(perms=self.perms)

    def randomize_parameters(self):
        self.rnd = random.randint(0, 1)
        self.perm = self.perms[random.randint(0, len(self.perms) - 1)]


class ConvertColor(VideoSpatialTransform):
    def __init__(self, current='RGB', transform='HSV'):
        self.transform = transform
        self.current = current

    def __repr__(self):
        return self.__class__.__name__ + _repr_params(current=self.current, transform=self.transform)

    def __call__(self, image):
        if self.current == 'RGB' and self.transform == 'HSV':
            image = image.convert('HSV')
        elif self.current == 'HSV' and self.transform == 'RGB':
            image = image.convert('RGB')
        else:
            raise NotImplementedError
        return image


class RandomContrast(VideoSpatialTransform):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __repr__(self):
        return self.__class__.__name__ + _repr_params(lower=self.lower, upper=self.upper)

    # expects float image
    def __call__(self, image):
        if self.rnd:
            if isinstance(image, np.ndarray):
                im = Image.fromarray(image)
            else:
                im = image.copy()
            enhancer = ImageEnhance.Contrast(im)
            im = enhancer.enhance(self.factor)
            if isinstance(image, np.ndarray):
                image = np.asarray(im)
            else:
                image = im
        return image

    def randomize_parameters(self):
        self.rnd = random.randint(0, 1)
        self.factor = random.uniform(self.lower, self.upper)


class RandomBrightness(VideoSpatialTransform):
    def __init__(self, delta=0.15):
        assert delta > 0.0
        assert delta < 1.0
        self.delta = delta

    def __repr__(self):
        return self.__class__.__name__ + _repr_params(delta=self.delta)

    def __call__(self, image):
        if self.rnd:
            if isinstance(image, np.ndarray):
                im = Image.fromarray(image)
            else:
                im = image.copy()
            enhancer = ImageEnhance.Brightness(im)
            im = enhancer.enhance(self.factor)
            if isinstance(image, np.ndarray):
                image = np.asarray(im)
            else:
                image = im
        return image

    def randomize_parameters(self):
        self.rnd = random.randint(0, 1)
        self.factor = random.uniform(1.0 - self.delta, 1.0 + self.delta)

class Imshow(VideoSpatialTransform):
    def __init__(self, winname='img', delay=0):
        self.delay = delay
        self.winname = winname

    def __call__(self, image):
        cv2.imshow(self.winname, image)
        cv2.waitKey(self.delay)
        return image


class RandomSharpness(VideoSpatialTransform):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __repr__(self):
        return self.__class__.__name__ + _repr_params(lower=self.lower, upper=self.upper)

    # expects float image
    def __call__(self, image):
        if self.rnd:
            if isinstance(image, np.ndarray):
                im = Image.fromarray(image)
            else:
                im = image.copy()
            enhancer = ImageEnhance.Sharpness(im)
            im = enhancer.enhance(self.factor)
            if isinstance(image, np.ndarray):
                image = np.asarray(im)
            else:
                image = im
        return image

    def randomize_parameters(self):
        self.rnd = random.randint(0, 1)
        self.factor = random.uniform(self.lower, self.upper)


class SwapChannels(VideoSpatialTransform):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        im2arr = np.array(image)
        im2arr = im2arr[:, :, self.swaps]
        image = Image.fromarray(im2arr)

        return image

    def __repr__(self):
        return self.__class__.__name__ + _repr_params(swaps=self.swaps)


class PhotometricDistort(VideoSpatialTransform):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            im = Image.fromarray(image)
        else:
            im = image.copy()

        im = self.rand_brightness(im)
        if self.rnd1:
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im = distort(im)

        im = self.rand_light_noise(im)
        if isinstance(image, np.ndarray):
            return np.asarray(im)
        return im

    def __repr__(self):
        return self.__class__.__name__ + _repr_params()

    def randomize_parameters(self):
        self.rnd1 = random.randint(0, 1)
        self.rand_brightness.randomize_parameters()
        self.rand_light_noise.randomize_parameters()
        for aug in self.pd:
            aug.randomize_parameters()
