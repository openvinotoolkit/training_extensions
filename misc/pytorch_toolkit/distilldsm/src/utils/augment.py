import numpy as np
from nilearn.image import smooth_img
import random
import itertools
from collections.abc import Iterable
from .affine import get_spacing_from_affine, assert_affine_is_diagonal


def add_noise(data, mean=0., sigma_factor=0.1):
    """
    Adds Gaussian noise.
    :param data: input numpy array
    :param mean: mean of the additive noise
    :param sigma_factor: standard deviation of the image will be multiplied by sigma_factor to obtain the standard
    deviation of the additive noise. Assumes standard deviation is the same for all channels.
    :return:
    """
    sigma = np.std(data) * sigma_factor
    noise = np.random.normal(mean, sigma, data.shape)
    return np.add(data, noise)

def random_permutation_key():
    """
    Generates and randomly selects a permutation key. See the documentation for the
    "generate_permutation_keys" function.
    """
    return random.choice(list(generate_permutation_keys()))

def generate_permutation_keys():
    """
    This function returns a set of "keys" that represent the 48 unique rotations &
    reflections of a 3D matrix.

    Each item of the set is a tuple:
    ((rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)

    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.

    48 unique rotations & reflections:
    https://en.wikipedia.org/wiki/Octahedral_symmetry#The_isometries_of_the_cube
    """
    return set(itertools.product(
        itertools.combinations_with_replacement(range(2), 2), range(2), range(2), range(2), range(2)))

def permute_data(data, key):
    """
    Permutes the given data according to the specification of the given key. Input data
    must be of shape (n_modalities, x, y, z).

    Input key is a tuple: (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)

    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.
    """
    data = np.copy(data)
    (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = key

    if rotate_y != 0:
        data = np.rot90(data, rotate_y, axes=(1, 3))
    if rotate_z != 0:
        data = np.rot90(data, rotate_z, axes=(2, 3))
    if flip_x:
        data = data[:, ::-1]
    if flip_y:
        data = data[:, :, ::-1]
    if flip_z:
        data = data[:, :, :, ::-1]
    if transpose:
        for i in range(data.shape[0]):
            data[i] = data[i].T
    return data

def random_permutation_x_y(x_data, y_data, channel_axis=0):
    """
    Performs random permutation on the data.
    :param x_data: numpy array containing the data. Data must be of shape (n_labels, x, y, z).
    :param y_data: numpy array containing the data. Data must be of shape (n_labels, x, y, z).
    :param channel_axis: if the channels are not in the first axis of the array (channel_axis != 0) then the channel
    axis will be moved to the first position for permutation and then moved back to the original position.
    :return: the permuted data
    """
    key = random_permutation_key()
    if channel_axis != 0:
        return [np.moveaxis(permute_data(np.moveaxis(data, channel_axis, 0), key), 0, channel_axis)
                for data in (x_data, y_data)]
    else:
        return permute_data(x_data, key), permute_data(y_data, key)

def translate_affine(affine, shape, translation_scales, copy=True):
    """
    :param translation_scales: (tuple) Contains x, y, and z translations scales from -1 to 1. 0 is no translation.
    1 is a forward (RAS-wise) translation of the entire image extent for that direction. -1 is a translation in the
    negative direction of the entire image extent. A translation of 1 is impractical for most purposes, though, as it
    moves the image out of the original field of view almost entirely. To perform a random translation, you can
    use numpy.random.normal(loc=0, scale=sigma, size=3) where sigma is the percent of image translation that would be
    randomly translated on average (0.05 for example).
    :return: affine
    """
    if copy:
        affine = np.copy(affine)
    spacing = get_spacing_from_affine(affine)
    extent = np.multiply(shape, spacing)
    translation = np.multiply(translation_scales, extent)
    affine[:3, 3] += translation
    return affine

def find_center(affine, shape, ndim=3):
    return np.matmul(affine,
                     list(np.divide(shape[:ndim], 2)) + [1])[:ndim]

def scale_affine(affine, shape, scale, ndim=3):
    """
    This assumes that the shape stays the same.
    :param affine: affine matrix for the image.
    :param shape: current shape of the data. This will remain the same.
    :param scale: iterable with length ndim, int, or float. A scale greater than 1 indicates the image will be zoomed,
    the spacing will get smaller, and the affine window will be smaller as well. A scale of less than 1 indicates
    zooming out with the spacing getting larger and the affine window getting bigger.
    :param ndim: number of dimensions (default is 3).
    :return:
    """
    if not isinstance(scale, Iterable):
        scale = np.ones(ndim) * scale
    else:
        scale = np.asarray(scale)

    # 1. find the image center
    center = find_center(affine, shape, ndim=ndim)

    # 2. translate the affine
    affine = affine.copy()
    origin = affine[:ndim, ndim]
    t = np.diag(np.ones(ndim + 1))
    t[:ndim, ndim] = (center - origin) * (1 - 1 / scale)
    affine = np.matmul(t, affine)

    # 3. scale the affine
    s = np.diag(list(1 / scale) + [1])
    affine = np.matmul(affine, s)
    return affine

def random_blur(image, mean, std):
    """
    mean: mean fwhm in millimeters.
    std: standard deviation of fwhm in millimeters.
    """
    return smooth_img(image, fwhm=np.abs(np.random.normal(mean, std, 3)).tolist())


def affine_swap_axis(affine, shape, axis=0):
    assert_affine_is_diagonal(affine)
    new_affine = np.copy(affine)
    origin = affine[axis, 3]
    new_affine[axis, 3] = origin + shape[axis] * affine[axis, axis]
    new_affine[axis, axis] = -affine[axis, axis]
    return new_affine
