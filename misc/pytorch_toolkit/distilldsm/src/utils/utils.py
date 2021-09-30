import nibabel as nib
import numpy as np
import json
from nilearn.image import resample_to_img, reorder_img, new_img_like
from scipy.ndimage import binary_erosion

def load_json(filename):
    with open(filename, 'r') as opened_file:
        return json.load(opened_file)

def dump_json(dataobj, filename):
    with open(filename, 'w') as opened_file:
        json.dump(dataobj, opened_file)

def compile_one_hot_encoding(data, n_labels, labels=None, dtype=np.uint8, return_4d=True):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :param dtype: output type of the array
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    data = np.asarray(data)
    while len(data.shape) < 5:
        data = data[None]
    assert data.shape[1] == 1
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, dtype=dtype)
    for label_index in range(n_labels):
        if labels is not None:
            if type(labels[label_index]) == list:
                # lists of labels will group together multiple labels from the label map into a single one-hot channel.
                for label in labels[label_index]:
                    y[:, label_index][data[:, 0] == label] = 1
            else:
                y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    if return_4d:
        assert y.shape[0] == 1
        y = y[0]
    return y

def convert_one_hot_to_label_map(one_hot_encoding, labels, axis=3, threshold=0.5, sum_then_threshold=False,
                                 dtype=np.int16, label_hierarchy=False):
    if label_hierarchy:
        return convert_one_hot_to_label_map_using_hierarchy(one_hot_encoding, labels, axis=axis, threshold=threshold,
                                                            dtype=dtype)
    else:
        # print(labels)
        if all([type(_labels) == list for _labels in labels]):
            # output the segmentation label maps into multiple volumes
            i = 0
            label_maps = list()
            for _labels in labels:
                _data = one_hot_encoding[..., i:i+len(_labels)]
                label_maps.append(convert_one_hot_to_label_map(_data, labels=_labels, axis=axis, threshold=threshold,
                                                               sum_then_threshold=sum_then_threshold, dtype=dtype))
                i = i + len(_labels)
            label_map = np.stack(label_maps, axis=axis)
        else:
            label_map = convert_one_hot_to_single_label_map_volume(one_hot_encoding, labels, threshold, axis,
                                                                   sum_then_threshold, dtype)
        return label_map

def convert_one_hot_to_single_label_map_volume(one_hot_encoding, labels, threshold=0.5, axis=3,
                                               sum_then_threshold=False, dtype=np.int16):
    # output a single label map volume
    segmentation_mask = mask_encoding(one_hot_encoding, len(labels), threshold=threshold, axis=axis,
                                      sum_then_threshold=sum_then_threshold)
    return assign_labels(one_hot_encoding, segmentation_mask, labels=labels, axis=axis,
                         dtype=dtype, label_indices=np.arange(len(labels)))

def mask_encoding(one_hot_encoding, n_labels, threshold=0.5, axis=3, sum_then_threshold=False):
    if sum_then_threshold:
        return np.sum(one_hot_encoding[..., :n_labels], axis=axis) > threshold
    else:
        return np.any(one_hot_encoding[..., :n_labels] > threshold, axis=axis)

def assign_labels(one_hot_encoding, segmentation_mask, labels, label_indices, axis=3, dtype=np.int16):
    max_arg_map = np.zeros(one_hot_encoding.shape[:axis], dtype=dtype)
    label_map = np.copy(max_arg_map)
    max_arg_map[segmentation_mask] = (np.argmax(one_hot_encoding[..., label_indices],
                                                axis=axis) + 1)[segmentation_mask]
    for index, label in enumerate(labels):
        label_map[max_arg_map == (index + 1)] = label
    return label_map

def convert_one_hot_to_label_map_using_hierarchy(one_hot_encoding, labels, threshold=0.5, axis=3, dtype=np.int16):
    # print(one_hot_encoding.shape, axis, labels)
    if (one_hot_encoding.ndim == 3):
        one_hot_encoding = np.expand_dims(one_hot_encoding, axis=3)
    # print(one_hot_encoding.shape)
    roi = np.ones(one_hot_encoding.shape[:axis], dtype=np.bool)
    label_map = np.zeros(one_hot_encoding.shape[:axis], dtype=dtype)
    for index, label in enumerate(labels):
        roi = np.logical_and(one_hot_encoding[..., index] > threshold, roi)
        label_map[roi] = label
    return label_map

def one_hot_image_to_label_map(one_hot_image, labels, axis=3, threshold=0.5, sum_then_threshold=True, dtype=np.int16,
                               label_hierarchy=None):
    label_map = convert_one_hot_to_label_map(get_nibabel_data(one_hot_image), labels=labels, axis=axis,
                                             threshold=threshold, sum_then_threshold=sum_then_threshold,
                                             dtype=dtype, label_hierarchy=label_hierarchy)
    return new_img_like(one_hot_image, label_map)

def copy_image(image):
    return image.__class__(np.copy(image.dataobj), image.affine)

def combine_images(images, axis=0, resample_unequal_affines=False, interpolation="linear"):
    base_image = images[0]
    data = list()
    max_dim = len(base_image.shape)
    for image in images:
        try:
            np.testing.assert_array_equal(image.affine, base_image.affine)
        except AssertionError as error:
            if resample_unequal_affines:
                image = resample_to_img(image, base_image, interpolation=interpolation)
            else:
                raise error
        image_data = image.get_data()
        dim = len(image.shape)
        if dim < max_dim:
            image_data = np.expand_dims(image_data, axis=axis)
        elif dim > max_dim:
            max_dim = max(max_dim, dim)
            data = [np.expand_dims(x, axis=axis) for x in data]
        data.append(image_data)
    if len(data[0].shape) > 3:
        array = np.concatenate(data, axis=axis)
    else:
        array = np.stack(data, axis=axis)
    return base_image.__class__(array, base_image.affine)

def move_channels_last(data):
    return np.moveaxis(data, 0, -1)

def move_channels_first(data):
    return np.moveaxis(data, -1, 0)

def nib_load_files(filenames, reorder=False, interpolation="linear"):
    if type(filenames) != list:
        filenames = [filenames]
    return [load_image(filename, reorder=reorder, interpolation=interpolation, force_4d=False)
            for filename in filenames]

def load_image(filename, feature_axis=3, resample_unequal_affines=True, interpolation="linear", force_4d=False,
               reorder=False):
    """
    :param feature_axis: axis along which to combine the images, if necessary.
    :param filename: can be either string path to the file or a list of paths.
    :return: image containing either the 1 image in the filename or a combined image based on multiple filenames.
    """

    if type(filename) != list:
        if not force_4d:
            return load_single_image(filename=filename, resample=interpolation, reorder=reorder)
        else:
            filename = [filename]

    return combine_images(nib_load_files(filename, reorder=reorder, interpolation=interpolation), axis=feature_axis,
                          resample_unequal_affines=resample_unequal_affines, interpolation=interpolation)

def load_single_image(filename, resample=None, reorder=True):
    image = nib.load(filename)
    if reorder:
        return reorder_img(image, resample=resample)
    return image

def estimate_binary_contour(binary):
    return np.logical_xor(binary, binary_erosion(binary, iterations=1))

def extract_sub_volumes(image, sub_volume_indices):
    data = image.dataobj[..., sub_volume_indices]
    return new_img_like(ref_niimg=image, data=data)

def mask(data, threshold=0, dtype=np.float):
    return np.asarray(data > threshold, dtype=dtype)

def get_nibabel_data(nibabel_image):
    return nibabel_image.get_fdata()

def in_config(string, dictionary, if_not_in_config_return=None):
    return dictionary[string] if string in dictionary else if_not_in_config_return

def add_one_hot_encoding_contours(one_hot_encoding):
    new_encoding = np.zeros(one_hot_encoding.shape[:-1] + (one_hot_encoding.shape[-1] * 2,),
                            dtype=one_hot_encoding.dtype)
    new_encoding[..., :one_hot_encoding.shape[-1]] = one_hot_encoding
    for index in range(one_hot_encoding.shape[-1]):
        new_encoding[..., one_hot_encoding.shape[-1] + index] = estimate_binary_contour(
            one_hot_encoding[..., index] > 0)
    return new_encoding
