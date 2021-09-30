import os
from functools import partial
import numpy as np
import nibabel as nib
from tensorflow.keras.utils import Sequence
from nilearn.image import new_img_like, resample_to_img, reorder_img
import random
import warnings
from .nilearn_custom_utils.nilearn_utils import crop_img
from .radiomic_utils import fetch_data
from .utils import (copy_image, extract_sub_volumes, mask,
                    compile_one_hot_encoding, load_image,
                    get_nibabel_data, add_one_hot_encoding_contours)
from .normalize import (zero_mean_normalize_image_data,
                        foreground_zero_mean_normalize_image_data,
                        zero_floor_normalize_image_data, zero_one_window)
from . import normalize
from .resample import resample
from .augment import (scale_affine, add_noise, affine_swap_axis,
                        translate_affine, random_blur, random_permutation_x_y)
from .affine import resize_affine


def normalization_name_to_function(normalization_name):
    if type(normalization_name) == list:
        return partial(normalize_data_with_multiple_functions, normalization_names=normalization_name)
    elif normalization_name == "zero_mean":
        return zero_mean_normalize_image_data
    elif normalization_name == "foreground_zero_mean":
        return foreground_zero_mean_normalize_image_data
    elif normalization_name == "zero_floor":
        return zero_floor_normalize_image_data
    elif normalization_name == "zero_one_window":
        return zero_one_window
    elif normalization_name == "mask":
        return mask
    elif normalization_name is not None:
        try:
            return getattr(normalize, normalization_name)
        except AttributeError:
            raise NotImplementedError(normalization_name + " normalization is not available.")
    else:
        return lambda x, **kwargs: x


def normalize_image_with_function(image, function, volume_indices=None, **kwargs):
    data = get_nibabel_data(image)
    if volume_indices is not None:
        data[..., volume_indices] = function(data[..., volume_indices], **kwargs)
    else:
        data = function(data, **kwargs)
    return new_img_like(image, data=data, affine=image.affine)


def normalize_data_with_multiple_functions(data, normalization_names, channels_axis=3, **kwargs):
    """

    :param data:
    :param normalization_names:
    :param channels_axis:
    :param kwargs: sets the normalization parameters, but should have multiple sets of parameters for the individual
    normalization functions.
    :return:
    """
    normalized_data = list()
    for name in normalization_names:
        func = normalization_name_to_function(name)
        _kwargs = dict(kwargs[name]) if name in kwargs else None
        if _kwargs and "volume_indices" in _kwargs and _kwargs["volume_indices"] is not None:
            volume_indices = _kwargs.pop("volume_indices")
            _data = data[..., volume_indices]
        else:
            _data = data
        normalized_data.append(func(_data, **_kwargs))
    return np.concatenate(normalized_data, axis=channels_axis)


def augment_affine(affine, shape, augment_scale_std=None, augment_scale_probability=1,
                   flip_left_right_probability=0, augment_translation_std=None, augment_translation_probability=1,
                   flip_front_back_probability=0):
    if augment_scale_std and decision(augment_scale_probability):
        scale = np.random.normal(1, augment_scale_std, 3)
        affine = scale_affine(affine, shape, scale)
    if decision(flip_left_right_probability):  # flips the left and right sides of the image randomly
        affine = affine_swap_axis(affine, shape=shape, axis=0)
    if decision(flip_front_back_probability):
        affine = affine_swap_axis(affine, shape=shape, axis=1)
    if augment_translation_std and decision(augment_translation_probability):
        affine = translate_affine(affine, shape,
                                  translation_scales=np.random.normal(loc=0, scale=augment_translation_std, size=3))
    return affine


def augment_image(image, augment_blur_mean=None, augment_blur_std=None, augment_blur_probability=1,
                  additive_noise_std=None, additive_noise_probability=1):
    if not (augment_blur_mean is None or augment_blur_std is None) and decision(augment_blur_probability):
        image = random_blur(image, mean=augment_blur_mean, std=augment_blur_std)
    if additive_noise_std and decision(additive_noise_probability):
        image.dataobj[:] = add_noise(image.dataobj, sigma_factor=additive_noise_std)
    return image


def format_feature_image(feature_image, window, crop=False, cropping_kwargs=None, augment_scale_std=None,
                         augment_scale_probability=1, additive_noise_std=None, additive_noise_probability=0,
                         flip_left_right_probability=0, augment_translation_std=None,
                         augment_translation_probability=0, augment_blur_mean=None, augment_blur_std=None,
                         augment_blur_probability=0, flip_front_back_probability=0, reorder=False,
                         interpolation="linear"):
    # print("Input to format feature image is ",feature_image.shape)
    if reorder:
        feature_image = reorder_img(feature_image, resample=interpolation)
    if crop:
        if cropping_kwargs is None:
            cropping_kwargs = dict()
        affine, shape = crop_img(feature_image, return_affine=True, **cropping_kwargs)
        # print("The output shape by crop_img is", shape)
    else:
        affine = feature_image.affine.copy()
        shape = feature_image.shape
    affine = augment_affine(affine, shape,
                            augment_scale_std=augment_scale_std,
                            augment_scale_probability=augment_scale_probability,
                            augment_translation_std=augment_translation_std,
                            augment_translation_probability=augment_translation_probability,
                            flip_left_right_probability=flip_left_right_probability,
                            flip_front_back_probability=flip_front_back_probability)
    feature_image = augment_image(feature_image,
                                  augment_blur_mean=augment_blur_mean,
                                  augment_blur_std=augment_blur_std,
                                  augment_blur_probability=augment_blur_probability,
                                  additive_noise_std=additive_noise_std,
                                  additive_noise_probability=additive_noise_probability)
    # print("Output of feature image after augmentation is", feature_image.shape)
    affine = resize_affine(affine, shape, window)
    return feature_image, affine


def decision(probability):
    if not probability or probability <= 0:
        return False
    elif probability >= 1:
        return True
    else:
        return random.random() < probability


class BaseSequence(Sequence):
    def __init__(self, filenames, batch_size, target_labels, window, spacing, metric_names, shuffle=True,
                 points_per_subject=1, flip=False, reorder=False, iterations_per_epoch=1, deformation_augmentation=None,
                 base_directory=None, subject_ids=None, inputs_per_epoch=None, channel_axis=3, normalization="zero_mean", normalization_args=None ):
        self.deformation_augmentation = deformation_augmentation
        self.base_directory = base_directory
        self.subject_ids = subject_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filenames = filenames
        self.inputs_per_epoch = inputs_per_epoch
        self.metric_names = metric_names

        if self.inputs_per_epoch is not None:
            if not type(self.filenames) == dict:
                raise ValueError("'inputs_per_epoch' is not None, but 'filenames' is not a dictionary.")
            self.filenames_dict = self.filenames
        else:
            self.filenames_dict = None
        self.target_labels = target_labels
        self.window = window
        self.points_per_subject = points_per_subject
        self.flip = flip
        self.reorder = reorder
        self.spacing = spacing
        self.iterations_per_epoch = iterations_per_epoch
        self.subjects_per_batch = int(np.floor(self.batch_size / self.points_per_subject))
        assert self.subjects_per_batch > 0
        self.channel_axis = channel_axis
        self.on_epoch_end()
        self.normalize = normalization is not None
        self.normalization_func = normalization_name_to_function(normalization)
        if normalization_args is not None:
            self.normalization_kwargs = normalization_args
        else:
            self.normalization_kwargs = dict()

    def get_number_of_subjects_per_epoch(self):
        return self.get_number_of_subjects() * self.iterations_per_epoch

    def get_number_of_subjects(self):
        return len(self.filenames)

    def generate_epoch_filenames(self):
        if self.inputs_per_epoch is not None:
            self.sample_filenames()
        _filenames = list(self.filenames)
        epoch_filenames = list()
        for i in range(self.iterations_per_epoch):
            if self.shuffle:
                np.random.shuffle(_filenames)
            epoch_filenames.extend(_filenames)
        self.epoch_filenames = list(epoch_filenames)

    def switch_to_augmented_filename(self, subject_id, filename):
        augmented_filename = self.deformation_augmentation.format(base_directory=self.base_directory,
                                                                  random_subject_id=np.random.choice(self.subject_ids),
                                                                  subject_id=subject_id,
                                                                  basename=os.path.basename(filename))
        if not os.path.exists(augmented_filename):
            raise RuntimeWarning("Augmented filename {} does not exists!".format(augmented_filename))
        else:
            filename = augmented_filename
        return filename

    def __len__(self):
        return self.get_number_of_batches_per_epoch()

    def get_number_of_batches_per_epoch(self):
        return int(np.floor(np.divide(self.get_number_of_subjects_per_epoch(),
                                      self.subjects_per_batch)))

    def __getitem__(self, idx):
        batch_filenames = self.epoch_filenames[idx * self.subjects_per_batch:(idx + 1) * self.subjects_per_batch]
        batch_x = list()
        for feature_filename, target_filename in batch_filenames:
            for x in fetch_data(feature_filename,
                                        target_filename,
                                        self.window,
                                        n_points=self.points_per_subject,
                                        flip=self.flip,
                                        reorder=self.reorder,
                                        spacing=self.spacing,
                                        classify=self._classify):
                batch_x.append(x)
        return np.asarray(batch_x)

    def on_epoch_end(self):
        self.generate_epoch_filenames()

    def sample_filenames(self):
        """
        Sample the filenames.
        """
        filenames = list()
        for key in self.filenames_dict:
            if self.inputs_per_epoch[key] == "all":
                filenames.extend(self.filenames_dict[key])
            else:
                _filenames = list(self.filenames_dict[key])
                np.random.shuffle(_filenames)
                filenames.extend(_filenames[:self.inputs_per_epoch[key]])
        self.filenames = filenames

    def normalize_image(self, image):
        if self.normalize:
            return normalize_image_with_function(image, self.normalization_func, **self.normalization_kwargs)
        return image

    def get_feature_image(self, idx, return_unmodified=False):
        return self.format_feature_image(self.epoch_filenames[idx], return_unmodified=return_unmodified)


class AugumentSettings(BaseSequence):
    def __init__(self, filenames, batch_size, window, spacing, interpolation='linear', crop=True, cropping_kwargs=None,
                 augment_scale_std=0, augment_scale_probability=1, additive_noise_std=0, additive_noise_probability=1,
                 augment_blur_mean=None, augment_blur_std=None, augment_blur_probability=1,
                 augment_translation_std=None, augment_translation_probability=1, flip_left_right_probability=0,
                 flip_front_back_probability=0, resample=None, **kwargs):
        """

        :param interpolation: interpolation to use when formatting the feature image.
        :param crop: if true, images will be cropped to remove the background from the feature image.
        :param cropping_pad_width: width of the padding around the foreground after cropping.
        :param augment_scale_std: randomly augment the scale with this standard deviation (mean of 1). If None, 0 or
        False, no augmentation will be done.
        :param augment_scale_probability: If the scale augmentation is set, randomly pick when to implement.
        :param additive_noise_std:
        :param additive_noise_probability:
        :param augment_blur_mean:
        :param augment_blur_std:
        :param augment_blur_probability:
        :param augment_translation_std:
        :param augment_translation_probability:
        :param flip_left_right_probability:
        :param resample: deprecated
        :param kwargs:
        """
        super().__init__(filenames=filenames, batch_size=batch_size, target_labels=tuple(), window=window,
                         spacing=spacing, **kwargs)
        self.interpolation = interpolation
        if resample is not None:
            warnings.warn("'resample' argument is deprecated. Use 'interpolation'.", DeprecationWarning)
        self.crop = crop
        self.augment_scale_std = augment_scale_std
        self.augment_scale_probability = augment_scale_probability
        self.additive_noise_std = additive_noise_std
        self.additive_noise_probability = additive_noise_probability
        self.cropping_kwargs = cropping_kwargs
        self.augment_blur_mean = augment_blur_mean
        self.augment_blur_std = augment_blur_std
        self.augment_blur_probability = augment_blur_probability
        self.augment_translation_std = augment_translation_std
        self.augment_translation_probability = augment_translation_probability
        self.flip_left_right_probability = flip_left_right_probability
        self.flip_front_back_probability = flip_front_back_probability


    

class WholeVolumeSegmentationSequence(AugumentSettings):
    def __init__(self, *args, feature_index=0, extract_sub_volumes=False,feature_sub_volumes_index=1, target_sub_volumes_index=3, target_interpolation="nearest", target_index=2, labels=None, add_contours=False, random_permutation_probability=0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.target_interpolation = target_interpolation
        self.target_index = target_index
        self.labels = labels
        self.add_contours = add_contours
        self.random_permutation_probability = random_permutation_probability
        if target_interpolation is None:
            self.target_interpolation = self.interpolation
        else:
            self.target_interpolation = target_interpolation
        self.target_index = target_index
        self.feature_index = feature_index
        self.extract_sub_volumes = extract_sub_volumes
        self.feature_sub_volumes_index = feature_sub_volumes_index
        self.target_sub_volumes_index = target_sub_volumes_index
        self.random_permutation_probability = random_permutation_probability
    def permute_inputs(self, x, y):
        if decision(self.random_permutation_probability):
            x, y = random_permutation_x_y(x, y, channel_axis=self.channel_axis)
        return x, y

    def resample_image(self, input_filenames):
        feature_image = self.format_feature_image(input_filenames=input_filenames)
        target_image = self.resample_target(self.load_target_image(feature_image, input_filenames),
                                            feature_image)
        feature_image = augment_image(feature_image,
                                      additive_noise_std=self.additive_noise_std,
                                      additive_noise_probability=self.additive_noise_probability,
                                      augment_blur_mean=self.augment_blur_mean,
                                      augment_blur_std=self.augment_blur_std,
                                      augment_blur_probability=self.augment_blur_probability)
        return feature_image, target_image

    def load_image(self, filenames, index, force_4d=True, interpolation="linear", sub_volume_indices=None):
        filename = filenames[index]
        # Reordering is done when the image is formatted
        image = load_image(filename, force_4d=force_4d, reorder=False, interpolation=interpolation)
        if sub_volume_indices:
            image = extract_sub_volumes(image, sub_volume_indices)
        return image

    def load_feature_image(self, input_filenames):
        if self.extract_sub_volumes:
            sub_volume_indices = input_filenames[self.feature_sub_volumes_index]
        else:
            sub_volume_indices = None
        return self.load_image(input_filenames, self.feature_index, force_4d=True, interpolation=self.interpolation,
                               sub_volume_indices=sub_volume_indices)

    def format_feature_image(self, input_filenames, return_unmodified=False):
        unmodified_image = self.load_feature_image(input_filenames)
        image, affine = format_feature_image(feature_image=self.normalize_image(unmodified_image),
                                             crop=self.crop,
                                             cropping_kwargs=self.cropping_kwargs,
                                             augment_scale_std=self.augment_scale_std,
                                             augment_scale_probability=self.augment_scale_probability,
                                             window=self.window,
                                             additive_noise_std=None,  # augmented later
                                             augment_blur_mean=None,  # augmented later
                                             augment_blur_std=None,  # augmented later
                                             flip_left_right_probability=self.flip_left_right_probability,
                                             flip_front_back_probability=self.flip_front_back_probability,
                                             augment_translation_std=self.augment_translation_std,
                                             augment_translation_probability=self.augment_translation_probability,
                                             reorder=self.reorder,
                                             interpolation=self.interpolation)
        resampled = resample(image, affine, self.window, interpolation=self.interpolation)
        if return_unmodified:
            return resampled, unmodified_image
        else:
            return resampled

    def load_target_image(self, feature_image, input_filenames):
        if self.target_index is None:
            target_image = copy_image(feature_image)
        else:
            if self.extract_sub_volumes:
                sub_volume_indices = input_filenames[self.target_sub_volumes_index]
            else:
                sub_volume_indices = None
            target_image = self.load_image(input_filenames, self.target_index, force_4d=True,
                                           sub_volume_indices=sub_volume_indices,
                                           interpolation=self.target_interpolation)
        return target_image

    def resample_target(self, target_image, feature_image):
        target_image = resample_to_img(target_image, feature_image, interpolation=self.target_interpolation)
        return target_image

    def resample_input(self, input_filenames):
        input_image, target_image = self.resample_image(input_filenames)
        target_data = get_nibabel_data(target_image)
        if self.labels is None:
            self.labels = np.unique(target_data)
        assert len(target_data.shape) == 4
        if target_data.shape[3] == 1:
            target_data = np.moveaxis(
                compile_one_hot_encoding(np.moveaxis(target_data, 3, 0),
                                         n_labels=len(self.labels),
                                         labels=self.labels,
                                         return_4d=True), 0, 3)
        else:
            _target_data = list()
            for channel, labels in zip(range(target_data.shape[self.channel_axis]), self.labels):
                _target_data.append(np.moveaxis(
                    compile_one_hot_encoding(np.moveaxis(target_data[..., channel, None], self.channel_axis, 0),
                                             n_labels=len(labels),
                                             labels=labels,
                                             return_4d=True), 0, self.channel_axis))
            target_data = np.concatenate(_target_data, axis=self.channel_axis)
        if self.add_contours:
            target_data = add_one_hot_encoding_contours(target_data)
        return self.permute_inputs(get_nibabel_data(input_image), target_data)
