import numpy as np
from .utils import move_channels_first, move_channels_last, load_single_image
from .normalize import zero_mean_normalize_image_data as unet3d_normalize
from .resample import resample as unet3d_resample
from .augment import permute_data, random_permutation_key


def compute_affine_from_point(point, window, spacing):
    affine = np.diag(np.ones(4))
    np.fill_diagonal(affine, list(spacing) + [1])
    window_extent = np.multiply(window, spacing)
    offset = window_extent/2
    affine[:3, 3] = point - offset
    return affine
    

def fetch_data_for_point(point, image, window, flip=False, interpolation='linear', spacing=None,
                         normalization_func=unet3d_normalize):
    if spacing is None:
        spacing = np.asarray(image.header.get_zooms())
    affine = compute_affine_from_point(point, window, spacing)
    _image = unet3d_resample(image, affine, window, interpolation)
    image_data = _image.get_data()
    if len(image_data.shape) == 3:
        image_data = image_data[..., None]
    ch_first = move_channels_first(image_data)
    if flip:
        ch_first = permute_data(ch_first, random_permutation_key())
    if normalization_func is not None:
        normalized = normalization_func(ch_first)
    else:
        normalized = ch_first
    image_data[:] = move_channels_last(normalized)
    return image_data

def get_label_indices(image):
    return np.stack(np.where(image.get_data() > 0), axis=-1)

def index_to_point(index, affine):
    return np.dot(affine, list(index) + [1])[:3]

def fetch_data(feature_filename, target_filename, input_window, 
               flip=False, interpolation='linear', n_points=1, spacing=(1, 1, 1),
               reorder=True, resample='continuous'):
    image = load_single_image(feature_filename, reorder=reorder, resample=resample)
    target_image = load_single_image(target_filename, reorder=reorder, resample=resample)
    indices = get_label_indices(target_image)
    _candidates = np.copy(indices)
    np.random.shuffle(_candidates)
    _candidates = _candidates.tolist()
    x = list()
    while len(x) < n_points:
        index = _candidates.pop()
        point = index_to_point(index, target_image.affine)
        with np.errstate(invalid='raise'):
            try:
                 data = fetch_data_for_point(point, image, input_window, flip=flip, 
                                             interpolation=interpolation, spacing=np.asarray(spacing))
            except FloatingPointError:
                continue
        x.append(data)
    return x 
