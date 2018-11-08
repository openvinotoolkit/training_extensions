import math
import numpy as np
from toolbox.layers import get_spatial_dims


def prior_box_specs(blob, image_size, box_specs, step, clip=False, offset=0.5, variance=None, data_format='NHWC'):
  """
  Generates numpy array of priors in caffe format


  :param blob: input feature blob  (we only need spatial dimension from it)
  :param image_size: input image size (height, width)
  :param box_specs: list of pairs [size, aspect_ratio]
  :param step:
  :param flip: flip each aspect ration or not
  :param clip: clip priors to image bounding box
  :param offset: a subpixel offset for priors location
  :param variance: optional array of lenghts 4 with variances to encode inpriors array
  :param data_format: NHWC or NCHW
  """
  assert isinstance(box_specs, list)
  assert variance is None or len(variance) == 4
  assert data_format in ['NHWC', 'NCHW']

  if isinstance(step, (list, tuple)):
    step_y, step_x = step
  else:
    step_y, step_x = step, step

  if len(blob.get_shape()) == 2:
    layer_height = layer_width = 1
  else:
    layer_height, layer_width = get_spatial_dims(blob, data_format)

  anchors = []
  for height in range(layer_height):
    for width in range(layer_width):
      center_y = (height + offset) * step_y
      center_x = (width + offset) * step_x

      for size, aspect_ratio in box_specs:
        box_w = size * math.sqrt(aspect_ratio)
        box_h = size / math.sqrt(aspect_ratio)
        xmin = (center_x - box_w / 2.) / image_size[1]
        ymin = (center_y - box_h / 2.) / image_size[0]
        xmax = (center_x + box_w / 2.) / image_size[1]
        ymax = (center_y + box_h / 2.) / image_size[0]
        anchors.extend([xmin, ymin, xmax, ymax])

  if clip:
    anchors = np.clip(anchors, 0., 1.).tolist()

  num_priors_per_pixel = len(anchors) // (layer_height * layer_width * 4)
  num_priors_alt_formula = len(box_specs)
  assert num_priors_per_pixel == num_priors_alt_formula

  if variance:
    anchors.extend(list(variance) * layer_height * layer_width * num_priors_per_pixel)

  top_shape = 1, (2 if variance else 1), layer_height * layer_width * num_priors_per_pixel * 4

  assert len(anchors) == np.prod(top_shape)
  priors_array = np.array([anchors], dtype=np.float32).reshape(top_shape)
  return priors_array, num_priors_per_pixel


def prior_box(blob, image_size, min_sizes, aspect_ratios, step, max_sizes=None, flip=True, clip=False, offset=0.5,
              variance=None, data_format='NHWC'):
  """
  Generates numpy array of priors in caffe format


  :param blob: input feature blob  (we only need spatial dimension from it)
  :param image_size: input image size (height, width)
  :param min_sizes:
  :param aspect_ratios:
  :param step:
  :param max_sizes:
  :param flip: flip each aspect ration or not
  :param clip: clip priors to image bounding box
  :param offset: a subpixel offset for priors location
  :param variance: optional array of lenghts 4 with variances to encode inpriors array
  :param data_format: NHWC or NCHW
  """
  assert isinstance(min_sizes, list) and isinstance(aspect_ratios, list)
  assert not max_sizes or len(max_sizes) == len(min_sizes)
  assert variance is None or len(variance) == 4

  max_sizes = max_sizes or []

  ratios = []
  for aspect_ratio in aspect_ratios:
    close = [r for r in ratios if math.fabs(aspect_ratio - r) < 1e-6]
    if not close:
      ratios.append(aspect_ratio)
      if flip:
        ratios.append(1. / aspect_ratio)

  box_specs = []

  for (ind, min_size) in enumerate(min_sizes):
    box_specs.append([min_size, 1.])

    if max_sizes:
      max_size = max_sizes[ind]
      box_specs.append([math.sqrt(min_size * max_size), 1.])

    # rest of priors
    for aspect_ratio in ratios:
      assert math.fabs(aspect_ratio - 1.) >= 1e-6
      box_specs.append([min_size, math.sqrt(aspect_ratio)])

  return prior_box_specs(blob, image_size, box_specs, step, clip, offset, variance, data_format)


def prior_box_clusterd(blob, image_size, clustered_sizes, step, clip=False, offset=0.5, variance=None,
                       data_format='NHWC'):
  """
  Generates numpy array of priors in caffe format

  :param blob: input feature blob  (we only need spatial dimension from it)
  :param image_size: input image size (height, width)
  :param clustered_sizes: list of (height, width) tuples
  :param step:
  :param clip: clip priors to image bounding box
  :param offset: a subpixel offset for priors location
  :param variance: optional array of lenghts 4 with variances to encode inpriors array
  :param data_format: NHWC or NCHW
  """
  assert variance is None or len(variance) == 4
  assert data_format in ['NHWC', 'NCHW']

  if isinstance(step, (list, tuple)):
    step_y, step_x = step
  else:
    step_y, step_x = step, step

  if len(blob.get_shape()) == 2:
    layer_height = layer_width = 1
  else:
    layer_height, layer_width = get_spatial_dims(blob, data_format)

  num_priors_per_pixel = len(clustered_sizes)
  top_shape = 1, (2 if variance else 1), layer_height * layer_width * num_priors_per_pixel * 4

  anchors = []
  for height in range(layer_height):
    for width in range(layer_width):
      center_x = (width + offset) * step_x / image_size[1]
      center_y = (height + offset) * step_y / image_size[0]

      for (box_rows, box_cols) in clustered_sizes:
        xmin = center_x - box_cols / 2.
        ymin = center_y - box_rows / 2.
        xmax = center_x + box_cols / 2.
        ymax = center_y + box_rows / 2.
        anchors.extend([xmin, ymin, xmax, ymax])

  if clip:
    anchors = np.clip(anchors, 0., 1.).tolist()

  if variance:
    anchors.extend(list(variance) * layer_width * layer_height * num_priors_per_pixel)

  assert len(anchors) == np.prod(top_shape)
  priors_array = np.array([anchors], dtype=np.float32).reshape(top_shape)
  return priors_array, num_priors_per_pixel
