import bisect

import cv2
import numpy as np

from ssd_detector.toolbox.bounding_box import *


# #############################################################################
# ################# Parameters and Transformer ################################
# #############################################################################

class ResizeParameter:
  WARP = 'WARP'
  FIT_SMALL_SIZE = 'FIT_SMALL_SIZE'
  FIT_LARGE_SIZE_AND_PAD = 'FIT_LARGE_SIZE_AND_PAD'

  def __init__(self, prob=1.0, resize_mode=WARP, height=0, width=0, height_scale=0., width_scale=0.,
               pad_mode=cv2.BORDER_CONSTANT, pad_value=(0, 0, 0),
               interp_mode=(cv2.INTER_LINEAR,
                            cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4)):
    self.prob = prob
    self.resize_mode = resize_mode
    self.height = height
    self.width = width
    self.height_scale = height_scale
    self.width_scale = width_scale
    self.pad_mode = pad_mode
    self.pad_value = pad_value
    self.interp_mode = interp_mode or []


class NoiseParameter:
  def __init__(self, prob=0., hist_eq=False, inverse=False, decolorize=False, gauss_blur=False, jpeg=-1,
               posterize=False,
               erode=False, clahe=False, saltpepper_fraction=0, saltpepper_values=None, convert_to_hsv=False,
               convert_to_lab=False):
    self.prob = prob
    self.hist_eq = hist_eq
    self.inverse = inverse
    self.decolorize = decolorize
    self.gauss_blur = gauss_blur
    self.jpeg = jpeg
    self.posterize = posterize
    self.erode = erode
    self.clahe = clahe
    self.saltpepper_fraction = saltpepper_fraction
    self.saltpepper_values = saltpepper_values
    self.convert_to_hsv = convert_to_hsv
    self.convert_to_lab = convert_to_lab


class DistortionParameter:
  def __init__(self, brightness_prob=0., brightness_delta=0., contrast_prob=0., contrast_lower=0., contrast_upper=0.,
               hue_prob=0., hue_delta=0., saturation_prob=0., saturation_lower=0., saturation_upper=0.,
               random_order_prob=0.):
    self.brightness_prob = brightness_prob
    self.brightness_delta = brightness_delta
    self.contrast_prob = contrast_prob
    self.contrast_lower = contrast_lower
    self.contrast_upper = contrast_upper
    self.hue_prob = hue_prob
    self.hue_delta = hue_delta
    self.saturation_prob = saturation_prob
    self.saturation_lower = saturation_lower
    self.saturation_upper = saturation_upper
    self.random_order_prob = random_order_prob


class ExpansionParameter:
  def __init__(self, prob=1., max_expand_ratio=1., fill_with_current_image_mean=False):
    self.prob = prob
    self.max_expand_ratio = max_expand_ratio
    self.fill_with_current_image_mean = fill_with_current_image_mean


class EmitConstraint:
  CENTER = 'CENTER'
  MIN_OVERLAP = 'MIN_OVERLAP'

  def __init__(self, emit_type=CENTER, emit_overlap=0):
    self.emit_type = emit_type
    self.emit_overlap = emit_overlap


class BatchSampler:
  def __init__(self, use_original_image=True, max_sample=0, max_trials=100,
               # from sampler
               min_scale=1., max_scale=1., min_aspect_ratio=1., max_aspect_ratio=1.,
               # from sample_constraint
               min_jaccard_overlap=None, max_jaccard_overlap=None, min_sample_coverage=None, max_sample_coverage=None,
               min_object_coverage=None, max_object_coverage=None):
    self.use_original_image = use_original_image
    self.max_sample = max_sample
    self.max_trials = max_trials
    self.min_scale = min_scale
    self.max_scale = max_scale
    self.min_aspect_ratio = min_aspect_ratio
    self.max_aspect_ratio = max_aspect_ratio

    self.min_jaccard_overlap = min_jaccard_overlap
    self.max_jaccard_overlap = max_jaccard_overlap
    self.min_sample_coverage = min_sample_coverage
    self.max_sample_coverage = max_sample_coverage
    self.min_object_coverage = min_object_coverage
    self.max_object_coverage = max_object_coverage


class TransformationParameter:
  def __init__(self, scale=1., mirror=False, crop_size=(0, 0), mean_value=None,
               resize_param=None, noise_param=None, distort_param=None, expand_param=None, emit_constraint=None):
    self.scale = scale
    self.mirror = mirror
    self.crop_size = crop_size
    self.mean_value = mean_value
    self.resize_param = resize_param
    self.noise_param = noise_param
    self.distort_param = distort_param
    self.expand_param = expand_param
    self.emit_constraint = emit_constraint


class DataTransformer:
  """
  Data transformer that mimics caffe behaviour

  """

  def __init__(self, is_training=True, transform_param=None):
    """
    DataTransformer constructor

    :type is_training: specified test or train phase
    :type transform_param: Transformation parameter
    """
    self.is_training = is_training
    self.transform_param = transform_param or TransformationParameter()

  def _infer_top_shape(self, image):
    """
    Infers transformed blob shape from parameters

    :type image: input image
    """
    image_h, image_w = image.shape[0:2]

    if self.transform_param.resize_param:
      original_aspect = float(image_w) / image_h
      image_h = self.transform_param.resize_param.height
      image_w = self.transform_param.resize_param.width
      if self.transform_param.resize_param.resize_mode == ResizeParameter.FIT_SMALL_SIZE:
        aspect = float(image_w) / image_h
        if original_aspect < aspect:
          image_h = int(image_w / original_aspect)
        else:
          image_w = int(original_aspect * image_h)

    crop_w, crop_h = self.transform_param.crop_size
    return (crop_h, crop_w) if crop_h * crop_w > 0 else (image_h, image_w)

  def _transform_image(self, image):
    """
    Geometrically transforms image according to parameters

    :type image: input image
    :rtype: transformed image
    """
    top_h, top_w = self._infer_top_shape(image)

    scale = self.transform_param.scale
    crop_w, crop_h = self.transform_param.crop_size
    do_mirror = self.transform_param.mirror and random.randint(0, 1)

    if self.transform_param.resize_param:
      image = apply_resize(image, self.transform_param.resize_param)

    if self.transform_param.noise_param:
      image = apply_noise(image, self.transform_param.noise_param)

    image_h, image_w = image.shape[0:2]

    w_off, h_off = (0, 0)
    if crop_h * crop_w > 0:
      if self.is_training:
        def exclusive_random(val):
          return random.randint(0, val - 1)

        h_off = exclusive_random(image_h - crop_h + 1)
        w_off = exclusive_random(image_w - crop_w + 1)
      else:
        h_off = (image_h - crop_h) / 2
        w_off = (image_w - crop_w) / 2
      image = image[h_off:h_off + crop_h, w_off:w_off + crop_w]

    xmin = float(w_off) / image_w
    ymin = float(h_off) / image_h
    xmax = float(w_off + top_w) / image_w
    ymax = float(h_off + top_h) / image_h
    crop_bbox = BoundingBox(xmin, ymin, xmax, ymax)

    if do_mirror:
      image = cv2.flip(image, 1)

    if self.transform_param.mean_value:
      image = image.astype(np.float32)
      image -= self.transform_param.mean_value

    if math.fabs(scale - 1) > 1e-2:
      image = image.astype(np.float32)
      image *= scale

    return image, crop_bbox, do_mirror

  def _meet_emit_constraint(self, src_bbox, bbox):
    """
    Checks if a bbox meet emit constraint w.r.t. src_bbox.
    :type src_bbox: source box
    :type bbox: input box
    """
    emit_constraint = self.transform_param.emit_constraint
    if not emit_constraint:
      return True  # satisfy if no constraint

    if emit_constraint.emit_type == EmitConstraint.CENTER:
      x_center = (bbox.xmin + bbox.xmax) / 2
      y_center = (bbox.ymin + bbox.ymax) / 2
      return src_bbox.xmin <= x_center <= src_bbox.xmax and src_bbox.ymin <= y_center <= src_bbox.ymax
    if emit_constraint.emit_type == EmitConstraint.MIN_OVERLAP:
      bbox_coverage = box_coverage(bbox, src_bbox)
      return bbox_coverage > emit_constraint.emit_overlap

  def _transform_annotation(self, original_shape, annotation, crop_bbox, do_mirror=False, do_resize=False):
    """
    Geometrically transforms annotation using parameters and crop_box used for similar image transformation

    :type original_shape: original image shape before transform
    :type annotation: annotation to transform
    :type crop_bbox: crop box used to transform image
    :type do_mirror: if mirror annotation using parameters
    :type do_resize: if resize annotation using parameters
    :rtype: transformed annotation
    """
    image_h, image_w = original_shape[0:2]
    transformed_annotation = {}

    for label, boxes in annotation.items():
      transformed_boxes = []

      for box in boxes:

        if do_resize and self.transform_param.resize_param:
          box = update_bbox_by_resize_policy(image_w, image_h, box, self.transform_param.resize_param)

        if not self._meet_emit_constraint(crop_bbox, box):
          continue

        proj_bbox = crop_bbox.project_box(box)
        if proj_bbox:
          if do_mirror:
            temp = proj_bbox.xmin
            proj_bbox.xmin = 1 - proj_bbox.xmax
            proj_bbox.xmax = 1 - temp

          if do_resize and self.transform_param.resize_param:
            proj_bbox = extrapolate_box(self.transform_param.resize_param, image_h, image_w, crop_bbox,
                                        proj_bbox)

          transformed_boxes.append(proj_bbox)

      if transformed_boxes:
        transformed_annotation[label] = transformed_boxes

    return transformed_annotation

  def transform(self, image, annotation):
    """
    Simultaneously transforms image and annotation according to parameters

    :param image: image to transform
    :param annotation: annotation to transform
    :return: transformed (image, annotation) tuple
    """
    transformed_image, crop_bbox, do_mirror = self._transform_image(image)
    transformed_annotation = self._transform_annotation(image.shape, annotation, crop_bbox, do_mirror,
                                                        do_resize=True)

    return transformed_image, transformed_annotation

  def distort_image(self, image):
    """
    Distorts image photometrically according to parameters

    :param image: image to transform
    :return: transformed image
    """
    if self.transform_param.distort_param:
      return apply_distort(image, self.transform_param.distort_param)
    return image

  def expand_image(self, image, annotation):
    """
    Expands image and annotaion according to parameters

    :param image: image to expand
    :param annotation: annotation to expand
    :return: transformed (image, annotation) tuple
    """
    expand_param = self.transform_param.expand_param
    if not expand_param:
      return image, annotation

    prob = random.random()
    if prob > expand_param.prob:
      return image, annotation

    if math.fabs(expand_param.max_expand_ratio - 1.) < 1e-2:
      return image, annotation

    expand_ratio = random.uniform(1., expand_param.max_expand_ratio)
    if expand_param.fill_with_current_image_mean:
      mean = np.mean(image, axis=(0, 1))
    else:
      mean = self.transform_param.mean_value

    expand_img, expand_bbox = expand_image(image, expand_ratio, mean)
    expand_annotation = self._transform_annotation(image.shape, annotation, expand_bbox)

    return expand_img, expand_annotation

  def crop_image(self, image, annotation, bbox):
    """
    Cropts image and acnnotaion according to bbox

    :param image: image to crop
    :param annotation: annotaion to crop
    :param bbox: crop box
    :return: transformed (image, annotation) tuple
    """
    cropped_image = crop_image(image, bbox)

    crop_bbox = bbox.clip_box()
    cropped_annotation = self._transform_annotation(image.shape, annotation, crop_bbox)
    return cropped_image, cropped_annotation


# #############################################################################
# ################# Transformer and Sampler ###################################
# #############################################################################

def create_default_transform_parameters(height=300, width=300):
  resize_param = ResizeParameter(height=height, width=width)
  emit_constraint = EmitConstraint(emit_type='CENTER')
  distort_param = DistortionParameter(brightness_prob=0.5, brightness_delta=32., contrast_prob=0.5,
                                      contrast_lower=0.5, contrast_upper=1.5,
                                      hue_prob=0.5, hue_delta=18, saturation_prob=0.5, saturation_lower=0.5,
                                      saturation_upper=1.5, random_order_prob=0.5)

  expand_param = ExpansionParameter(prob=0.5, max_expand_ratio=2.0)
  train_param = TransformationParameter(mirror=True, resize_param=resize_param, distort_param=distort_param,
                                        expand_param=expand_param, emit_constraint=emit_constraint)

  val_param = TransformationParameter(resize_param=resize_param)
  return train_param, val_param


def create_default_samplers():
  samplers = [BatchSampler(max_sample=1, max_trials=1)]

  for overlap in [0.1, 0.3, 0.5, 0.7, 0.9]:
    sampler = BatchSampler(min_scale=0.3, max_scale=1.0, min_aspect_ratio=0.5, max_aspect_ratio=2.0, max_sample=1,
                           max_trials=50)
    sampler.min_jaccard_overlap = overlap
    samplers.append(sampler)

  sampler = BatchSampler(min_scale=0.3, max_scale=1.0, min_aspect_ratio=0.5, max_aspect_ratio=2.0, max_sample=1,
                         max_trials=50)
  sampler.max_jaccard_overlap = 1.0
  samplers.append(sampler)
  return samplers


class AnnotatedDataTransformer:
  def __init__(self, transform_param=None, is_training=True, train_samplers=create_default_samplers()):
    assert not train_samplers or transform_param

    self.data_transformer = None
    self.batch_samplers = None

    if is_training:
      self.batch_samplers = train_samplers

    if transform_param:
      self.data_transformer = DataTransformer(is_training=is_training, transform_param=transform_param)

  def transform(self, image, annotation):
    if self.data_transformer:
      image = self.data_transformer.distort_image(image)
      image, annotation = self.data_transformer.expand_image(image, annotation)

      if self.batch_samplers:
        sampled_boxes = generate_batch_samples(annotation, self.batch_samplers)

        if sampled_boxes:
          # Randomly pick a sampled bbox and crop the expand_datum.
          index = random.randint(0, len(sampled_boxes) - 1)
          image, annotation = self.data_transformer.crop_image(image, annotation, sampled_boxes[index])

      image, annotation = self.data_transformer.transform(image, annotation)

    return image, annotation


# #############################################################################
# ############ Photometric image transforms ###################################
# #############################################################################

def _random_brightness(image, brightness_prob, brightness_delta):
  prob = random.random()
  if prob < brightness_prob:
    delta = random.uniform(-brightness_delta, brightness_delta)
    delta_mat = np.full_like(image, abs(delta))
    if delta > 0:
      res = cv2.add(image, delta_mat)
    else:
      res = cv2.subtract(image, delta_mat)
    return res
  return image


def _random_contrast(image, contrast_prob, lower, upper):
  prob = random.random()
  if prob < contrast_prob:
    delta = random.uniform(lower, upper)
    res = cv2.addWeighted(image, delta, 0, 0, 0)
    return res
  return image


def _random_saturation(image, saturation_prob, lower, upper):
  prob = random.random()
  if prob < saturation_prob:
    delta = random.uniform(lower, upper)
    if math.fabs(delta - 1.) > 1e-3:
      hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
      hsv[:, :, 1] = cv2.addWeighted(hsv[:, :, 1], delta, 0, 0, 0)
      return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
  return image


def _random_hue(image, hue_prob, hue_delta):
  prob = random.random()
  if prob < hue_prob:
    delta = random.uniform(-hue_delta, hue_delta)
    if math.fabs(delta) > 0:
      hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
      hsv[:, :, 0] = cv2.add(hsv[:, :, 0], delta)
      return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
  return image


def _random_order_channels(image, random_order_prob):
  prob = random.random()
  if prob < random_order_prob:
    perms = np.arange(len(image.shape))
    np.random.shuffle(perms)
    return image[..., perms]
  return image


def apply_distort(image, distort_param):
  prob = random.random()
  if prob > 0.5:
    image = _random_brightness(image, distort_param.brightness_prob, distort_param.brightness_delta)
    image = _random_contrast(image, distort_param.contrast_prob, distort_param.contrast_lower,
                             distort_param.contrast_upper)
    image = _random_saturation(image, distort_param.saturation_prob, distort_param.saturation_lower,
                               distort_param.saturation_upper)
    image = _random_hue(image, distort_param.hue_prob, distort_param.hue_delta)
    image = _random_order_channels(image, distort_param.random_order_prob)
  else:
    image = _random_brightness(image, distort_param.brightness_prob, distort_param.brightness_delta)
    image = _random_saturation(image, distort_param.saturation_prob, distort_param.saturation_lower,
                               distort_param.saturation_upper)
    image = _random_hue(image, distort_param.hue_prob, distort_param.hue_delta)
    image = _random_contrast(image, distort_param.contrast_prob, distort_param.contrast_lower,
                             distort_param.contrast_upper)
    image = _random_order_channels(image, distort_param.random_order_prob)
  return image


def apply_noise(image, noise_param):
  if noise_param.decolorize:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

  if noise_param.gauss_blur:
    image = cv2.GaussianBlur(image, (7, 7), 1.5)

  if noise_param.hist_eq:
    if image.channels() > 1:
      ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
      ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
      image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
      image = cv2.equalizeHist(image)

  if noise_param.clahe:
    clahe = cv2.createCLAHE(clipLimit=4)
    if image.channels > 1:
      ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
      ycrcb[..., 0] = clahe.apply(ycrcb[..., 0])
      image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
      image = clahe.apply(image)

  if noise_param.jpeg > 0:
    buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, noise_param.jpeg])
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

  if noise_param.erode:
    elem = cv2.getStructuringElement(2, (3, 3), (1, 1))
    image = cv2.erode(image, elem)

  if noise_param.posterize:
    def color_reduce(img, div=64):
      div_2 = div / 2
      table = (i / div * div + div_2 for i in range(256))
      table = np.array(table).astype(np.uint8)
      return cv2.LUT(img, table)

    image = color_reduce(image)

  if noise_param.inverse:
    image = cv2.bitwise_not(image)

  if noise_param.saltpepper_fraction > 0 and noise_param.saltpepper_values:
    height, width = image.shape[0:2]
    noise_pixels_num = int(noise_param.saltpepper_fraction * width * height)

    def constant_noise(noise_pixels_num, val, img):
      assert isinstance(val, list)
      iheight, iwidth, channels = img.shape

      for _ in range(noise_pixels_num):
        i = random.randint(iwidth - 1)
        j = random.randint(iheight - 1)

        if channels == 1:
          img[i, j] = val[0]
        elif channels == 3:
          img[i, j, :] = val

    constant_noise(noise_pixels_num, noise_param.saltpepper_values, image)

  if noise_param.convert_to_hsv:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  if noise_param.convert_to_lab:
    lab_image = image.astype(np.float32) * (1.0 / 255)
    image = cv2.cvtColor(lab_image, cv2.COLOR_BGR2Lab)

  return image


#  #############################################################################
#  ############ Geometric image transforms #####################################
#  #############################################################################

def expand_image(image, expand_ratio, mean_value=None):
  height, width, chs = image.shape

  # Get the bbox dimension.
  new_h = int(height * expand_ratio)
  new_w = int(width * expand_ratio)

  h_off = int(math.floor(random.uniform(0., new_h - height)))
  w_off = int(math.floor(random.uniform(0., new_w - width)))

  xmin = -w_off / float(width)
  ymin = -h_off / float(height)
  xmax = (new_w - w_off) / float(width)
  ymax = (new_h - h_off) / float(height)
  expand_box = BoundingBox(xmin, ymin, xmax, ymax)

  fill_value = mean_value if mean_value is not None else [0] * chs
  expanded_image = np.full(shape=(new_h, new_w, chs), fill_value=fill_value, dtype=image.dtype)
  expanded_image[h_off:h_off + height, w_off:w_off + width, :] = image
  return expanded_image, expand_box


def crop_image(img, bbox):
  img_height, img_width = img.shape[0:2]

  # Get the bbox dimension.
  clipped_bbox = bbox.clip_box()
  scaled_bbox = clipped_bbox.scale_box(img_height, img_width)

  # Crop the image using bbox.
  w_off = int(scaled_bbox.xmin)
  h_off = int(scaled_bbox.ymin)
  width = int(scaled_bbox.xmax - scaled_bbox.xmin)
  height = int(scaled_bbox.ymax - scaled_bbox.ymin)
  return img[h_off:h_off + height, w_off:w_off + width]


def aspect_keeping_resize_and_pad(image, new_width, new_height, pad_type, pad_val, interp_mode):
  orig_aspect = float(image.cols) / image.rows
  new_aspect = float(new_width) / new_height

  if orig_aspect > new_aspect:
    height = math.floor(float(new_width) / orig_aspect)
    resized = cv2.resize(image, (new_width, height), interpolation=interp_mode)
    h = resized.shape[0]
    padding = math.floor((new_height - h) / 2.0)
    resized = cv2.copyMakeBorder(resized, padding, new_height - h - padding, 0, 0, borderType=pad_type,
                                 value=pad_val)
  else:
    width = math.floor(orig_aspect * new_height)
    resized = cv2.resize(image, (width, new_height), 0, 0, interpolation=interp_mode)
    w = resized.shape[1]
    padding = math.floor((new_width - w) / 2.0)
    resized = cv2.copyMakeBorder(resized, 0, 0, padding, new_width - w - padding, borderType=pad_type,
                                 value=pad_val)
  return resized


def aspect_keeping_resize_by_small(in_img, new_width, new_height, interp_mode):
  orig_aspect = float(in_img.cols) / in_img.rows
  new_aspect = float(new_width) / new_height

  if orig_aspect < new_aspect:
    height = math.floor(float(new_width) / orig_aspect)
    resized = cv2.resize(in_img, (new_width, height), 0, 0, interpolation=interp_mode)
  else:
    width = math.floor(orig_aspect * new_height)
    resized = cv2.resize(in_img, (width, new_height), 0, 0, interpolation=interp_mode)

  return resized


def apply_resize(image, resize_param):
  assert isinstance(resize_param.interp_mode, (list, tuple))

  interp_mode = cv2.INTER_LINEAR
  num_interp_mode = len(resize_param.interp_mode)
  if num_interp_mode > 0:
    probs = [1. / num_interp_mode] * num_interp_mode
    cumulative = np.cumsum(probs)
    val = random.uniform(0, cumulative[-1])
    prob_num = bisect.bisect_left(cumulative, val)
    interp_mode = resize_param.interp_mode[prob_num]

  if resize_param.resize_mode == ResizeParameter.WARP:
    return cv2.resize(image, dsize=(resize_param.width, resize_param.height), interpolation=interp_mode)
  elif resize_param.resize_mode == ResizeParameter.FIT_LARGE_SIZE_AND_PAD:
    return aspect_keeping_resize_and_pad(image, resize_param.width, resize_param.height, resize_param.pad_mode,
                                         resize_param.pad_val, interp_mode)
  elif resize_param.resize_mode == ResizeParameter.FIT_SMALL_SIZE:
    return aspect_keeping_resize_by_small(image, resize_param.width, resize_param.height, interp_mode)
  else:
    raise Exception()


def update_bbox_by_resize_policy(old_width, old_height, bbox, resize_param):
  new_height = resize_param.height
  new_width = resize_param.width
  orig_aspect = float(old_width) / old_height
  new_aspect = new_width / new_height

  x_min = bbox.xmin * old_width
  y_min = bbox.ymin * old_height
  x_max = bbox.xmax * old_width
  y_max = bbox.ymax * old_height

  if resize_param.resize_mode == ResizeParameter.WARP:
    x_min = max(0., x_min * new_width / old_width)
    x_max = min(new_width, x_max * new_width / old_width)
    y_min = max(0., y_min * new_height / old_height)
    y_max = min(new_height, y_max * new_height / old_height)

  elif resize_param.resize_mode == ResizeParameter.FIT_LARGE_SIZE_AND_PAD:
    if orig_aspect > new_aspect:
      padding = (new_height - new_width / orig_aspect) / 2
      x_min = max(0., x_min * new_width / old_width)
      x_max = min(new_width, x_max * new_width / old_width)
      y_min = y_min * (new_height - 2 * padding) / old_height
      y_min = padding + max(0., y_min)
      y_max = y_max * (new_height - 2 * padding) / old_height
      y_max = padding + min(new_height, y_max)
    else:
      padding = (new_width - orig_aspect * new_height) / 2
      x_min = x_min * (new_width - 2 * padding) / old_width
      x_min = padding + max(0., x_min)
      x_max = x_max * (new_width - 2 * padding) / old_width
      x_max = padding + min(new_width, x_max)
      y_min = max(0., y_min * new_height / old_height)
      y_max = min(new_height, y_max * new_height / old_height)

  elif resize_param.resize_mode == ResizeParameter.FIT_SMALL_SIZE:
    if orig_aspect < new_aspect:
      new_height = new_width / orig_aspect
    else:
      new_width = orig_aspect * new_height

    x_min = max(0., x_min * new_width / old_width)
    x_max = min(new_width, x_max * new_width / old_width)
    y_min = max(0., y_min * new_height / old_height)
    y_max = min(new_height, y_max * new_height / old_height)

  result = BoundingBox(difficult=bbox.difficult)
  result.xmin = x_min / new_width
  result.ymin = y_min / new_height
  result.xmax = x_max / new_width
  result.ymax = y_max / new_height
  return result
