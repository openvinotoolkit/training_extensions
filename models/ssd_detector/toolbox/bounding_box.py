import math, random


class BoundingBox:
  def __init__(self, xmin=0., ymin=0., xmax=0., ymax=0., difficult=False):  # , score=0, label=0):
    self.xmin = xmin
    self.ymin = ymin
    self.xmax = xmax
    self.ymax = ymax

    self.difficult = difficult
    # self.score = score
    # self.label = label

  def width(self):
    return self.xmax - self.xmin

  def height(self):
    return self.ymax - self.ymin

  def to_list(self):
    return [self.xmin, self.ymin, self.xmax, self.ymax]

  def __str__(self):
    return '[xmin={} ymin={} xmax={} ymax={} diff={}]'.format(self.xmin, self.ymin, self.xmax, self.ymax,
                                                              self.difficult)

  def size(self):
    if self.xmax < self.xmin or self.ymax < self.ymin:
      return 0

    width = self.xmax - self.xmin
    height = self.ymax - self.ymin
    return width * height

  def clip_box(self):
    xmin = max(min(self.xmin, 1.), 0.)
    xmax = max(min(self.xmax, 1.), 0.)
    ymin = max(min(self.ymin, 1.), 0.)
    ymax = max(min(self.ymax, 1.), 0.)
    return BoundingBox(xmin, ymin, xmax, ymax, self.difficult)

  def scale_box(self, height, width):
    xmin = self.xmin * width
    ymin = self.ymin * height
    xmax = self.xmax * width
    ymax = self.ymax * height
    return BoundingBox(xmin, ymin, xmax, ymax, self.difficult)

  def project_box(self, box):
    if box.xmin >= self.xmax or box.xmax <= self.xmin or box.ymin >= self.ymax or box.ymax <= self.ymin:
      return None

    src_width = self.xmax - self.xmin
    src_height = self.ymax - self.ymin
    xmin = (box.xmin - self.xmin) / src_width
    ymin = (box.ymin - self.ymin) / src_height
    xmax = (box.xmax - self.xmin) / src_width
    ymax = (box.ymax - self.ymin) / src_height
    proj_bbox = BoundingBox(xmin, ymin, xmax, ymax, box.difficult).clip_box()
    return proj_bbox if proj_bbox.size() > 0 else None

  def locate_box(self, box):
    src_width = self.xmax - self.xmin
    src_height = self.ymax - self.ymin
    xmin = self.xmin + box.xmin * src_width
    ymin = self.ymin + box.ymin * src_height
    xmax = self.xmin + box.xmax * src_width
    ymax = self.ymin + box.ymax * src_height
    return BoundingBox(xmin, ymin, xmax, ymax, box.difficult)

  def is_cross_boundary(self):
    return self.xmin < 0 or self.xmin > 1 or self.ymin < 0 or self.ymin > 1 or \
           self.xmax < 0 or self.xmax > 1 or self.ymax < 0 or self.ymax > 1


def intersect_box(box1, box2):
  if box2.xmin > box1.xmax or box2.xmax < box1.xmin or box2.ymin > box1.ymax or box2.ymax < box1.ymin:
    return BoundingBox()

  xmin = max(box1.xmin, box2.xmin)
  ymin = max(box1.ymin, box2.ymin)
  xmax = min(box1.xmax, box2.xmax)
  ymax = min(box1.ymax, box2.ymax)
  return BoundingBox(xmin, ymin, xmax, ymax)


def box_coverage(box1, box2):
  inter_box = intersect_box(box1, box2)
  inter_size = inter_box.size()
  return inter_size / box1.size() if inter_size > 0 else 0


def jaccard_overlap(box1, box2):
  inter = intersect_box(box1, box2).size()
  return inter / (box1.size() + box2.size() - inter) if inter > 0 else 0


def sample_box(sampler):
  scale = random.uniform(sampler.min_scale, sampler.max_scale)
  aspect_ratio = random.uniform(sampler.min_aspect_ratio, sampler.max_aspect_ratio)

  aspect_ratio = max(aspect_ratio, scale ** 2.)
  aspect_ratio = min(aspect_ratio, 1 / (scale ** 2.))

  # Figure out bbox dimension.
  bbox_width = scale * math.sqrt(aspect_ratio)
  bbox_height = scale / math.sqrt(aspect_ratio)

  # Figure out top left coordinates.
  w_off = random.uniform(0., 1. - bbox_width)
  h_off = random.uniform(0., 1. - bbox_height)

  return BoundingBox(w_off, h_off, w_off + bbox_width, h_off + bbox_height)


def extrapolate_box(param, height, width, crop_bbox, bbox):
  height_scale = param.height_scale
  width_scale = param.width_scale

  if height_scale > 0 and width_scale > 0 and param.resize_mode == 'FIT_SMALL_SIZE':
    orig_aspect = float(width) / height
    resize_height = param.height
    resize_width = param.width
    resize_aspect = resize_width / resize_height
    if orig_aspect < resize_aspect:
      resize_height = resize_width / orig_aspect
    else:
      resize_width = resize_height * orig_aspect

    crop_height = resize_height * (crop_bbox.ymax - crop_bbox.ymin)
    crop_width = resize_width * (crop_bbox.xmax - crop_bbox.xmin)

    result = BoundingBox(difficult=bbox.difficult)
    result.xmin = bbox.xmin * crop_width / width_scale
    result.xmax = bbox.xmax * crop_width / width_scale
    result.ymin = bbox.ymin * crop_height / height_scale
    result.ymax = bbox.ymax * crop_height / height_scale
    return result

  return bbox


def satisfy_sample_constraint(sampled_box, object_boxes, constr):
  has_jaccard_overlap = constr.min_jaccard_overlap or constr.max_jaccard_overlap
  has_sample_coverage = constr.min_sample_coverage or constr.max_sample_coverage
  has_object_coverage = constr.min_object_coverage or constr.max_object_coverage
  satisfy = not has_jaccard_overlap and not has_sample_coverage and not has_object_coverage
  if satisfy:
    return True

  for box in object_boxes:
    jaccard = jaccard_overlap(sampled_box, box)
    sample_cov = box_coverage(sampled_box, box)
    object_cov = box_coverage(box, sampled_box)

    if constr.min_jaccard_overlap and jaccard < constr.min_jaccard_overlap:
      continue

    if constr.max_jaccard_overlap and jaccard > constr.max_jaccard_overlap:
      continue

    if constr.min_sample_coverage and sample_cov < constr.min_sample_coverage:
      continue
    if constr.max_sample_coverage and sample_cov > constr.max_sample_coverage:
      continue

    if constr.min_object_coverage and object_cov < constr.min_object_coverage:
      continue

    if constr.max_object_coverage and object_cov > constr.max_object_coverage:
      continue

    return True

  return False


def generate_batch_samples(annotation, batch_samplers):
  object_boxes = [b for _, boxes in annotation.items() for b in boxes]
  sampled_boxes = []

  for sampler in batch_samplers:
    if not sampler.use_original_image:
      continue

    current_boxes = []
    for _ in range(sampler.max_trials):
      if len(current_boxes) >= sampler.max_sample:
        break

      # Generate sampled_bbox in the normalized space [0, 1].
      sampled_bbox = sample_box(sampler)

      # Determine if the sampled bbox is positive or negative by the constraint.
      if satisfy_sample_constraint(sampled_bbox, object_boxes, sampler):
        current_boxes.append(sampled_bbox)

    sampled_boxes.extend(current_boxes)
  return sampled_boxes
