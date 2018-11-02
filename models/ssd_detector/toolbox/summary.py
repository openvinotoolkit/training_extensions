import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pickle
import tensorflow as tf


def create_tensors_and_streaming_ops_for_assigned_priors(assigned_priors, priors_info, num_classes, weights=1.):
  total_priors_number = np.sum(
    [np.prod(priors_count) * num_priors_per_pixel for priors_count, num_priors_per_pixel in priors_info])
  metric_ops = {}

  with tf.name_scope('priors_streaming_summary'):
    assigned_priors_total = tf.reduce_mean(tf.reduce_max(assigned_priors[:, :, 5:], axis=-1), axis=0)
    assigned_priors_total = tf.reshape(assigned_priors_total, [total_priors_number])  # For streaming operations
    stream_assigned_priors_total = tf.metrics.mean_tensor(assigned_priors_total, weights=weights)
    metric_ops['summary'] = stream_assigned_priors_total

    for i in range(num_classes):
      if i == 0:  # Skip background class
        continue
      class_id = 4 + i  # 4 - localization
      class_name = 'class_{0}'.format(i)

      assigned_priors_type = tf.reduce_mean(assigned_priors[:, :, class_id], axis=0)  # Mean by batch
      assigned_priors_type = tf.reshape(assigned_priors_type, [total_priors_number])  # For streaming operations

      stream_assigned_priors_type = tf.metrics.mean_tensor(assigned_priors_type, weights=weights)
      metric_ops[class_name] = stream_assigned_priors_type

  return metric_ops


def get_detailed_assigned_priors_summary(assigned_priors, priors_info, name):
  """
  Get assigned priors 1D tensors by SSD heads and priors type.

  Args:
    assigned_priors: Assigned priors, tensor of shape (num_priors).
    priors_info: Information about priors, list of pairs for every ssd head: tensor_dimensions, num_priors_per_pixel.
    name: Output name.

  Returns:
    detailed_assigned_priors: Dictionary with tensors for every SSD head and prior type.
  """
  assert len(assigned_priors.shape) == 1

  detailed_assigned_priors = dict()
  detailed_assigned_priors['priors/{0}'.format(name)] = assigned_priors

  start = 0
  total_priors_number = int(assigned_priors.shape[0])

  for head_id, (tensor_dimensions, num_priors_per_pixel) in enumerate(priors_info):
    priors_per_type = np.prod(tensor_dimensions)
    priors_count = np.prod(tensor_dimensions) * num_priors_per_pixel

    prior_map = np.zeros(shape=total_priors_number, dtype=np.bool)
    for i in range(priors_count):
      prior_map[start + i] = True

    if isinstance(assigned_priors, tf.Tensor):
      assigned_priors_head = tf.boolean_mask(assigned_priors, prior_map)
      assigned_priors_head = tf.reshape(assigned_priors_head, [priors_count])
    else:
      assigned_priors_head = assigned_priors[prior_map]

    detailed_assigned_priors['priors_by_head/{0}/head_{1}'.format(name, head_id)] = assigned_priors_head

    for offset in range(num_priors_per_pixel):
      prior_map = np.zeros(shape=total_priors_number, dtype=np.bool)
      for i in range(priors_per_type):
        prior_map[start + offset + i * num_priors_per_pixel] = True

      if isinstance(assigned_priors, tf.Tensor):
        assigned_priors_head_type = tf.boolean_mask(assigned_priors, prior_map)
        assigned_priors_head_type = tf.reshape(assigned_priors_head_type, [priors_per_type])
      else:
        assigned_priors_head_type = assigned_priors[prior_map]

      assigned_priors_head_type_name = 'priors_by_head_and_type/{0}/head_{1}/prior_{2}'.format(name, head_id,
                                                                                               offset)
      detailed_assigned_priors[assigned_priors_head_type_name] = assigned_priors_head_type

    start += priors_count

  return detailed_assigned_priors


def get_detailed_assigned_priors_summary_tf(assigned_priors_dict, priors_info):
  detailed_assigned_priors = dict()

  with tf.name_scope('detailed_priors_summary'):
    for name, (assigned_priors, _) in assigned_priors_dict.items():
      with tf.name_scope(name):
        detailed_assigned_priors.update(
          get_detailed_assigned_priors_summary(assigned_priors, priors_info, name))

  return detailed_assigned_priors


def group_ssd_heads(assigned_priors, prefix='prior_histogram/'):
  groups = {}
  priors_by_head_and_type = [(key.replace(prefix + 'priors_by_head_and_type/', '', 1), val)
                             for key, val in sorted(assigned_priors.items())
                             if key.startswith(prefix + 'priors_by_head_and_type/')]

  if not priors_by_head_and_type:
    return {}

  def __extract_key(key):
    keys = key.split('/')
    class_name = keys[0]
    head_id = int(keys[1].split('_')[1])
    prior_id = int(keys[2].split('_')[1])
    return (class_name, head_id, prior_id)

  priors_by_head_and_type = [(__extract_key(key), val) for key, val in priors_by_head_and_type]

  max_head_id = np.max([head_id for (_, head_id, prior_id), val in priors_by_head_and_type]) + 1
  max_prior_id = np.max([prior_id for (_, head_id, prior_id), val in priors_by_head_and_type]) + 1

  for (class_name, _, _), _ in priors_by_head_and_type:
    mat = [[None for x in range(max_prior_id)] for y in range(max_head_id)]
    groups.setdefault(class_name, mat)

  for (class_name, head_id, prior_id), val in priors_by_head_and_type:
    groups[class_name][head_id][prior_id] = val

  return groups


def fig_to_data(fig):
  fig.canvas.draw()
  width, height = fig.canvas.get_width_height()
  buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  buf.shape = (height, width, 3)
  return buf


def write_histogram_2d_tf(assigned_priors, priors_info, name, step, log_dir):
  try:
    name = name.decode()  # Convert byte-string to str
    log_dir = log_dir.decode()  # Convert byte-string to str
  except AttributeError:
    pass

  priors_info = pickle.loads(priors_info)

  detailed_assigned_priors = get_detailed_assigned_priors_summary(assigned_priors, priors_info, name)
  group = group_ssd_heads(detailed_assigned_priors, '')
  write_histogram_2d(group, step, log_dir, use_lognorm=False)
  write_histogram_2d(group, step, log_dir, use_lognorm=True)

  return True


def write_histogram_2d(assigned_priors_group, step, log_dir, use_lognorm=False):
  summaries = []
  for type_str, group in assigned_priors_group.items():
    max_head_id = len(group)
    max_prior_id = len(group[0])
    fig, axes = plt.subplots(max_head_id, max_prior_id, figsize=(max_prior_id * 2.5, max_head_id * 2))
    fig.tight_layout()

    if use_lognorm:
      name = 'priors_hist2d_log/' + type_str
    else:
      name = 'priors_hist2d/' + type_str

    all_data = [prior for head in group for prior in head if prior is not None]

    max_val = np.max([np.max(data) for data in all_data])
    min_data = [np.min(data[data > 0]) for data in all_data if np.sum(data > 0) != 0]
    min_val = np.min(min_data) if min_data else 0

    if use_lognorm:
      if min_val == 0:
        continue

    for i, head in enumerate(group):
      for j, prior in enumerate(head):
        if prior is None:
          axes[i, j].axis('off')
        else:
          prior = prior.copy()
          prior[prior <= 0] = min_val * 0.01
          prior_shape = int(np.sqrt(prior.shape))
          prior = np.reshape(prior, (prior_shape, prior_shape))
          dims = prior_shape, prior_shape
          x_values, y_values = np.meshgrid(range(prior.shape[0]), range(prior.shape[1]))
          ret = axes[i, j].hist2d(x_values.reshape((-1)), y_values.reshape((-1)), weights=prior.reshape((-1)),
                                  bins=dims,
                                  cmap=plt.cm.brg,
                                  norm=LogNorm() if use_lognorm else Normalize(),
                                  vmin=min_val * 0.95, vmax=max_val)
          colorbar_map = ret[3]
          axes[i, j].axis('on')
          axes[i, j].plot()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.05, 0.02, 0.9])
    fig.colorbar(colorbar_map, cax=cbar_ax)

    img = fig_to_data(fig)
    plt.close(fig)

    encoded_image = cv2.imencode('.jpg', img)[1].tostring()
    img_sum = tf.Summary.Image(encoded_image_string=encoded_image, height=img.shape[0], width=img.shape[1])
    summaries.append(tf.Summary.Value(tag=name, image=img_sum))

  if summaries:
    with tf.summary.FileWriterCache._lock:
      if log_dir not in tf.summary.FileWriterCache._cache:
        tf.summary.FileWriterCache._cache[log_dir] = tf.summary.FileWriter(log_dir,
                                                                           graph=None)  # Don't store the graph
      writer = tf.summary.FileWriterCache._cache[log_dir]

    writer.add_summary(tf.Summary(value=summaries), step)
