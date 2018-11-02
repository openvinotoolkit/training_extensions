"""
  Loss functions.
"""

import tensorflow as tf
from tensorflow.python.ops.control_flow_ops import with_dependencies


class MultiboxLoss:
  def __init__(self, loc_weight=1.0, neg_pos_ratio=3.0):
    """
    Multibox loss.

    Args:
      loc_weight: Weight of localization loss.
      neg_pos_ratio: Max ratio of negative to positive boxes in loss.
    """
    self.loc_weight = loc_weight
    self.neg_pos_ratio = neg_pos_ratio
    self.eval_tensors = {}

  @staticmethod
  def _localization_loss(ground_truth, prediction):
    """
    Compute L1-smooth loss.

    Args:
      ground_truth: Ground truth bounding boxes, shape: (?, #priors, 4).
      prediction: Predicted bounding boxes, shape: (?, #priors, 4).

    Returns:
      L1-smooth loss, shape: (?, #priors).
    """
    l1_loss = tf.losses.huber_loss(ground_truth, prediction, reduction=tf.losses.Reduction.NONE)
    return tf.reduce_sum(l1_loss, axis=-1)

  @staticmethod
  def _classification_loss(ground_truth, logits):
    """
      Compute sigmoid cross_entropy loss.

    Args:
      ground_truth: Ground truth targets, shape: (?, #priors, #classes).
      logits: Predicted logits, shape: (?, #priors, #classes).

    Returns:
      Sigmoid cross_entropy loss, shape: (?, #priors).
    """
    # softmax_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ground_truth, logits=logits)
    # return softmax_loss
    per_entry_cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=ground_truth, logits=logits)
    return tf.reduce_sum(per_entry_cross_ent, axis=-1)

  def __add_evaluation(self, evaluation_tensors):
    """
      Add tensors to a summary.
    Args:
      evaluation_tensors: Dictionary: evaluation name -> tensor scalar.
    """
    for log_name, log_value in evaluation_tensors.items():
      tf.summary.scalar(log_name, log_value)
      self.eval_tensors[log_name] = log_value

  def eval_summary(self, ground_truth, prediction):
    """
      Compute evaluation metrics (for EVAL mode).

    Args:
      ground_truth: Ground truth, shape: (?, #priors, 4 + #classes).
      prediction: Dictionary of predicted tensors, shape: {'locs'  : (?, #priors, 4), \
                                                           'confs' : (?, #priors, #classes), \
                                                           'logits': (?, #priors, #classes)}.
    Returns:
      Loss stub, shape: (1,).
    """
    localization_loss = self._localization_loss(ground_truth[:, :, :4],
                                                prediction['locs'])  # shape: (batch_size, num_priors)
    classification_loss = self._classification_loss(ground_truth[:, :, 4:],
                                                    prediction['logits'])  # shape: (batch_size, num_priors)
    positives = tf.reduce_max(ground_truth[:, :, 5:], axis=-1)  # shape: (batch_size, num_priors)
    num_positives = tf.reduce_sum(positives)  # shape: (1,)
    loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)  # shape: (batch_size,)
    classification_loss = tf.reduce_sum(classification_loss, axis=-1)  # shape: (batch_size,)

    evaluation_tensors = {
      'total_classification_loss':  tf.reduce_mean(classification_loss),
      'total_localization_loss': tf.reduce_mean(loc_loss),
    }

    self.__add_evaluation(evaluation_tensors)

    total_loss = tf.reduce_mean(classification_loss + self.loc_weight * loc_loss) / tf.maximum(1.0, num_positives)
    return total_loss

  def loss(self, ground_truth, prediction, bboxes):
    """
      Compute multibox loss.

    Args:
      ground_truth: Ground truth, shape: (?, #priors, 4 + #classes).
      prediction: Dictionary of predicted tensors, shape: {'locs'  : (?, #priors, 4), \
                                                           'confs' : (?, #priors, #classes), \
                                                           'logits': (?, #priors, #classes)}.
    Returns:
      Prediction loss, shape: (?,).
    """
    with tf.variable_scope('loss_function'):
      batch_size = tf.shape(prediction['locs'])[0]
      num_priors = tf.shape(prediction['locs'])[1]

      localization_loss = MultiboxLoss._localization_loss(ground_truth[:, :, :4],
                                                          prediction['locs'])  # shape: (batch_size, num_priors)
      classification_loss = MultiboxLoss._classification_loss(ground_truth[:, :, 4:],
                                                              prediction['logits'])  # shape: (batch_size, num_priors)

      ground_truth.set_shape([prediction['locs'].shape[0]] + ground_truth.shape[1:].as_list())

      negatives = ground_truth[:, :, 4]  # shape: (batch_size, num_priors)
      positives = tf.reduce_max(ground_truth[:, :, 5:], axis=-1)  # shape: (batch_size, num_priors)
      pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)  # shape: (batch_size,)

      num_positives = tf.reduce_sum(positives)  # shape: (1,)
      neg_class_loss_all = classification_loss * negatives  # shape: (batch_size, num_priors)
      n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.float32)  # shape: (1,)

      def no_negatives():
        return tf.zeros([batch_size], dtype=tf.float32), tf.constant(0, dtype=tf.int32)

      def hard_negative_mining():
        bboxes_per_batch = tf.unstack(bboxes)
        classification_loss_per_batch = tf.unstack(classification_loss)
        num_positives_per_batch = tf.unstack(tf.reduce_sum(positives, axis=-1))
        neg_class_loss_per_batch = tf.unstack(neg_class_loss_all)

        neg_class_losses = []
        total_negatives = []

        for bboxes_per_image, classification_loss_per_image, num_positives_per_image, neg_class_loss_per_image in \
            zip(bboxes_per_batch, classification_loss_per_batch, num_positives_per_batch, neg_class_loss_per_batch):
          min_negatives_keep = tf.maximum(self.neg_pos_ratio * num_positives_per_image, 3)
          num_negatives_keep = tf.minimum(min_negatives_keep,
                                          tf.count_nonzero(neg_class_loss_per_image, dtype=tf.float32))

          indices = tf.image.non_max_suppression(bboxes_per_image, classification_loss_per_image,
                                                 tf.to_int32(num_negatives_keep), iou_threshold=0.99)
          num_negatives = tf.size(indices)
          total_negatives.append(num_negatives)
          expanded_indexes = tf.expand_dims(indices, axis=1)  # shape: (num_negatives, 1)
          negatives_keep = tf.scatter_nd(expanded_indexes, updates=tf.ones_like(indices, dtype=tf.int32),
                                         shape=tf.shape(classification_loss_per_image))  # shape: (num_priors,)
          negatives_keep = tf.to_float(tf.reshape(negatives_keep, [num_priors]))  # shape: (batch_size, num_priors)
          neg_class_losses.append(tf.reduce_sum(classification_loss_per_image * negatives_keep, axis=-1))  # shape: (1,)

        return tf.stack(neg_class_losses), tf.reduce_sum(tf.stack(total_negatives))

      neg_class_loss, total_negatives = tf.cond(tf.equal(n_neg_losses, tf.constant(0.)),
                                                no_negatives, hard_negative_mining)  # shape: (batch_size,)
      class_loss = pos_class_loss + neg_class_loss  # shape: (batch_size,)
      loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)  # shape: (batch_size,)

      total_loss = tf.reduce_sum(class_loss + self.loc_weight * loc_loss) / tf.maximum(1.0, num_positives)

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      if update_ops:
        updates = tf.group(*update_ops)
        total_loss = with_dependencies([updates], total_loss)

      total_classification_loss = tf.reduce_mean(tf.reduce_sum(classification_loss, axis=-1))
      total_localization_loss = tf.reduce_mean(loc_loss, axis=-1)

      evaluation_tensors = {
        'total_classification_loss': total_classification_loss,
        'total_localization_loss': total_localization_loss,
        'num_positives_per_batch': num_positives,
        'num_negatives_per_batch': total_negatives
      }

      self.__add_evaluation(evaluation_tensors)

      return total_loss
