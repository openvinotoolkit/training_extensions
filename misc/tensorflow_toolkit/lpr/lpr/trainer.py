# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import re
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from lpr.networks.lprnet import LPRNet
from spatial_transformer import transformer


class InputData:
  # pylint: disable=too-many-arguments
  def __init__(self, batch_size, input_shape, file_list_path,
               apply_basic_aug=False, apply_stn_aug=True, apply_blur_aug=False):
    self.batch_size = batch_size
    self.input_shape = input_shape
    self.file_list_path = file_list_path
    self.apply_basic_aug = apply_basic_aug
    self.apply_stn_aug = apply_stn_aug
    self.apply_blur_aug = apply_blur_aug

  def input_fn(self):
    file_src = tf.train.string_input_producer([self.file_list_path])
    image, label = read_data(self.batch_size, self.input_shape, file_src)

    if self.apply_basic_aug:
      image = augment(image)

    if self.apply_stn_aug:
      image = augment_with_stn(image)

    # blur/sharpen augmentation
    if self.apply_blur_aug:
      data, = tf.py_func(random_blur, [image], [tf.float32])
      data.set_shape([self.batch_size] + list(self.input_shape))  # [batch_size, height, width, channels_num]
    else:
      data = image

    return data, label


def read_data(batch_size, input_shape, file_src):
  reader = tf.TextLineReader()
  _, value = reader.read(file_src)
  filename, label = tf.decode_csv(value, [[''], ['']], ' ')
  image_file = tf.read_file(filename)

  height, width, channels_num = input_shape
  rgb_image = tf.image.decode_png(image_file, channels=channels_num)
  rgb_image_float = tf.image.convert_image_dtype(rgb_image, tf.float32)
  resized_images = tf.image.resize_images(rgb_image_float, [height, width])
  resized_images.set_shape(input_shape)

  min_after_dequeue = 30000
  capacity = min_after_dequeue + 3 * batch_size
  image_batch, label_batch = tf.train.shuffle_batch([resized_images, label], batch_size=batch_size, capacity=capacity,
                                                    min_after_dequeue=min_after_dequeue)
  return image_batch, label_batch


# Function for basic image augmentation - photometric distortions
def augment(images):
  augmented = tf.image.random_brightness(images, max_delta=0.2)
  augmented = tf.image.random_contrast(augmented, lower=0.8, upper=1.2)
  augmented = tf.add(augmented, tf.truncated_normal(tf.shape(augmented), stddev=0.02))
  return augmented


# Function for STN image augmentation - geometric transformations with STN
def augment_with_stn(images):
  identity = identity_transform(images)
  noise = tf.truncated_normal(identity.get_shape(), stddev=0.1)
  # curriculum_rate = tf.clip_by_value(0.0001 * tf.cast(global_step, tf.float32), 0.0, 1.0)
  # noise = tf.scalar_mul(curriculum_rate, noise)
  return apply_stn(images, tf.add(identity, noise))


# Function for identity transformation
def identity_transform(images):
  shape = images.get_shape()
  ident = tf.constant(np.array([[[1., 0, 0], [0, 1., 0]]]).astype('float32'))
  return tf.tile(ident, [shape[0].value, 1, 1])


# Function wrapper for STN application
def apply_stn(images, transform_params):
  shape = images.get_shape()
  out_size = (shape[1], shape[2])
  warped = transformer(images, transform_params, out_size)
  warped.set_shape(shape)
  return warped


def random_blur(images):
  result = []
  for k in range(images.shape[0]):
    samples = np.random.normal(scale=1.)
    kernel = np.array([[0., samples, 0.], [samples, 1. - 4. * samples, samples], [0., samples, 0.]])
    result.append(cv2.filter2D(images[k], -1, kernel).astype(np.float32))
  return np.array(result)


# Function for construction whole network
def inference(rnn_cells_num, input, num_classes):
  cnn = LPRNet.lprnet(input)

  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      normalizer_fn=slim.batch_norm,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    classes = slim.conv2d(cnn, num_classes, [1, 13])
    pattern = slim.fully_connected(slim.flatten(classes), rnn_cells_num)  # patterns number
    width = int(cnn.get_shape()[2])
    pattern = tf.reshape(pattern, (-1, 1, 1, rnn_cells_num))
    pattern = tf.tile(pattern, [1, 1, width, 1])
    # pattern = slim.fully_connected(pattern, num_classes * width, normalizer_fn=None, activation_fn=tf.nn.sigmoid)
    # pattern = tf.reshape(pattern, (-1, 1, width, num_classes))

  inf = tf.concat(axis=3, values=[classes, pattern])  # skip connection over RNN
  inf = slim.conv2d(inf, num_classes, [1, 1], normalizer_fn=None,
                    activation_fn=None)  # fully convolutional linear activation

  inf = tf.squeeze(inf, [1])

  return inf


class CTCUtils:
  vocab = {}
  r_vocab = {}

  # Generate CTC from labels
  @staticmethod
  def compute_ctc_from_labels(labels):
    x_ix = []
    x_val = []
    batch_size = labels.shape[0]
    for batch_i in range(batch_size):
      label = labels[batch_i]
      for time, val in enumerate(encode(label.decode('utf-8'), CTCUtils.vocab)):
        x_ix.append([batch_i, time])
        x_val.append(val)
    x_shape = [batch_size, np.asarray(x_ix).max(0)[1] + 1]

    return [np.array(x_ix), np.array(x_val), np.array(x_shape)]

  # Function for computing simple accuracy metric
  @staticmethod
  def accuracy(gt_labels, values):
    prediction = decode(values, CTCUtils.r_vocab)

    encoded_gt_labels = list()
    for gt_label in gt_labels:
      encoded_label = encode(gt_label.decode('utf-8'), CTCUtils.vocab)
      encoded_gt_labels.append(encoded_label)

    gt_labels = decode(np.array(encoded_gt_labels), CTCUtils.r_vocab)

    batch_size = len(gt_labels)
    mean_accuracy = 0.0

    for k in range(batch_size):
      if gt_labels[k] == prediction[k]:
        mean_accuracy += 1.0 / batch_size
    return mean_accuracy


# Function for encoding actual LPR label in vector of numbers
def encode(label_string, vocab):
  return [vocab[c] for c in re.findall('(<[^>]*>|.)', label_string)]


# Function for decoding vector of numbers into LPR label
def decode(values, reverse_vocab):
  result = []
  for j in range(values.shape[0]):
    string_label = ''
    for value in values[j]:
      string_label += reverse_vocab[value]
    result.append(string_label)
  return result


def decode_beams(vals, r_vocab):
  beams_list = []
  for val in vals:
    decoded_number = ''
    for code in val:
      decoded_number += r_vocab[code]
    beams_list.append(decoded_number)
  return beams_list


def decode_ie_output(vals, r_vocab):
  vals = vals.flatten()
  decoded_number = ''
  for val in vals:
    if val < 0:
      break
    decoded_number += r_vocab[val]
  return decoded_number


class LPRVocab:
  @staticmethod
  def create_vocab(train_list_path, val_list_path, use_h_concat=False, use_oi_concat=False):
    [vocab, r_vocab, num_classes] = LPRVocab._create_standard_vocabs(train_list_path, val_list_path)
    if use_h_concat:
      [vocab, r_vocab, num_classes] = LPRVocab._concat_all_hieroglyphs(vocab, r_vocab)
    if use_oi_concat:
      [vocab, r_vocab, num_classes] = LPRVocab._concat_oi(vocab, r_vocab)

    return vocab, r_vocab, num_classes

  @staticmethod
  def _char_range(char1, char2):
    """Generates the characters from `char1` to `char2`, inclusive."""
    for char_code in range(ord(char1), ord(char2) + 1):
      yield chr(char_code)

  # Function for reading special symbols
  @staticmethod
  def _read_specials(filepath):
    characters = set()
    with open(filepath, 'r') as file_:
      for line in file_:
        current_label = line.split(' ')[-1].strip()
        characters = characters.union(re.findall('(<[^>]*>|.)', current_label))
    return characters

  @staticmethod
  def _create_standard_vocabs(train_list_path, val_list_path):
    chars = set().union(LPRVocab._char_range('A', 'Z')).union(LPRVocab._char_range('0', '9'))
    chars = chars.union(LPRVocab._read_specials(train_list_path)).union(LPRVocab._read_specials(val_list_path))
    chars = list(chars)
    chars.sort()
    chars.append('_')
    num_classes = len(chars)
    vocab = dict(zip(chars, range(num_classes)))
    r_vocab = dict(zip(range(num_classes), chars))
    r_vocab[-1] = ''
    return [vocab, r_vocab, num_classes]

  # Function for treating all hieroglyphs as 1 class
  @staticmethod
  def _concat_all_hieroglyphs(vocab_before, r_vocab_before):
    chars = vocab_before.keys()
    tf.logging.info("Old number of classes: {}".format(len(chars)))

    # Get all hieroglyphs from train and test
    hieroglyphs = list((set(chars) - set(LPRVocab._char_range('A', 'Z'))) - set(LPRVocab._char_range('0', '9')))

    tf.logging.info('Total hieroglyphs num: {}'.format(len(hieroglyphs)))
    tf.logging.info('Vocabulary before: ')
    tf.logging.info(vocab_before)

    chars = list(set().union(LPRVocab._char_range('A', 'Z')).union(LPRVocab._char_range('0', '9')))
    chars.sort()
    new_num_classes = len(chars)
    vocab_after = dict(zip(chars, range(new_num_classes)))
    vocab_after.update(dict(zip(hieroglyphs, [new_num_classes] * len(hieroglyphs))))
    vocab_after['_'] = new_num_classes + 1
    new_num_classes += 2

    tf.logging.info('Vocabulary after: ')
    tf.logging.info(vocab_after)

    tf.logging.info('Reverse vocabulary before: ')
    tf.logging.info(r_vocab_before)

    r_vocab_after = dict((v, k) for k, v in vocab_after.iteritems())
    r_vocab_after[-1] = ''

    tf.logging.info('Reverse vocabulary after: ')
    tf.logging.info(r_vocab_after)

    tf.logging.info('New number of classes: {}'.format(new_num_classes))
    tf.logging.info('Vocabulary len: {}'.format(len(vocab_after)))
    tf.logging.info('Reverse vocabulary length: {}'.format(len(r_vocab_after)))

    return [vocab_after, r_vocab_after, new_num_classes]

  # Function for treating O/0, I/1 as 1 class
  @staticmethod
  def _concat_oi(vocab_before, r_vocab_before):
    chars = vocab_before.keys()
    tf.logging.info('Old number of classes: {}'.format(len(chars)))

    tf.logging.info('Vocabulary before: ')
    tf.logging.info(vocab_before)

    # Remove '0' and '1'
    chars = list(set(chars) - set(['0', '1']))
    chars.sort()
    new_num_classes = len(chars)
    vocab_after = dict(zip(chars, range(new_num_classes)))
    vocab_after['0'] = vocab_after['O']
    vocab_after['1'] = vocab_after['I']
    vocab_after['_'] = new_num_classes + 1
    new_num_classes += 1

    tf.logging.info('Vocabulary after: ')
    tf.logging.info(vocab_after)

    tf.logging.info('Reverse vocabulary before: ')
    tf.logging.info(r_vocab_before)

    r_vocab_after = dict((v, k) for k, v in vocab_after.iteritems())
    r_vocab_after[-1] = ''

    tf.logging.info('Reverse vocabulary after: ')
    tf.logging.info(r_vocab_after)

    tf.logging.info('New number of classes: {}'.format(new_num_classes))
    tf.logging.info('Vocabulary len: {}'.format(len(vocab_after)))
    tf.logging.info('Reverse vocabulary length: {}'.format(len(r_vocab_after)))

    return [vocab_after, r_vocab_after, new_num_classes]
