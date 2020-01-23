# Copyright 2019 Intel Corporation
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""This script helps generate similar training and validation splits
as the ones used in TensorFlow Object Detection API, which is much
different than organic coco train2017 and val2017.
By selecting ~8k specific images(called minival) for validation,
total training image set would be [coco_train2017+coco_val2017-minival]
(called train_plus in this script).

This script is designed for benchmarking a fine-tuned model with
retraining or transfer learning based on a TensorFlow Object
Detection API pretrained model to have a fair and reliable validation
result, since the pretrained models are claimed having been trained with
the same splits.

Please follow below dataset structure:
\_dataset
  \_images
  | \_train2017
  | | |_train_image0
  | | |_train_image1
  | | ...
  | | |_train_image118286
  | \_val2017
  | | |_val_image0
  | | |_val_image1
  | | ...
  | | |_val_image4999
  \_annotations
  | |_instances_train2017.json
  | |_instances_val2017.json
  |_mscoco_minival_ids.txt

NOTE: `train2017` and `val2017` folders are supposed to exist at passed-in
`--images` directory. `instances_train2017.json` and `instances_val2017.json`
are supposed to exist at passed-in `--annotations` directory.
"""

import json
import hashlib
import os
import contextlib2

import tensorflow as tf
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

# pylint: disable=too-many-locals,too-many-arguments

flags = tf.app.flags
tf.flags.DEFINE_string(
        'image_folder',
        None,
        'Folder of images, must include `train2017` and `val2017` folders.')
tf.flags.DEFINE_string(
        'annotation_folder',
        None,
        'Folder of annotations, must include `instances_train2017.json` '
        'and `instances_val2017.json` files.')
tf.flags.DEFINE_string(
        'coco_minival_ids_file',
        None,
        'File of selected id in coco2017 for validation.')
tf.flags.DEFINE_string(
        'output_folder',
        None,
        'Folder of output tfrecord')
FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


TRAIN_FOLDER_NAME = 'train2017'
TRAIN_ANNOTATION_NAME = 'instances_train2017.json'
TRAIN_TFRECORD_NAME = 'coco_train2017_plus.record'
VAL_FOLDER_NAME = 'val2017'
VAL_ANNOTATION_NAME = 'instances_val2017.json'
VAL_TFRECORD_NAME = 'coco_minival2017.record'


def create_tfrecord_from_coco(train_image_folder,
                              val_image_folder,
                              train_annotation_file,
                              val_annotation_file,
                              train_tfrecord,
                              val_tfrecord,
                              coco_minival_ids_file):
    """Create tfrecord from organic coco dataset and annotations.

    :param train_image_folder: folder of organic training images;
    :param val_image_folder: folder of organic validation images;
    :param train_annotation_file: file path to organic training annotations;
    :param val_annotation_file: file path to organic validation annotations;
    :param train_tfrecord: path and prefix of output training tfrecord;
    :param val_tfrecord: path and prefix of output validation tfrecord;
    :param coco_minival_ids_file: a file containing a list of minival coco id
                                  for validation, this file could usually be found
                                  as `mscoco_minival_ids.txt` at tensorflow/models
                                  repository;
    """
    minival_list = parse_minival_ids(coco_minival_ids_file)
    images, annotations, categories = merge_annotations(
        train_annotation_file, val_annotation_file)

    tf.logging.info('number of total images is {}'.format(len(images)))
    tf.logging.info('number of total annotations is {}'.format(len(annotations)))

    category_index = label_map_util.create_category_index(categories)
    annotation_index = generate_annotation_index(annotations)
    train_plus_images, minival_images = split_minival(images, minival_list)

    tf.logging.info('category index is {}'.format(category_index))
    tf.logging.info('number of train plus images is {}'
                    .format(len(train_plus_images)))
    tf.logging.info('number of minival images is {}'
                    .format(len(minival_images)))

    image_folders = [train_image_folder, val_image_folder]
    write_tfrecord(image_list=train_plus_images,
                   anno_index=annotation_index,
                   category_index=category_index,
                   image_folders=image_folders,
                   output_tfrecord=train_tfrecord)
    write_tfrecord(image_list=minival_images,
                   anno_index=annotation_index,
                   category_index=category_index,
                   image_folders=image_folders,
                   output_tfrecord=val_tfrecord,
                   tfrecord_shards=10)


def create_tf_example(image,
                      anno_list,
                      image_folders,
                      category_index):
    """Create a tf example for a single image.

    :param image: coco-like image object;
    :param anno_list: a list of coco-like annotation objects;
    :param image_folders: a list of folders which may contain required image;
    :param category_index: category index dictionary generated by `label_map_util`;
    """
    image_h = image['height']
    image_w = image['width']
    filename = image['file_name']
    image_id = image['id']
    if image_w <= 0 and image_h <= 0:
        raise ValueError('image width or height is smaller than or equal to zero!')

    # Traverse all image folders since we're not sure where exactly the file is.
    image_path = None
    for image_folder in image_folders:
        tmp_path = os.path.join(image_folder, filename)
        if os.path.exists(tmp_path):
            image_path = tmp_path
            break
    if image_path is None:
        raise OSError('image_path {} does not exist!'.format(image_path))

    with tf.gfile.GFile(image_path, 'rb') as f:
        encoded_jpg = f.read()
    if not encoded_jpg:
        raise ValueError('cannot read image from image_path!')

    key = hashlib.sha256(encoded_jpg).hexdigest()

    x1s = []
    y1s = []
    x2s = []
    y2s = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []

    for anno in anno_list:
        x1, y1, w, h = tuple(anno['bbox'])
        if w <= 0 and h <= 0:
            raise ValueError('annotation w or h value is wrong')

        x2, y2 = x1+w, y1+h
        if x1 < 0 or y1 < 0:
            raise ValueError('annotation coordinates is out of image')
        if x2 > image_w or y2 > image_h:
            raise ValueError('annotation coordinates is out of image')

        x1s.append(float(x1)/image_w)
        y1s.append(float(y1)/image_h)
        x2s.append(float(x2)/image_w)
        y2s.append(float(y2)/image_h)
        is_crowd.append(anno['iscrowd'])
        area.append(anno['area'])

        category_id = int(anno['category_id'])
        category_name = category_index[category_id]['name'].encode('utf8')
        category_ids.append(category_id)
        category_names.append(category_name)

    feature_dict = {
        'image/height': dataset_util.int64_feature(image_h),
        'image/width': dataset_util.int64_feature(image_w),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(x1s),
        'image/object/bbox/xmax': dataset_util.float_list_feature(x2s),
        'image/object/bbox/ymin': dataset_util.float_list_feature(y1s),
        'image/object/bbox/ymax': dataset_util.float_list_feature(y2s),
        'image/object/class/text': dataset_util.bytes_list_feature(category_names),
        'image/object/is_crowd': dataset_util.int64_list_feature(is_crowd),
        'image/object/area': dataset_util.float_list_feature(area),
    }
    # Masks are ignored here.
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return key, example


def generate_annotation_index(annotations):
    """Turn annotations into annotation index dictionary with
    image id as key and list of coco-like annotation objects as value.

    :param annotations: coco-like annotations list;
    :return: annotation index dictionary with image id as key and
             list of coco-like annotation objects as value;
    """
    annotation_index = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id not in annotation_index:
            annotation_index[image_id] = []
        annotation_index[image_id].append(annotation)
    return annotation_index


def load_annotation(annotation_file):
    """Load a coco annotation file.

    :param annotation_file: a coco annotation file;
    :return: images object in coco annotation file;
             annotations object in coco annotation file;
             categories object in coco annotation file;
    """
    assert os.path.exists(annotation_file), \
        'annotation file {} does not exist!'.format(annotation_file)
    with tf.gfile.GFile(annotation_file, 'r') as f:
        info = json.load(f)
    assert info, \
        'annotation is None, please check the annotation file'

    images = info['images']
    annotations = info['annotations']
    categories = info['categories']

    return images, annotations, categories


def merge_annotations(train_annotation_file, val_annotation_file):
    """Load organic annotation files and merge training and validation annotations.

    :param train_annotation_file: organic coco training annotation file;
    :param val_annotation_file: organic coco validation annotation file;
    :return: merged images object with both training and validation images;
             merged annotations with both training and validation annotations;
             categories object in coco annotation file;
    """
    train_images, train_annotations, categories = \
        load_annotation(train_annotation_file)
    val_images, val_annotations, _ = \
        load_annotation(val_annotation_file)

    tf.logging.info('number of organic train2017 images is {}'
                    .format(len(train_images)))
    tf.logging.info('number of organic val2017 images is {}'
                    .format(len(val_images)))

    train_images.extend(val_images)
    train_annotations.extend(val_annotations)

    return train_images, train_annotations, categories


def parse_minival_ids(coco_minival_ids_file):
    """Load minival ids file to a list.

    :param coco_minival_ids_file: path of coco minival ids file;
    :return: list of selected minival id for validation;
    """
    assert os.path.exists(coco_minival_ids_file), \
        'minival_ids_file {} does not exist!'.format(coco_minival_ids_file)
    with open(coco_minival_ids_file, 'r') as f:
        minival_list = [int(line) for line in f.readlines()]
    assert minival_list,\
        'minival ids is None, please check the coco_minival_ids_file'

    return minival_list


def split_minival(images, minival_list):
    """Split total images to training and validation splits according
    to selected minival id.

    :param images: list of total coco-like images(including organic
                   training and validation splits);
    :param minival_list: list of selected minival id for validation;
    :return: brand new train_plus split and minival split according;
    """
    train_plus_images = []
    minival_images = []
    for image in images:
        image_id = image['id']
        if image_id not in minival_list:
            train_plus_images.append(image)
        else:
            minival_images.append(image)
    return train_plus_images, minival_images


def write_tfrecord(image_list,
                   anno_index,
                   category_index,
                   image_folders,
                   output_tfrecord,
                   tfrecord_shards=100,
                   logging_interval=1000):
    """Translate coco-like annotation format into tfrecord and write down.

    :param image_list: a list of coco-like image objects;
    :param anno_index: annotation index dictionary with image id as key and list of
                       coco-like annotation objects as value;
    :param category_index: category index dictionary generated by `label_map_util`;
    :param image_folders: a list of folders which may contain required image;
    :param output_tfrecord: path and prefix of output tfrecord;
    :param tfrecord_shards: number of generated tfrecord shards;
    :param logging_interval: log will be printed out per logging interval;
    """
    length = len(image_list)

    with contextlib2.ExitStack() as tf_record_close_stack:
        tfrecord_writer = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_tfrecord, tfrecord_shards)
        skipped = 0
        for i, image in enumerate(image_list):
            try:
                if i % logging_interval == 0:
                    tf.logging.info('Processing {}/{} image'.format(i, length))
                image_id = image['id']

                if image_id not in anno_index:
                    skipped += 1
                    continue
                anno_list = anno_index[image_id]
                _, tf_example = create_tf_example(
                    image, anno_list, image_folders, category_index)
                shard_index = i % tfrecord_shards
                tfrecord_writer[shard_index].write(tf_example.SerializeToString())
            except (ValueError, OSError):
                skipped += 1

    tf.logging.info('Finished writing, total {} annotations, '
                    'skipped {} images.'.format(length, skipped))


def main(_):
    image_folder = FLAGS.image_folder
    annotation_folder = FLAGS.annotation_folder
    coco_minival_ids_file = FLAGS.coco_minival_ids_file
    output_folder = FLAGS.output_folder

    assert image_folder,\
        'required argument `image_folder` is missing'
    assert annotation_folder,\
        'required argument `annotation_folder` is missing'
    assert coco_minival_ids_file,\
        'required argument `coco_minival_ids_file` is missing'
    assert output_folder,\
        'required argument `output_folder` is missing'

    if not tf.gfile.IsDirectory(output_folder):
        tf.gfile.MakeDirs(output_folder)

    train_image = os.path.join(image_folder, TRAIN_FOLDER_NAME)
    train_annotation = os.path.join(annotation_folder, TRAIN_ANNOTATION_NAME)
    train_tfrecord = os.path.join(output_folder, TRAIN_TFRECORD_NAME)
    val_image = os.path.join(image_folder, VAL_FOLDER_NAME)
    val_annotation = os.path.join(annotation_folder, VAL_ANNOTATION_NAME)
    val_tfrecord = os.path.join(output_folder, VAL_TFRECORD_NAME)

    create_tfrecord_from_coco(train_image, val_image,
                              train_annotation, val_annotation,
                              train_tfrecord, val_tfrecord,
                              coco_minival_ids_file)


if __name__ == '__main__':
    tf.app.run()
