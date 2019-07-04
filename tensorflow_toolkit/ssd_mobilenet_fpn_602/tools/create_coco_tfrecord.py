import contextlib2
import json
import hashlib
import os

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import tensorflow as tf


"""
This script helps generate similar training and validation splits 
as the ones used in TensorFlow Object Detection API, which is much
different than organic coco train2017 and val2017.
By selecting ~8k specific images(called minival) for validation,
total training image set would be [coco_train2017+coco_val2017-minival].

This script is designed for benchmarking a fine-tuned model with
retraining or transfer learning based on a TensorFlow Object
Detection API pre-trained model to have a fair and reliable validation
result, since the pre-trained models are claimed having been trained with
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
"""


flags = tf.app.flags
tf.flags.DEFINE_string(
        'image_folder',
        None,
        'Folder of images, must include named train and val folders.')
tf.flags.DEFINE_string(
        'annotation_folder',
        None,
        'Folder of annotations, must include named annoation files.')
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


def create_tf_example(image,
                      anno_list,
                      image_folders,
                      category_index):

    image_h = image['height']
    image_w = image['width']
    filename = image['file_name']
    image_id = image['id']
    if image_w <= 0 and image_h <= 0:
        raise ValueError('image width or height is smaller than or equal to zero!')

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
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return key, example


def parse_minival_ids(coco_minival_ids_file):
    assert os.path.exists(coco_minival_ids_file), \
        'minival_ids_file {} does not exist!'.format(coco_minival_ids_file)
    with open(coco_minival_ids_file, 'r') as f:
        minival_list = [int(line) for line in f.readlines()]
    assert minival_list,\
        'minival ids is None, please check the coco_minival_ids_file'

    return minival_list


def load_annotation(annotation_file):
    assert os.path.exists(annotation_file), \
        'annotation file {} does not exist!'.format(annotation_file)
    with tf.gfile.GFile(annotation_file, 'r') as f:
        info = json.load(f)
    assert info,\
        'annotation is None, please check the annotation file'

    images = info['images']
    annotations = info['annotations']
    categories = info['categories']

    return images, annotations, categories


def merge_annotations(train_annotation_file, val_annotation_file):
    train_images, train_annotations, categories =\
        load_annotation(train_annotation_file)
    val_images, val_annotations, _ =\
        load_annotation(val_annotation_file)

    tf.logging.info('number of organic train2017 images is {}'
                    .format(len(train_images)))
    tf.logging.info('number of organic val2017 images is {}'
                    .format(len(val_images)))

    train_images.extend(val_images)
    train_annotations.extend(val_annotations)

    return train_images, train_annotations, categories


def split_minival(images, minival_list):
    train_plus_images = []
    minival_images = []
    for image in images:
        image_id = image['id']
        if image_id not in minival_list:
            train_plus_images.append(image)
        else:
            minival_images.append(image)
    return train_plus_images, minival_images


def generate_annotation_index(annotations):
    annotation_index = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id not in annotation_index:
            annotation_index[image_id] = []
        annotation_index[image_id].append(annotation)
    return annotation_index


def write_tfrecord(image_arr,
                   anno_index,
                   category_index,
                   image_folders,
                   output_tfrecord,
                   tfrecord_shards=100,
                   logging_interval=1000):
    length = len(image_arr)

    with contextlib2.ExitStack() as tf_record_close_stack:
        tfrecord_writer = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_tfrecord, tfrecord_shards)
        skipped = 0
        for i, image in enumerate(image_arr):
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
                pass

    tf.logging.info('Finished writing, total {} annotations, '
                    'skipped {} images.'.format(length, skipped))


def create_tfrecord_from_coco(train_image_folder,
                              val_image_folder,
                              train_annotation_file,
                              val_annotation_file,
                              train_tfrecord,
                              val_tfrecord,
                              coco_minival_ids_file):
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
    write_tfrecord(image_arr=train_plus_images,
                   anno_index=annotation_index,
                   category_index=category_index,
                   image_folders=image_folders,
                   output_tfrecord=train_tfrecord)
    write_tfrecord(image_arr=minival_images,
                   anno_index=annotation_index,
                   category_index=category_index,
                   image_folders=image_folders,
                   output_tfrecord=val_tfrecord,
                   tfrecord_shards=10)


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

