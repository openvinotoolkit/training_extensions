""" This module allows you to create and write tf record dataset. """

import argparse
import os
import tempfile
from tqdm import tqdm

import numpy as np

import tensorflow as tf
import cv2

import annotation


def parse_args():
    """ Parses arguments. """

    args = argparse.ArgumentParser()
    args.add_argument('--input_datasets', required=True, help='Comma-separated datasets paths.')
    args.add_argument('--output', required=True,
                      help='Path where output tf record will be written to.')
    args.add_argument('--imshow_delay', type=int, default=-1,
                      help='If it is non-negative, this script will draw detected and groundtruth'
                           'boxes')

    return args.parse_args()


def convert_to_example(image_data, labels, labels_text, bboxes, oriented_bboxes, shape):
    """ Convert dataset element to tf.train.Example. """

    oriented_bboxes = np.asarray(oriented_bboxes)
    bboxes = np.asarray(bboxes)

    def get_list(obj, idx):
        if len(obj) > 0:
            return list(obj[:, idx])
        return []

    def float_feature(feature):
        return tf.train.Feature(float_list=tf.train.FloatList(value=feature))

    def byte_feature(feature):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=feature))

    def int64_feature(feature):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=feature))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(shape))),
        'image/object/bbox/xmin': float_feature(get_list(bboxes, 0)),
        'image/object/bbox/ymin': float_feature(get_list(bboxes, 1)),
        'image/object/bbox/xmax': float_feature(get_list(bboxes, 2)),
        'image/object/bbox/ymax': float_feature(get_list(bboxes, 3)),
        'image/object/bbox/x1': float_feature(get_list(oriented_bboxes, 0)),
        'image/object/bbox/y1': float_feature(get_list(oriented_bboxes, 1)),
        'image/object/bbox/x2': float_feature(get_list(oriented_bboxes, 2)),
        'image/object/bbox/y2': float_feature(get_list(oriented_bboxes, 3)),
        'image/object/bbox/x3': float_feature(get_list(oriented_bboxes, 4)),
        'image/object/bbox/y3': float_feature(get_list(oriented_bboxes, 5)),
        'image/object/bbox/x4': float_feature(get_list(oriented_bboxes, 6)),
        'image/object/bbox/y4': float_feature(get_list(oriented_bboxes, 7)),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': byte_feature(labels_text),
        'image/format': byte_feature([b'JPEG']),
        'image/encoded': byte_feature([image_data])}))
    return example


def write_to_tfrecords(output_path, datasets, resize_to=None, imshow_delay=-1):
    """ Write datasets to tf record file. """

    assert isinstance(datasets, list)

    def visualize(image, bboxes, imshow_delay):
        for xmin, ymin, xmax, ymax in bboxes:
            xmin, ymin, xmax, ymax = xmin * weight, ymin * height, xmax * weight, ymax * height
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)

        image = cv2.resize(image, image_resized.shape[0:2][::-1])
        cv2.imshow('image', image)
        cv2.waitKey(imshow_delay)

    ignore_label = -1
    text_label = 1

    tmpfile = os.path.join(tempfile.mkdtemp(), 'image.png')

    with tf.io.TFRecordWriter(output_path) as tfrecord_writer:
        for dataset_path in datasets:
            for frame in tqdm(annotation.TextDetectionDataset(dataset_path).annotation):
                image = cv2.imread(frame['image_path'], cv2.IMREAD_COLOR)
                image_resized = cv2.resize(image, resize_to) if resize_to is not None else image
                cv2.imwrite(tmpfile, image_resized)
                image_data = tf.io.gfile.GFile(tmpfile, 'rb').read()

                shape = image.shape
                height, weight = shape[0:2]

                bboxes = []
                labels = []
                labels_text = []
                oriented_bboxes = []

                for bbox in frame['bboxes']:
                    oriented_box = np.asarray(bbox['quadrilateral'], dtype=np.float32)
                    oriented_box = oriented_box / ([weight, height] * 4)
                    np.clip(oriented_box, 0.0, 1.0, out=oriented_box)

                    x_coordinates = oriented_box.reshape(4, 2)[:, 0]
                    y_coordinates = oriented_box.reshape(4, 2)[:, 1]
                    bboxes.append([x_coordinates.min(), y_coordinates.min(),
                                   x_coordinates.max(), y_coordinates.max()])
                    labels_text.append(bbox['transcription'].encode("utf8"))
                    oriented_bboxes.append(oriented_box)

                    if not bbox['readable']:
                        labels.append(ignore_label)
                    else:
                        labels.append(text_label)

                example = convert_to_example(image_data, labels, labels_text,
                                             bboxes, oriented_bboxes, shape)
                tfrecord_writer.write(example.SerializeToString())

                if imshow_delay >= 0:
                    visualize(image, bboxes, imshow_delay)


def main():
    """ Main function. """

    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_to_tfrecords(output_path=args.output, datasets=args.input_datasets.split(','),
                       imshow_delay=args.imshow_delay)


if __name__ == '__main__':
    main()
