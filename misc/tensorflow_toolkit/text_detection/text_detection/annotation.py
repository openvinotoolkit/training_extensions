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

import os
import json
import tempfile
from tqdm import tqdm
import cv2
import numpy as np
import tensorflow as tf


class TextDetectionDataset:
    """ TextDetectionDataset list of following instances
        {
        'image_path',
        'bboxes':
            [
                {
                    'quadrilateral': [int, int, int, int, int, int, int, int],
                    'transcription': str
                    'language': str
                    'readable': bool
                },
                ...
            ]
        }
    """

    def __init__(self, path=None):
        if path is None:
            self.annotation = []
        else:
            if os.path.exists(path):
                with open(path) as read_file:
                    self.annotation = json.load(read_file)

    def __add__(self, dataset):
        text_detection_dataset = TextDetectionDataset()
        text_detection_dataset.annotation = self.annotation + dataset.annotation
        return text_detection_dataset

    def __len__(self):
        return len(self.annotation)

    def write(self, path):
        """ Writes dataset annotation as json file. """

        with open(path, 'w') as read_file:
            json.dump(self.annotation, read_file)

    def visualize(self, put_text, imshow_delay=1):
        """ Visualizes annotation using cv2.imshow from OpenCV. Press `Esc` to exit. """

        for frame in tqdm(self.annotation):
            image = cv2.imread(frame['image_path'], cv2.IMREAD_COLOR)
            for bbox in frame['bboxes']:
                lwd = 2
                color = (0, 255, 0)
                if not bbox['readable']:
                    color = (128, 128, 128)
                points = bbox['quadrilateral']
                if put_text:
                    cv2.putText(image, bbox['transcription'], tuple(points[0:2]), 1, 1.0, color)
                cv2.line(image, tuple(points[0:2]), tuple(points[2:4]), color, lwd)
                cv2.line(image, tuple(points[2:4]), tuple(points[4:6]), color, lwd)
                cv2.line(image, tuple(points[4:6]), tuple(points[6:8]), color, lwd)
                cv2.line(image, tuple(points[6:8]), tuple(points[0:2]), color, lwd)
            try:
                image = cv2.resize(image, (1920, 1080))
                cv2.imshow('image', image)
                k = cv2.waitKey(imshow_delay)
                if k == 27:
                    break
            except:
                print('Error: image is empty or corrupted: ', frame['image_path'])

    @staticmethod
    def read_from_icdar2015(images_folder, annotations_folder, is_training):
        """ Converts annotation from ICDAR 2015 format to internal format. """

        def parse_line(line):
            line = line.split(',')
            quadrilateral = [int(x) for x in line[:8]]
            transcription = ','.join(line[8:])
            readable = True
            language = 'english'
            if transcription == '###':
                transcription = ''
                readable = False
                language = ''
            return {'quadrilateral': quadrilateral, 'transcription': transcription,
                    'readable': readable, 'language': language}

        dataset = TextDetectionDataset()

        n_images = 1000 if is_training else 500
        for i in range(1, n_images + 1):
            image_path = os.path.join(images_folder, 'img_{}.jpg'.format(i))
            annotation_path = os.path.join(annotations_folder, 'gt_img_{}.txt'.format(i))

            frame = {'image_path': image_path,
                     'bboxes': []}

            with open(annotation_path, encoding='utf-8-sig') as read_file:
                content = [line.strip() for line in read_file.readlines()]
                for line in content:
                    frame['bboxes'].append(parse_line(line))

            dataset.annotation.append(frame)

        return dataset

    @staticmethod
    def read_from_icdar2013(images_folder, annotations_folder, is_training):
        """ Converts annotation from ICDAR 2013 format to internal format. """

        def parse_line(line, sep):
            line = line.split(sep)
            xmin, ymin, xmax, ymax = [int(x) for x in line[:4]]
            assert xmin < xmax
            assert ymin < ymax
            quadrilateral = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
            transcription = (sep.join(line[4:]))[1:-1]
            return {'quadrilateral': quadrilateral, 'transcription': transcription,
                    'readable': True, 'language': 'english'}

        dataset = TextDetectionDataset()

        begin, end = (100, 328 + 1) if is_training else (1, 233 + 1)
        gt_format = 'gt_{}.txt' if is_training else 'gt_img_{}.txt'
        img_format = '{}.jpg' if is_training else 'img_{}.jpg'

        for i in range(begin, end):
            frame = {'image_path': os.path.join(images_folder, img_format.format(i)), 'bboxes': []}
            annotation_path = os.path.join(annotations_folder, gt_format.format(i))

            with open(annotation_path, encoding='utf-8-sig') as read_file:
                for line in [line.strip() for line in read_file.readlines()]:
                    frame['bboxes'].append(parse_line(line, sep=' ' if is_training else ', '))

            dataset.annotation.append(frame)

        return dataset

    @staticmethod
    def read_from_msra_td500(folder):
        """ Converts annotation from MSRA-TD500 format to internal format. """

        def parse_line(line):
            line = line.split(' ')
            _, difficult, top_left_x, top_left_y, width, height, rotation = [float(x) for x in line]
            box = cv2.boxPoints(((top_left_x + width / 2, top_left_y + height / 2),
                                 (width, height), rotation * 57.2958))
            quadrilateral = [int(x) for x in box.reshape([-1])]
            readable = difficult == 0
            return {'quadrilateral': quadrilateral, 'transcription': '',
                    'readable': readable, 'language': ''}

        dataset = TextDetectionDataset()

        for image_name in sorted(os.listdir(folder)):
            if image_name.endswith('JPG'):
                image_path = os.path.join(folder, image_name)
                annotation_path = os.path.join(folder, image_name.replace('.JPG', '.gt'))

                frame = {'image_path': image_path,
                         'bboxes': []}

                with open(annotation_path, encoding='utf-8-sig') as read_file:
                    content = [line.strip() for line in read_file.readlines()]
                    for line in content:
                        frame['bboxes'].append(parse_line(line))

                dataset.annotation.append(frame)

        return dataset

    @staticmethod
    def read_from_coco_text(path, no_boxes_is_ok=False, sets=None):
        """ Converts annotation from COCO-TEXT format to internal format. """

        if sets is None:
            sets = ['train']

        dataset = TextDetectionDataset()

        with open(path) as read_file:

            json_loaded = json.load(read_file)

            for id, value in json_loaded['imgs'].items():
                image_path = os.path.join(os.path.dirname(path), 'train2014', value['file_name'])
                dataset_type = value['set']

                if dataset_type not in sets:
                    continue

                bboxes = []
                for annotation_id  in json_loaded['imgToAnns'][id]:
                    annotation_value = json_loaded['anns'][str(annotation_id)]

                    text = annotation_value['utf8_string']
                    language = annotation_value['language']
                    readable = annotation_value['legibility'] == 'legible'

                    mask = np.reshape(np.array(annotation_value['mask'], np.int32), (-1, 2))
                    box = cv2.boxPoints(cv2.minAreaRect(mask))
                    quadrilateral = [int(x) for x in box.reshape([-1])]

                    bboxes.append({
                        'quadrilateral': quadrilateral,
                        'transcription': text,
                        'readable': readable,
                        'language': language})


                if no_boxes_is_ok or bboxes:
                    dataset.annotation.append({
                        'image_path': image_path,
                        'bboxes': bboxes})

        return dataset

    @staticmethod
    def read_from_toy_dataset(folder):
        """ Converts annotation from toy dataset (available) to internal format. """

        def parse_line(line):
            line = line.split(',')
            quadrilateral = [int(x) for x in line[:8]]
            transcription = ','.join(line[8:])
            readable = True
            language = ''
            if transcription == '###':
                transcription = ''
                readable = False
                language = ''
            return {'quadrilateral': quadrilateral, 'transcription': transcription,
                    'readable': readable, 'language': language}

        dataset = TextDetectionDataset()

        n_images = 5
        for i in range(1, n_images + 1):
            image_path = os.path.join(folder, 'img_{}.jpg'.format(i))
            annotation_path = os.path.join(folder, 'gt_img_{}.txt'.format(i))

            frame = {'image_path': image_path,
                     'bboxes': []}

            with open(annotation_path, encoding='utf-8-sig') as read_file:
                content = [line.strip() for line in read_file.readlines()]
                for line in content:
                    frame['bboxes'].append(parse_line(line))

            # for batch 20
            for _ in range(4):
                dataset.annotation.append(frame)

        return dataset

    @staticmethod
    def read_from_icdar2019_mlt(folder):
        """ Converts annotation from toy dataset (available) to internal format. """

        def parse_line(line):
            line = line.split(',')
            quadrilateral = [int(x) for x in line[:8]]
            language = line[8]
            transcription = ','.join(line[9:])
            readable = True
            if transcription == '###':
                transcription = ''
                readable = False
            return {'quadrilateral': quadrilateral, 'transcription': transcription,
                    'readable': readable, 'language': language}

        dataset = TextDetectionDataset()


        for image_part in [1, 2]:
            for image_path in os.listdir(os.path.join(folder, 'ImagesPart{}'.format(image_part))):
                annotation_path = os.path.join(folder, 'train_gt_t13', image_path)[:-3] + 'txt'
                image_path = os.path.join(folder, 'ImagesPart{}'.format(image_part), image_path)

                frame = {'image_path': image_path, 'bboxes': []}

                with open(annotation_path, encoding='utf-8-sig') as read_file:
                    content = [line.strip() for line in read_file.readlines()]
                    for line in content:
                        frame['bboxes'].append(parse_line(line))

                dataset.annotation.append(frame)

        return dataset

    @staticmethod
    def read_from_icdar2017_mlt(folder, _):
        """ Converts annotation from toy dataset (available) to internal format. """

        def parse_line(line):
            line = line.split(',')
            quadrilateral = [int(x) for x in line[:8]]
            language = line[8]
            transcription = ','.join(line[9:])
            readable = True
            if transcription == '###':
                transcription = ''
                readable = False
            return {'quadrilateral': quadrilateral, 'transcription': transcription,
                    'readable': readable, 'language': language}

        dataset = TextDetectionDataset()

        for image_path in os.listdir(os.path.join(folder, 'ch8_validation_images')):
            annotation_path = os.path.join(folder, 'ch8_validation_localization_transcription_gt_v2',
                                           'gt_' + image_path)[:-3] + 'txt'
            image_path = os.path.join(folder, 'ch8_validation_images', image_path)

            frame = {'image_path': image_path, 'bboxes': []}

            with open(annotation_path, encoding='utf-8-sig') as read_file:
                content = [line.strip() for line in read_file.readlines()]
                for line in content:
                    frame['bboxes'].append(parse_line(line))

            dataset.annotation.append(frame)

        return dataset



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
            for frame in tqdm(TextDetectionDataset(dataset_path).annotation):
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
