""" This module allows you to convert source dataset to internal format. """

import argparse
import collections
import json
import os

import numpy as np
from tqdm import tqdm

import cv2


def parse_args():
    """ Parses input arguments. """

    args = argparse.ArgumentParser()
    args.add_argument('--out_annotation', help='Path where annotaion will be saved to.',
                      required=True)
    args.add_argument('--in_annotation', help='Path to annotation in source format.')
    args.add_argument('--images', help='Path to dataset images.', required=True)
    args.add_argument('--type', choices=['icdar15', 'toy', 'icdar17_mlt', 'icdar19_mlt', 'cocotext_v2'],
                      help='Source dataset type/name.', required=True)
    args.add_argument('--train', action='store_true')
    args.add_argument('--imshow_delay', type=int, default=-1,
                      help='If it is non-negative, this script will draw detected and groundtruth'
                           'boxes')

    return args.parse_args()


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
    def read_from_coco_text(path, no_boxes_is_ok=False, sets=['train']):
        """ Converts annotation from COCO-TEXT format to internal format. """

        dataset = TextDetectionDataset()

        with open(path) as read_file:

            json_loaded = json.load(read_file)

            for id, value in json_loaded['imgs'].items():
                image_path = os.path.join(os.path.dirname(path),'train2014', value['file_name'])
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
                        'language': language}
                    )


                if no_boxes_is_ok or bboxes:
                    dataset.annotation.append({
                        'image_path': image_path,
                        'bboxes': bboxes
                    })

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
    def read_from_icdar2017_mlt(folder, is_training):
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
            annotation_path = os.path.join(folder, 'ch8_validation_localization_transcription_gt_v2', 'gt_' + image_path)[:-3] + 'txt'
            image_path = os.path.join(folder, 'ch8_validation_images', image_path)

            frame = {'image_path': image_path, 'bboxes': []}

            with open(annotation_path, encoding='utf-8-sig') as read_file:
                content = [line.strip() for line in read_file.readlines()]
                for line in content:
                    frame['bboxes'].append(parse_line(line))

            dataset.annotation.append(frame)

        return dataset


def main():
    """ Main function. """
    args = parse_args()

    if args.type == 'icdar15':
        text_detection_dataset = TextDetectionDataset.read_from_icdar2015(
            args.images, args.annotation, is_training=args.train)
    elif args.type == 'icdar19_mlt':
        text_detection_dataset = TextDetectionDataset.read_from_icdar2019_mlt(args.images)
    elif args.type == 'icdar17_mlt':
        text_detection_dataset = TextDetectionDataset.read_from_icdar2017_mlt(args.images, is_training=False)
    elif args.type == 'cocotext_v2':
        text_detection_dataset = TextDetectionDataset.read_from_coco_text(args.images)
    elif args.type == 'toy':
        text_detection_dataset = TextDetectionDataset.read_from_toy_dataset(args.images)

    text_detection_dataset.write(args.out_annotation)
    if args.imshow_delay >= 0:
        text_detection_dataset.visualize(put_text=True, imshow_delay=args.imshow_delay)


if __name__ == '__main__':
    main()
