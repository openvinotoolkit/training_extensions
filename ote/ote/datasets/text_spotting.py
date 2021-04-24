"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import copy
import json
import logging
import os
from collections import defaultdict

import cv2
import imagesize # pylint: disable=import-error
import numpy as np
from tqdm import tqdm


def box2poly(box):
    x, y, w, h = box
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def poly2box(poly):
    xs = poly[0::2]
    ys = poly[1::2]
    xmin = min(xs)
    xmax = max(xs)

    ymin = min(ys)
    ymax = max(ys)

    return [xmin, ymin, xmax - xmin, ymax - ymin]


class TextOnlyCocoAnnotation:
    """ Class for working with MSCOCO-like annotation for text. """

    def __init__(self, path=None, root=''):

        self.label_map = {'text': 1}

        self.annotation = {
            "type": "instances",
            "images": [],
            "categories": sorted([{"supercategory": "none", "name": key, "id": value}
                                  for key, value in self.label_map.items()],
                                 key=lambda x: x["id"]),
            "annotations": [],
        }

        if path is not None:
            assert os.path.exists(path), path
            with open(path) as read_file:
                self.annotation = json.load(read_file)

        if root:
            for image_info in self.annotation['images']:
                image_info['file_name'] = os.path.join(root, image_info['file_name'])

        self.img_id_2_ann_id = defaultdict(list)
        for index, ann in enumerate(self.annotation['annotations']):
            assert index == ann['id']
            self.img_id_2_ann_id[ann['image_id']].append(ann['id'])

        self.img_path_2_img_id = dict()
        for index, img in enumerate(self.annotation['images']):
            assert index == img['id']
            self.img_path_2_img_id[img['file_name']] = index

    @staticmethod
    def fit_box_in_image(box, image_size):
        width, height = image_size
        x, y, w, h = box
        if x + w > width:
            w = width - x
        if y + h > height:
            h = height - y
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        if 0 <= x < x + w <= width and 0 <= y < y + h <= height:
            return x, y, w, h
        return None

    def add_bbox(self, image_path, image_size, obj):
        """ Adds new text object to annotation. """

        if image_path not in self.img_path_2_img_id:
            self.img_path_2_img_id[image_path] = len(self.img_path_2_img_id)

            self.annotation['images'].append({
                "file_name": image_path,
                "height": image_size[1],
                "width": image_size[0],
                "id": self.img_path_2_img_id[image_path]
            })

        new_ann_id = len(self.annotation['annotations'])
        self.img_id_2_ann_id[self.img_path_2_img_id[image_path]].append(new_ann_id)
        bbox = self.fit_box_in_image(obj['bbox'], image_size)

        if bbox:
            self.annotation['annotations'].append({
                "bbox": bbox,  # x, y, w, h
                "segmentation": obj['segmentation'],
                "attributes": obj['attributes'],
                "ignore": 0,
                "id": new_ann_id,
                "image_id": self.img_path_2_img_id[image_path],
                "area": obj['bbox'][2] * obj['bbox'][3],
                "iscrowd": 1 - int(obj['attributes']['legible']),
                "category_id": self.label_map['text']
            })

    def __iadd__(self, other):
        ignored = []

        for image_info in tqdm(other.annotation['images']):
            ann_ids = other.img_id_2_ann_id[image_info['id']]

            image = cv2.imread(image_info['file_name'])
            if image.shape[:2] != (image_info['height'], image_info['width']):
                ignored.append(image_info['file_name'])
                continue
            for ann_id in ann_ids:
                ann = other.annotation['annotations'][ann_id]
                self.add_bbox(image_info['file_name'], (image_info['width'], image_info['height']),
                              copy.deepcopy(ann))
        for i, item in enumerate(ignored):
            logging.warning(f'Shapes mismatch (annotation vs image), skipped: {i}, {item}')
        return self

    def write(self, path):
        """ Writes annotation as json file. """

        annotation = copy.deepcopy(self.annotation)

        for image_info in annotation['images']:
            image_info['file_name'] = os.path.relpath(image_info['file_name'],
                                                      os.path.dirname(path))

        with open(path, 'w') as read_file:
            json.dump(annotation, read_file)

    def check_objects_inside(self):
        for frame in tqdm(self.annotation['images']):
            image_path = frame['file_name']
            image_size = imagesize.get(image_path)
            for ann_id in self.img_id_2_ann_id[frame['id']]:
                obj = self.annotation['annotations'][ann_id]
                bbox = obj['bbox']
                is_xmax_ok = 0 <= bbox[0] < bbox[0] + bbox[2] < image_size[0]
                is_ymax_ok = 0 <= bbox[1] < bbox[1] + bbox[3] < image_size[1]
                assert is_xmax_ok and is_ymax_ok, f'{image_path} {bbox}'

    def visualize(self, put_text, imshow_delay=1, shuffle=False):
        """ Visualizes annotation using cv2.imshow from OpenCV. Press `Esc` to exit. """

        max_image_size = 1280, 768

        images = list(self.annotation['images'])
        if shuffle:
            import random
            random.shuffle(images)

        for frame in tqdm(images):
            image_path = frame['file_name']
            assert os.path.exists(image_path), f'does not exist {image_path}'
            image = cv2.imread(image_path)
            for ann_id in self.img_id_2_ann_id[frame['id']]:
                obj = self.annotation['annotations'][ann_id]
                lwd = 2
                color = (0, 255, 0)
                if obj['iscrowd']:
                    color = (128, 128, 128)
                bbox = obj['bbox']
                if put_text:
                    cv2.putText(image, obj['attributes']['transcription'], tuple(bbox[0:2]), 1, 1.0,
                                color)
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              color, lwd)

                contours = np.array(obj['segmentation'])
                contours = contours.reshape([contours.shape[0], contours.shape[1] // 2, 2])

                cv2.drawContours(image, contours, 0, color, 1)

                if 'chars' in obj['attributes']:
                    for char in obj['attributes']['chars']:
                        bbox = char['bbox']
                        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color)
                        if put_text:
                            cv2.putText(image, char['char'], tuple(char['bbox'][0:2]), 1, 1.0, color)
            try:
                if image.shape[1] > max_image_size[0] or image.shape[0] > max_image_size[1]:
                    print('resized')
                    image = cv2.resize(image, max_image_size)
                cv2.imshow('image', image)
                k = cv2.waitKey(imshow_delay)
                if k == 27:
                    break

            except (IndexError, cv2.error):
                print('Error: image is empty or corrupted: ', frame['file_name'])

    def extract_text_recognition_dataset(self, path):
        """  Crops text instances and saves as another dataset. """

        os.makedirs(os.path.join(path, 'images'))

        annotation = []

        for frame in tqdm(self.annotation['images']):
            image = cv2.imread(frame['file_name'], cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
            for ann_id in self.img_id_2_ann_id[frame['id']]:
                obj = self.annotation['annotations'][ann_id]
                if obj['attributes']['legible']:
                    bbox = obj['bbox']
                    try:
                        transcription = obj['attributes']['transcription']
                        if transcription.isalnum():
                            coord_x1, coord_y1, coord_x2, coord_y2 = bbox[0], bbox[1], bbox[0] + \
                                                                     bbox[2], bbox[1] + bbox[3]
                            coord_x1 = max(0, coord_x1)
                            coord_x2 = min(image.shape[1] - 1, coord_x2)
                            coord_y1 = max(0, coord_y1)
                            coord_y2 = min(image.shape[0] - 1, coord_y2)
                            crop_path = os.path.join(path, 'images', f'image{len(annotation)}.jpg')
                            annotation.append(f'{crop_path} {transcription}')
                            cv2.imwrite(crop_path, image[coord_y1:coord_y2, coord_x1:coord_x2])
                    except KeyError:
                        print('Missing transcription in ', frame['file_name'])
                        break
                    except IndexError:
                        print('Error in image processing ', frame['file_name'])
                        break

        with open(os.path.join(path, 'annotation.txt'), 'w') as file:
            file.write('\n'.join(annotation))


class ICDAR2013DatasetConverter:
    """ Class for conversion of ICDAR2013 to TextOnlyCocoAnnotation. """

    def __init__(self, images_folder, annotations_folder, character_annotations_folder='', is_train=True, root=''):
        self.images_folder = images_folder
        self.annotations_folder = annotations_folder
        self.characters_annotations_folder = character_annotations_folder
        self.is_train = is_train

        if root:
            self.annotations_folder = os.path.join(root, self.annotations_folder)
            self.images_folder = os.path.join(root, self.images_folder)
            if character_annotations_folder:
                self.characters_annotations_folder = os.path.join(root, self.characters_annotations_folder)

    def __call__(self, *args, **kwargs):
        dataset = TextOnlyCocoAnnotation()

        begin, end = (100, 328 + 1) if self.is_train else (1, 233 + 1)
        gt_format = 'gt_{}.txt' if self.is_train else 'gt_img_{}.txt'
        img_format = '{}.jpg' if self.is_train else 'img_{}.jpg'
        char_gt_format = '{}_GT.txt'

        for i in tqdm(range(begin, end)):
            image_path = os.path.join(self.images_folder, img_format.format(i))
            image_size = imagesize.get(image_path)
            annotation_path = os.path.join(self.annotations_folder, gt_format.format(i))

            with open(annotation_path, encoding='utf-8-sig') as read_file:
                if self.characters_annotations_folder:
                    char_annotation_path = os.path.join(self.characters_annotations_folder, char_gt_format.format(i))
                    with open(char_annotation_path) as f:
                        content = f.readlines()
                        content = ''.join(content)
                        content = content.split('\n\n')
                        characters = [line.split('\n') for line in content if not line.strip().startswith('#')]
                for j, line in enumerate([line.strip() for line in read_file.readlines()]):
                    obj = self.parse_line(line)

                    if self.characters_annotations_folder:

                        obj['attributes']['chars'] = []
                        for chars in characters[j]:
                            if not chars:
                                continue
                            chars = chars.split(' ')
                            bbox = [int(x) for x in chars[5:9]]
                            bbox[2] -= bbox[0]
                            bbox[3] -= bbox[1]
                            char = ' '.join(chars[9:])
                            if len(char) == 3 and char[0] == char[-1] == '"':
                                char = char[1]
                            if char == ' ':
                                continue
                            assert len(char) == 1, f'char = "{char}"'
                            obj['attributes']['chars'].append({
                                'bbox': bbox,
                                'segmentation': box2poly(bbox),
                                'char': char
                            })
                        united_chars = ''.join([x['char'] for x in obj['attributes']['chars']])
                        if united_chars != obj['attributes']['transcription']:
                            logging.warning(f"Transcription of {obj['attributes']['transcription']} in "
                                            f"{annotation_path} "
                                            f"has been changed to {united_chars}."
                                            f"It is known error in original annotation.")
                            obj['attributes']['transcription'] = united_chars

                    dataset.add_bbox(image_path, image_size, obj)

        return dataset

    def parse_line(self, line):
        """ Parses line of ICDAR2013 annotation. """

        sep = ' ' if self.is_train else ', '
        line = line.split(sep)
        xmin, ymin, xmax, ymax = [int(x) for x in line[:4]]
        assert xmin < xmax
        assert ymin < ymax
        transcription = (sep.join(line[4:]))[1:-1]
        word_annotation = {
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
            'segmentation': [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]],
            'attributes': {
                'transcription': transcription,
                'legible': 1,
                'language': 'english',
            }
        }
        return word_annotation


class ICDAR2015DatasetConverter:
    """ Class for conversion of ICDAR2015 to TextOnlyCocoAnnotation. """

    def __init__(self, images_folder, annotations_folder, is_train, root=''):
        self.images_folder = images_folder
        self.annotations_folder = annotations_folder
        self.is_train = is_train

        if root:
            self.annotations_folder = os.path.join(root, self.annotations_folder)
            self.images_folder = os.path.join(root, self.images_folder)

    @staticmethod
    def parse_line(line):
        """ Parses line of ICDAR2015 annotation. """

        line = line.split(',')
        quadrilateral = [int(x) for x in line[:8]]
        transcription = ','.join(line[8:])
        legible = 1
        language = 'english'
        if transcription == '###':
            transcription = ''
            legible = 0
            language = ''

        xmin = min(quadrilateral[0::2])
        xmax = max(quadrilateral[0::2])

        ymin = min(quadrilateral[1::2])
        ymax = max(quadrilateral[1::2])

        word_annotation = {
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
            'segmentation': [quadrilateral],
            'attributes': {
                'transcription': transcription,
                'legible': legible,
                'language': language,
            }
        }
        return word_annotation

    def __call__(self, *args, **kwargs):
        """ Converts annotation from ICDAR 2015 format to internal format. """

        dataset = TextOnlyCocoAnnotation()

        n_images = 1000 if self.is_train else 500
        for i in tqdm(range(1, n_images + 1)):
            image_path = os.path.join(self.images_folder, 'img_{}.jpg'.format(i))
            annotation_path = os.path.join(self.annotations_folder, 'gt_img_{}.txt'.format(i))

            with open(annotation_path, encoding='utf-8-sig') as read_file:
                content = [line.strip() for line in read_file.readlines()]
                for line in content:
                    dataset.add_bbox(image_path, imagesize.get(image_path), self.parse_line(line))

        return dataset


class ICDAR2017MLTDatasetConverter:
    """ Class for conversion of ICDAR2017 to TextOnlyCocoAnnotation. """

    def __init__(self, folder, subset, is_latin_required, root=''):
        '''
        Converts ICDAR2017 MLT to TextOnlyCocoAnnotation
        :param folder: Folder with extracted zip archives containing images and annotation.
        :param subset: 'train' or 'val'
        :param is_latin_required: if it is True than images that do not contain latin text will be
                                  filtered out.
        '''
        self.folder = folder
        self.subset = subset
        self.is_latin_required = is_latin_required

        if root:
            self.folder = os.path.join(root, self.folder)

        assert self.subset in ['train', 'val']

        if self.subset == 'train':
            for i in range(1, 9):
                assert os.path.exists(os.path.join(self.folder, f'ch8_training_images_{i}'))
            assert os.path.exists(
                os.path.join(self.folder, 'ch8_training_localization_transcription_gt_v2'))
        elif self.subset == 'val':
            assert os.path.exists(
                os.path.join(self.folder, 'ch8_validation_images'))
            assert os.path.exists(
                os.path.join(self.folder, 'ch8_validation_localization_transcription_gt_v2'))

    @staticmethod
    def parse_line(line):
        """ Parses line of ICDAR2015 annotation. """

        line = line.split(',')
        quadrilateral = [int(x) for x in line[:8]]
        language = line[8]
        transcription = ','.join(line[9:])
        legible = 1
        if transcription == '###':
            transcription = ''
            legible = 0
            language = ''

        xmin = min(quadrilateral[0::2])
        xmax = max(quadrilateral[0::2])

        ymin = min(quadrilateral[1::2])
        ymax = max(quadrilateral[1::2])

        word_annotation = {
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
            'segmentation': [quadrilateral],
            'attributes': {
                'transcription': transcription,
                'legible': legible,
                'language': language,
            }
        }
        return word_annotation

    def collect_train_paths(self):
        """ Collects images and annotations paths for training set. """

        image_paths = []
        annotation_paths = []
        n_images = 7200
        for i in tqdm(range(1, n_images + 1)):
            added = False
            for extension in ['jpg', 'png']:
                image_path = os.path.join(self.folder,
                                          f'ch8_training_images_{(i - 1) // 1000 + 1}',
                                          f'img_{i}.{extension}')
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    added = True
                    break
            if added:
                annotation_paths.append(
                    os.path.join(self.folder, 'ch8_training_localization_transcription_gt_v2',
                                 f'gt_img_{i}.txt')
                )
            else:
                logging.warning(f'Could not find: {image_path[:-3]}*')
        return image_paths, annotation_paths

    def collect_val_paths(self):
        """ Collects images and annotations paths for validation set. """

        image_paths = []
        annotation_paths = []
        n_images = 1800
        for i in tqdm(range(1, n_images + 1)):
            added = False
            for extension in ['jpg', 'png']:
                image_path = os.path.join(self.folder,
                                          'ch8_validation_images',
                                          f'img_{i}.{extension}')
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    added = True
                    break
            if added:
                annotation_paths.append(
                    os.path.join(self.folder, 'ch8_validation_localization_transcription_gt_v2',
                                 f'gt_img_{i}.txt')
                )
            else:
                logging.warning(f'Could not find: {image_path[:-3]}*')
        return image_paths, annotation_paths

    def __call__(self, *args, **kwargs):
        """ Converts annotation from ICDAR 2017 format to internal format. """

        dataset = TextOnlyCocoAnnotation()

        if self.subset == 'train':
            image_paths, annotation_paths = self.collect_train_paths()
        elif self.subset == 'val':
            image_paths, annotation_paths = self.collect_val_paths()

        for image_path, annotation_path in tqdm(zip(image_paths, annotation_paths)):
            word_annotations = []
            with open(annotation_path, encoding='utf-8-sig') as read_file:
                content = [line.strip() for line in read_file.readlines()]
                for line in content:
                    word_annotations.append(self.parse_line(line))
            should_add = not self.is_latin_required
            if self.is_latin_required:
                for word_annotation in word_annotations:
                    if word_annotation['attributes']['language'].lower() == 'latin':
                        should_add = True
                        break
            if should_add:
                for word_annotation in word_annotations:
                    dataset.add_bbox(image_path, imagesize.get(image_path), word_annotation)

        return dataset


class ICDAR2019MLTDatasetConverter:
    """ Class for conversion of ICDAR2019 to TextOnlyCocoAnnotation. """

    def __init__(self, folder, is_latin_required, root=''):
        '''
        Converts ICDAR2019 MLT to TextOnlyCocoAnnotation
        :param folder: Folder with extracted zip archives containing images and annotation.
        :param is_latin_required: if it is True than images that do not contain latin text will be
                                  filtered out.
        '''
        self.folder = folder
        self.is_latin_required = is_latin_required

        if root:
            self.folder = os.path.join(root, self.folder)

        assert os.path.exists(os.path.join(self.folder, 'ImagesPart1'))
        assert os.path.exists(os.path.join(self.folder, 'ImagesPart2'))
        assert os.path.exists(os.path.join(self.folder, 'train_gt_t13'))

    @staticmethod
    def parse_line(line):
        """ Parses line of ICDAR2019 annotation. """

        line = line.split(',')
        quadrilateral = [int(x) for x in line[:8]]
        language = line[8]
        transcription = ','.join(line[9:])
        legible = 1
        if transcription == '###':
            transcription = ''
            legible = 0
            language = ''

        xmin = min(quadrilateral[0::2])
        xmax = max(quadrilateral[0::2])

        ymin = min(quadrilateral[1::2])
        ymax = max(quadrilateral[1::2])

        word_annotation = {
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
            'segmentation': [quadrilateral],
            'attributes': {
                'transcription': transcription,
                'legible': legible,
                'language': language,
            }
        }
        return word_annotation

    def collect_train_paths(self):
        """ Collects images and annotations paths for training set. """

        image_paths = []
        annotation_paths = []

        n_images = 10000
        for i in range(1, n_images + 1):
            added = False
            for extension in ['jpg', 'png']:
                image_path = os.path.join(self.folder,
                                          f'ImagesPart{(i - 1) // 5000 + 1}',
                                          f'tr_img_{i:05}.{extension}')
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    added = True
                    break
            if added:
                annotation_paths.append(
                    os.path.join(self.folder, 'train_gt_t13', f'tr_img_{i:05}.txt')
                )
            else:
                print(f'Could not find: {image_path[:-3]}*')

        return image_paths, annotation_paths

    def __call__(self, *args, **kwargs):
        """ Converts annotation from ICDAR 2019 format to internal format. """

        dataset = TextOnlyCocoAnnotation()

        image_paths, annotation_paths = self.collect_train_paths()

        for image_path, annotation_path in tqdm(zip(image_paths, annotation_paths)):
            word_annotations = []
            with open(annotation_path, encoding='utf-8-sig') as read_file:
                content = [line.strip() for line in read_file.readlines()]
                for line in content:
                    word_annotations.append(self.parse_line(line))
            should_add = not self.is_latin_required
            if self.is_latin_required:
                for word_annotation in word_annotations:
                    if word_annotation['attributes']['language'].lower() == 'latin':
                        should_add = True
                        break
            if should_add:
                for word_annotation in word_annotations:
                    dataset.add_bbox(image_path, imagesize.get(image_path), word_annotation)

        return dataset


class ICDAR2019ARTDatasetConverter:

    def __init__(self, folder, is_latin_required=False, root='',
                 exclude_totaltext_test=False, totaltext_to_art_path='', totaltext_test_images_dir=''):
        '''
        Converts ICDAR2019 ART to TextOnlyCocoAnnotation
        :param folder: Folder with extracted zip archives containing images
        :param is_latin_required: if it is True than images that do not contain latin text will be filtered out.
        '''
        self.folder = folder
        self.is_latin_required = is_latin_required

        if root:
            self.folder = os.path.join(root, self.folder)

        assert os.path.exists(os.path.join(self.folder, 'train_images'))
        assert os.path.exists(os.path.join(self.folder, 'train_labels.json'))

        self.exclude_art19_ids = set()

        if exclude_totaltext_test:
            assert totaltext_test_images_dir
            assert totaltext_to_art_path

            if root:
                totaltext_test_images_dir = os.path.join(root, totaltext_test_images_dir)
                totaltext_to_art_path = os.path.join(root, totaltext_to_art_path)

            exclude_totaltext_ids = set(
                imagename.split('.')[0][3:] for imagename in os.listdir(totaltext_test_images_dir)
            )

            with open(totaltext_to_art_path) as read_file:
                self.exclude_art19_ids = set(x.split(' ')[1].split('.')[0] for x in read_file
                                             if x.split(' ')[0].split('.')[0][11:] in exclude_totaltext_ids)
    @staticmethod
    def parse_line(obj):
        """ Parses line of ICDAR2019ART annotation. """
        quadrilateral = []
        for point in obj['points']:
            quadrilateral += point
        xmin = min(quadrilateral[0::2])
        xmax = max(quadrilateral[0::2])

        ymin = min(quadrilateral[1::2])
        ymax = max(quadrilateral[1::2])
        if not (xmin < xmax and ymin < ymax):
            logging.warning(f"skip: {obj}")
            return None
        language = obj['language'].lower()
        legibility = 1 - int(obj['illegibility'])
        transcription = obj['transcription']
        if transcription == '###':
            transcription = ''
            legibility = 0
            language = ''

        word_annotation = {
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
            'segmentation': [quadrilateral],
            'attributes': {
                'transcription': transcription,
                'legible': legibility,
                'language': language,
            }
        }
        return word_annotation

    def __call__(self, *args, **kwargs):

        dataset = TextOnlyCocoAnnotation()

        img_format = '{}.jpg'
        annotation_path = os.path.join(self.folder, 'train_labels.json')

        with open(annotation_path) as f:
            annotations = json.load(f)
            for image in tqdm(annotations):
                image_path = os.path.join(self.folder, 'train_images', img_format.format(image))
                if image in self.exclude_art19_ids:
                    continue
                if not os.path.exists(image_path):
                    print(f'Could not find: {image_path[:-3]}*')

                word_annotations = []
                for instance in annotations[image]:
                    obj = self.parse_line(instance)
                    if obj:
                        word_annotations.append(obj)

                should_add = not self.is_latin_required
                if self.is_latin_required:
                    for word_annotation in word_annotations:
                        if word_annotation['attributes']['language'].lower() == 'latin':
                            should_add = True
                            break

                if should_add:
                    for word_annotation in word_annotations:
                        dataset.add_bbox(image_path, imagesize.get(image_path), word_annotation)

        return dataset


class MSRATD500DatasetConverter:
    """ Class for conversion of MSRA-TD500 to TextOnlyCocoAnnotation. """

    def __init__(self, folder, root=''):
        self.folder = folder

        if root:
            self.folder = os.path.join(root, self.folder)

    @staticmethod
    def parse_line(line):
        """ Parses line of MSRA-TD500 annotation. """

        line = line.split(' ')
        _, _, top_left_x, top_left_y, width, height, rotation = [float(x) for x in line]
        box = cv2.boxPoints(((top_left_x + width / 2, top_left_y + height / 2),
                             (width, height), rotation * 57.2958))
        quadrilateral = [int(x) for x in box.reshape([-1])]
        xmin = min(quadrilateral[0::2])
        xmax = max(quadrilateral[0::2])

        ymin = min(quadrilateral[1::2])
        ymax = max(quadrilateral[1::2])

        word_annotation = {
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
            'segmentation': [quadrilateral],
            'attributes': {
                'transcription': '',
                'legible': 1,
                'language': '',
            }
        }

        return word_annotation

    def __call__(self, *args, **kwargs):
        """ Converts annotation from MSRA-TD500 format to internal format. """

        dataset = TextOnlyCocoAnnotation()

        for image_name in tqdm(sorted(os.listdir(self.folder))):
            if image_name.endswith('JPG'):
                image_path = os.path.join(self.folder, image_name)
                annotation_path = os.path.join(self.folder, image_name.replace('.JPG', '.gt'))

                with open(annotation_path, encoding='utf-8-sig') as read_file:
                    content = [line.strip() for line in read_file.readlines()]
                    for line in content:
                        dataset.add_bbox(image_path, imagesize.get(image_path),
                                         self.parse_line(line))

        return dataset


class COCOTextDatasetConverter:
    """ Class for conversion of COCO-Text to TextOnlyCocoAnnotation. """

    def __init__(self, path, sets=None, root=''):
        self.path = path

        if root:
            self.path = os.path.join(root, self.path)

        self.sets = sets
        if self.sets is None:
            self.sets = ['train']  # 'val

    @staticmethod
    def parse_annotation_instance(annotation):
        """ Parses annotation instance of COCO-Text dataset. """

        text = annotation['utf8_string']
        language = annotation['language']
        legible = int(annotation['legibility'] == 'legible')

        mask = np.reshape(np.array(annotation['mask'], np.int32), (-1, 2))
        box = cv2.boxPoints(cv2.minAreaRect(mask))
        quadrilateral = [int(x) for x in box.reshape([-1])]

        xmin = min(quadrilateral[0::2])
        xmax = max(quadrilateral[0::2])

        ymin = min(quadrilateral[1::2])
        ymax = max(quadrilateral[1::2])

        word_annotation = {
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
            'segmentation': [quadrilateral],
            'attributes': {
                'transcription': text,
                'legible': legible,
                'language': language,
            }
        }

        return word_annotation

    def __call__(self):
        """ Converts annotation from COCO-TEXT format to internal format. """

        dataset = TextOnlyCocoAnnotation()

        with open(self.path) as read_file:

            json_loaded = json.load(read_file)

            for i, value in tqdm(json_loaded['imgs'].items()):
                image_path = os.path.join(os.path.dirname(self.path), 'train2014',
                                          value['file_name'])
                dataset_type = value['set']

                if dataset_type not in self.sets:
                    print(dataset_type)
                    continue

                for annotation_id in json_loaded['imgToAnns'][i]:
                    annotation_value = json_loaded['anns'][str(annotation_id)]
                    word_annotation = self.parse_annotation_instance(annotation_value)
                    dataset.add_bbox(image_path, imagesize.get(image_path), word_annotation)

        return dataset


class CvatXml11Converter:

    def __init__(self, images_folder, annotation_path, root=''):
        self.annotation_path = annotation_path
        self.images_folder = images_folder

    def __call__(self):
        dataset = TextOnlyCocoAnnotation()
        import xml.etree.ElementTree as ET
        mytree = ET.parse(self.annotation_path)
        myroot = mytree.getroot()
        for image_el in tqdm(myroot):
            if image_el.tag == 'image':
                image_name = image_el.get('name')
                image_path = os.path.join(self.images_folder, image_name)
                image_size = int(image_el.get('width')), int(image_el.get('height'))
                for polygon_el in image_el:
                    label = polygon_el.get('label')
                    if label == 'text':
                        points = polygon_el.get('points')
                        attributes = {}
                        for attribute in polygon_el:
                            attributes[attribute.get('name').lower()] = attribute.text

                        if attributes['language'].lower() != 'english':
                            continue

                        if attributes['legible'].lower() != 'true':
                            attributes['text'] = ''

                        if attributes['text'] is None:
                            continue

                        try:
                            word_polygon = [int(float(x)) for x in points.replace(';', ',').split(',')]
                        except Exception: # pylint: disable=broad-except
                            print('skipped')
                            print(ET.tostring(polygon_el, encoding='unicode'))
                            print('')
                            continue
                        word_annotation = {
                                'bbox': poly2box(word_polygon),
                                'segmentation': [word_polygon],
                                'attributes': {
                                    'transcription': attributes['text'],
                                    'legible': attributes['legible'].lower() == 'true',
                                    'language': attributes['language'].lower(),
                                    'chars': []
                                }
                            }
                        dataset.add_bbox(image_path, image_size, word_annotation)
        return dataset



str_to_class = {
    'ICDAR2013DatasetConverter': ICDAR2013DatasetConverter,
    'ICDAR2015DatasetConverter': ICDAR2015DatasetConverter,
    'ICDAR2017MLTDatasetConverter': ICDAR2017MLTDatasetConverter,
    'ICDAR2019MLTDatasetConverter': ICDAR2019MLTDatasetConverter,
    'ICDAR2019ARTDatasetConverter': ICDAR2019ARTDatasetConverter,
    'MSRATD500DatasetConverter': MSRATD500DatasetConverter,
    'COCOTextDatasetConverter': COCOTextDatasetConverter,
    'CvatXml11Converter': CvatXml11Converter
}
