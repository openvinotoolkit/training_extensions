"""
MIT License

Copyright (c) 2019 luopeixiang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

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

import json
import math
import os
import re
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from os.path import join

import cv2 as cv
import lmdb
import numpy as np
import scipy.io
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from ..data.utils import get_num_lines_in_file
from ..data.vocab import split_number

ALPHANUMERIC_VOCAB = set('abcdefghijklmnopqrstuvwxyz0123456789')


def rotate_point(xy_vector, angle):
    rho = math.sqrt(xy_vector[0] ** 2 + xy_vector[1] ** 2)
    theta = math.degrees(math.atan2(xy_vector[1], xy_vector[0]))
    theta = theta - angle
    x, y = rho * math.cos(math.radians(theta)), rho * math.sin(math.radians(theta))
    return x, y


class BatchRandomSampler(Sampler):
    """This is a class representing random batch sampler
    Only the indices of the dataset are shuffled so if dataset is sorted
    (e.g. on the shape) the original order is saved
    """

    def __init__(self, batch_size, dataset):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.len = len(dataset) // self.batch_size
        self.indices = np.array(range(self.len * self.batch_size))[::self.batch_size]

    def __iter__(self):
        np.random.shuffle(self.indices)
        batch = []
        for idx in self.indices:
            for shift in range(self.batch_size):
                batch.append(idx + shift)
            yield batch
            batch = []

    def __len__(self):
        return self.len


def img_size(pair):
    img = pair.get('img')
    return tuple(img.shape)


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.pairs = []

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)


class Im2LatexDataset(BaseDataset):
    def __init__(self, data_path, annotation_file, min_shape=(8, 8)):
        """args:
        data_path: root dir storing the prepoccessed data
        ann_file: path to annotation file
        """
        super().__init__()
        self.data_path = data_path
        self.images_dir = join(data_path, 'images_processed')
        self.formulas = self._get_formulas()
        self.pairs = self._get_pairs(annotation_file, min_shape)

    def _get_formulas(self):
        formulas_file = join(self.data_path, 'formulas.norm.lst')
        with open(formulas_file, 'r') as f:
            formulas = []
            for line in f:
                formula = line.strip('\n')
                res_formula = ''
                for sign in formula.split():
                    res_formula += ' ' + ' '.join(split_number(sign))

                res_formula = res_formula.lstrip()
                formulas.append(res_formula)
        return formulas

    def _get_pairs(self, subset, min_shape):
        # the line in this file map image to formulas
        map_file = join(self.data_path, subset)
        total_lines = get_num_lines_in_file(map_file)
        # get image-formulas pairs
        pairs = []
        with open(map_file, 'r') as f:
            for line in tqdm(f, total=total_lines):
                img_name, formula_id = line.strip('\n').split()
                # load img and its corresponding formula
                img_path = join(self.images_dir, img_name)
                img = cv.imread(img_path, cv.IMREAD_COLOR)
                if img.shape[0:2] <= tuple(min_shape):
                    continue
                formula = self.formulas[int(formula_id)]
                el = {'img_name': img_name,
                      'text': formula,
                      'img': img,
                      }
                pairs.append(el)
        pairs.sort(key=img_size, reverse=True)
        return pairs


class ICDAR2013RECDataset(BaseDataset):
    def __init__(self, data_path, annotation_file, root='', min_shape=(8, 8), grayscale=False,
                 fixed_img_shape=None, case_sensitive=True, min_txt_len=0):
        super().__init__()
        self.data_path = data_path
        self.annotation_file = annotation_file
        if root:
            self.annotation_file = os.path.join(root, self.annotation_file)
            self.data_path = os.path.join(root, self.data_path)
        self.pairs = self._load(min_shape, grayscale, fixed_img_shape, case_sensitive, min_txt_len)

    def _load(self, min_shape, grayscale, fixed_img_shape, case_sensitive, min_txt_len):
        with open(self.annotation_file, encoding='utf-8-sig') as f:
            annotation_file = f.readlines()
        annotation_file = [line.strip() for line in annotation_file]
        delimiter = ', ' if ',' in annotation_file[0] else ' '
        image_names, texts = zip(*[line.split(delimiter) for line in annotation_file])
        pairs = []
        total_len = len(image_names)
        for i, image_nm in tqdm(enumerate(image_names), total=total_len):
            filename = os.path.join(self.data_path, image_nm)
            img = cv.imread(filename, cv.IMREAD_COLOR)
            if grayscale:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            if fixed_img_shape is not None:
                img = cv.resize(img, tuple(fixed_img_shape[::-1]))
            if img.shape[0:2] <= tuple(min_shape):
                continue
            text = texts[i].strip('"')
            if len(text) < min_txt_len:
                continue
            # replace those character not in self.character with ''
            # see here https://github.com/Media-Smart/vedastr/blob/1364526fc770ea99e6d8a8ea1cb972b269983a5f/vedastr/datasets/base.py#L64
            character = "".join(sorted(ALPHANUMERIC_VOCAB, key=lambda x: ord(x)))
            out_of_char = f'[^{character}]'
            text = re.sub(out_of_char, '', text.lower())
            if not case_sensitive:
                text = text.lower()
            text = ' '.join(text)
            el = {'img_name': filename,
                  'text': text,
                  'img': img,
                  }
            pairs.append(el)
        pairs.sort(key=img_size, reverse=True)
        return pairs


class MJSynthDataset(BaseDataset):
    def __init__(self, data_path, annotation_file, min_shape=(8, 8),
                 fixed_img_shape=None, case_sensitive=True, min_txt_len=0, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.ann_file = annotation_file
        self.fixed_img_shape = fixed_img_shape
        self.pairs = self._load(min_shape, case_sensitive, min_txt_len, num_workers)

    def __getitem__(self, index):
        el = deepcopy(self.pairs[index])
        img = cv.imread(os.path.join(self.data_path, el['img_path']), cv.IMREAD_COLOR)
        if self.fixed_img_shape is not None:
            img = cv.resize(img, tuple(self.fixed_img_shape[::-1]))
        el['img'] = img
        return el

    def _load(self, min_shape, case_sensitive, min_txt_len, num_workers):
        pairs = []

        def read_img(image_path):
            gt_text = ' '.join(image_path.split('_')[1])
            if not self.fixed_img_shape:
                img = cv.imread(os.path.join(self.data_path, image_path), cv.IMREAD_COLOR)
                if img is None:
                    return None
                if img.shape[0:2] <= tuple(min_shape):
                    return None
                img_shape = tuple(img.shape)
                del img
            else:
                img_shape = self.fixed_img_shape
            if not case_sensitive:
                gt_text = gt_text.lower()
            if len(gt_text) < min_txt_len:
                return None
            el = {'img_name': os.path.split(image_path)[1],
                  'text': gt_text,
                  'img_path': image_path,
                  'img_shape': img_shape
                  }
            return el

        with open(os.path.join(self.data_path, self.ann_file)) as input_file:
            annotation = [line.split()[0] for line in input_file]
            annotation = [os.path.join(self.data_path, line) for line in annotation]
        pool = ThreadPool(num_workers)

        for elem in tqdm(pool.imap_unordered(read_img, annotation), total=len(annotation)):
            if elem is not None:
                pairs.append(elem)
        if self.fixed_img_shape is None:
            pairs.sort(key=lambda img: img['img_shape'], reverse=True)
        return pairs


class IIIT5KDataset(BaseDataset):
    def __init__(self, data_path, annotation_file, min_shape=(8, 8), grayscale=False,
                 fixed_img_shape=None, case_sensitive=True):
        super().__init__()
        self.data_path = data_path
        self.annotation_file = annotation_file
        self.pairs = self._load(min_shape, fixed_img_shape, grayscale, case_sensitive)

    def _load(self, min_shape, fixed_img_shape, grayscale, case_sensitive):
        pairs = []
        annotation = scipy.io.loadmat(os.path.join(self.data_path, self.annotation_file))
        annotation = (annotation[self.annotation_file.replace('.mat', '')]).squeeze()
        for obj in tqdm(annotation):
            img_path = obj[0][0]
            text = obj[1][0]
            img = cv.imread(os.path.join(self.data_path, img_path), cv.IMREAD_COLOR)
            if grayscale:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            if fixed_img_shape is not None:
                img = cv.resize(img, tuple(fixed_img_shape[::-1]))
            elif img.shape[0:2] <= tuple(min_shape):
                continue
            text = ' '.join(text)
            if not case_sensitive:
                text = text.lower()
            el = {'img_name': img_path,
                  'text': text,
                  'img': img,
                  }
            pairs.append(el)
        pairs.sort(key=img_size, reverse=True)
        return pairs


class LMDBDataset(BaseDataset):
    def __init__(self, data_path, fixed_img_shape=None, case_sensitive=False, grayscale=False):
        super().__init__()
        self.data_path = data_path
        self.fixed_img_shape = fixed_img_shape
        self.case_sensitive = case_sensitive
        self.grayscale = grayscale
        self.database = lmdb.open(bytes(self.data_path, encoding='utf-8'), readonly=True, lock=False)
        self.pairs = self._load()
        self.txn = self.database.begin(write=False)

    def _load(self):
        pairs = []
        with self.database.begin(write=False) as txn:
            num_iterations = int(txn.get('num-samples'.encode()))
            for index in tqdm(range(1, num_iterations + 1)):  # in lmdb indexation starts with one
                text = txn.get(f'label-{index:09d}'.encode()).decode('utf-8')
                if not self.case_sensitive:
                    text = text.lower()
                character = "".join(sorted(ALPHANUMERIC_VOCAB, key=lambda x: ord(x)))
                out_of_char = f'[^{character}]'
                # replace those character not in self.character with ''
                text = re.sub(out_of_char, '', text.lower())
                text = ' '.join(text)
                el = {'img_name': f'image-{index:09d}',
                      'text': text,
                      }
                pairs.append(el)
        return pairs

    def __getitem__(self, index):
        el = deepcopy(self.pairs[index])
        img_key = el['img_name'].encode()
        image_bytes = self.txn.get(img_key)
        img = cv.imdecode(np.frombuffer(image_bytes, np.uint8), cv.IMREAD_UNCHANGED)
        if len(img.shape) < 3:
            img = np.stack((img,) * 3, axis=-1)
        if self.grayscale:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        if self.fixed_img_shape is not None:
            img = cv.resize(img, tuple(self.fixed_img_shape[::-1]))
        el['img'] = img
        return el


class UnrealTextDataset(BaseDataset):
    def __init__(self, data_path, fixed_img_shape=None, case_sensitive=False, grayscale=False, crop_prob=1.0):
        super().__init__()
        self.data_path = data_path
        self.fixed_img_shape = fixed_img_shape
        self.case_sensitive = case_sensitive
        self.grayscale = grayscale
        self.pairs = self._load()
        self.crop_prob = crop_prob
        self.generator = np.random.default_rng(seed=42)

    def _load(self):
        pairs = []
        for folder in tqdm(os.listdir(self.data_path)):
            if 'sub_0' in folder:
                folder_path = os.path.join(self.data_path, folder)
                labels = os.path.join(folder_path, 'labels')
                for label in os.listdir(labels):
                    with open(os.path.join(labels, label)) as label_set:
                        content = json.load(label_set)
                    img_name = content.get('imgfile')
                    for i in range(len(content.get('bbox'))):
                        text = content.get('text')[i]
                        if not set(text.lower()) <= ALPHANUMERIC_VOCAB:
                            continue
                        if not self.case_sensitive:
                            text = text.lower()
                        ymin = min(content.get('bbox')[i][1::2])
                        ymax = max(content.get('bbox')[i][1::2])
                        if any(x < 0 for x in content.get('bbox')[i]) or ymax - ymin < 10:
                            continue
                        text = ' '.join(text)
                        elem = {'img_name': img_name,
                                'img_path': os.path.join(folder_path, img_name),
                                'box': content.get('bbox')[i],
                                'text': text,
                                }
                        pairs.append(elem)
        return pairs

    def __getitem__(self, index):
        el = deepcopy(self.pairs[index])
        img = cv.imread(el['img_path'])
        if len(img.shape) < 3:
            img = np.stack((img,) * 3, axis=-1)
        if self.grayscale:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        box = el['box']
        prob = self.generator.random()
        if prob < self.crop_prob:
            # crop image
            xmin = min(box[::2])
            xmax = max(box[::2])
            ymin = min(box[1::2])
            ymax = max(box[1::2])
            img = img[ymin:ymax, xmin:xmax, :]
        else:
            # perspective transform
            src = np.array([box[0:2], box[2:4], box[4:6], box[6:8]], dtype=np.float32)
            dst = np.array([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]],
                           [0, img.shape[0]]], dtype=np.float32)
            matrix = cv.getPerspectiveTransform(src, dst)
            img = cv.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
        assert img.shape[0] > 0 and img.shape[1] > 0, box
        if self.fixed_img_shape is not None:
            img = cv.resize(img, tuple(self.fixed_img_shape[::-1]))
        el['img'] = img
        return el


class CocoLikeDataset(BaseDataset):
    def __init__(self, data_path, annotation_file, min_shape=(8, 8), grayscale=False,
                 fixed_img_shape=None, case_sensitive=True, rotation_angle=0):
        super().__init__()
        self.data_path = data_path
        self.annotation_file = annotation_file
        self.min_shape = min_shape
        self.grayscale = grayscale
        self.fixed_img_shape = fixed_img_shape
        self.case_sensitive = case_sensitive
        self.pairs = self._load()
        self.rotation_angle = rotation_angle

    def _load(self):
        pairs = []
        with open(self.annotation_file) as ann:
            annotation = json.load(ann)
        annotations = annotation['annotations']
        image_names = annotation['images']
        for ann in tqdm(annotations):
            img_name = image_names[ann['image_id']]['file_name']
            text = ann['attributes']['transcription']
            if not(all(x > 0 for x in ann['bbox']) and set(text.lower()) <= ALPHANUMERIC_VOCAB and ann['attributes']['legible']):
                continue
            if not self.case_sensitive:
                text = text.lower()
            text = ' '.join(text)
            pairs.append(
                {'img_name': os.path.split(img_name)[-1],
                 'img_path': os.path.join(self.data_path, img_name),
                 'bbox': [int(x) for x in ann['bbox']],
                 'text': text,
                 }
            )
        return pairs

    def __getitem__(self, index):
        el = deepcopy(self.pairs[index])
        box = el['bbox']
        img = cv.imread(el['img_path'])
        assert img is not None, el['img_path']
        if len(img.shape) < 3:
            img = np.stack((img,) * 3, axis=-1)
        if self.grayscale:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        x, y, w, h = box

        if self.rotation_angle != 0:
            old_x, old_y = x, y
            angle = np.random.uniform(-self.rotation_angle, self.rotation_angle)
            center = tuple(np.array((x + w / 2, y + h / 2)))
            rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
            img = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)

            xy_vector = -w / 2, -h / 2 # vector from center to TL
            rotated = rotate_point(xy_vector, angle)
            new_x, new_y = int(center[0] + rotated[0]), int(center[1] + rotated[1])
            new_w, new_h = int(center[0] - rotated[0] - x), int(center[1] - rotated[1] - y)
            x,y,w,h = cv.boundingRect(np.array([
                [x, y],
                [x+w, y],
                [x, y+h],
                [x+w, y+h],
                [new_x, new_y],
                [new_x + new_w, new_y],
                [new_x, new_y + new_h],
                [new_x + new_w, new_y + new_h]
            ]))

        img= img[y:y+h, x:x+w, :]
        if self.fixed_img_shape is not None:
            img= cv.resize(img, tuple(self.fixed_img_shape[::-1]))
        el['img']= img
        return el


str_to_class= {
    'Im2LatexDataset': Im2LatexDataset,
    'ICDAR2013RECDataset': ICDAR2013RECDataset,
    'MJSynthDataset': MJSynthDataset,
    'IIIT5KDataset': IIIT5KDataset,
    'LMDBDataset': LMDBDataset,
    'UnrealTextDataset': UnrealTextDataset,
    'CocoLikeDataset': CocoLikeDataset,
}
