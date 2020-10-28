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

from copy import deepcopy
from os.path import join

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from ..data.utils import get_num_lines_in_file
from ..data.vocab import split_number

COLOR_WHITE = (255, 255, 255)
NUMBER_SIGNS = set("0123456789.")


class BatchRandomSampler(Sampler):
    """This is a class representing random batch sampler
    Only the indices of the dataset are shuffled so if dataset is sorted
    (e.g. on the shape) the original order is saved
    """

    def __init__(self, batch_size, dataset):

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


class Im2LatexDataset(Dataset):
    def __init__(self, data_dir, split):
        """args:
        data_dir: root dir storing the prepoccessed data
        split: train, validate, test or toy
        """
        self.data_dir = data_dir
        self.images_dir = join(data_dir, "images_processed")
        self.formulas = self._get_formulas()
        self.pairs = self._get_pairs(split)

    def __getitem__(self, index):

        el = deepcopy(self.pairs[index])
        return el

    def __len__(self):
        return len(self.pairs)

    def _get_formulas(self):
        formulas_file = join(self.data_dir, "formulas.norm.lst")
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

    def _get_pairs(self, split):
        # the line in this file map image to formulas
        map_file = join(self.data_dir, split)
        total_lines = get_num_lines_in_file(map_file)
        # get image-formulas pairs
        pairs = []
        with open(map_file, 'r') as f:
            for line in tqdm(f, total=total_lines):
                img_name, formula_id = line.strip('\n').split()
                # load img and its corresponding formula
                img_path = join(self.images_dir, img_name)
                img = cv.imread(img_path, cv.IMREAD_COLOR)
                formula = self.formulas[int(formula_id)]
                el = {"img_name": img_name,
                      "formula": formula,
                      "img": img,
                      }
                pairs.append(el)
        pairs.sort(key=img_size, reverse=True)
        return pairs


def img_size(pair):
    img = pair.get("img")
    return tuple(img.shape)
