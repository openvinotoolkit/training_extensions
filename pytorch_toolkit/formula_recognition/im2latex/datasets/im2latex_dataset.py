from copy import deepcopy
from os.path import join

import cv2 as cv
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, Sampler
from torchvision import transforms

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
        self.indices = np.array(range(len(dataset)))[::self.batch_size][:-1]

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
    def __init__(self, data_dir, split, transform=None):
        """args:
        data_dir: root dir storing the prepoccessed data
        split: train, validate, test or toy
        """
        assert split in ["train", "validate", "test", "toy", "reduced_val"]
        self.data_dir = data_dir
        self.images_dir = join(data_dir, "images_processed")
        self.formulas = self._get_formulas()
        self.transform = transform
        self.pairs = self._get_pairs(split)

    def __getitem__(self, index):

        el = deepcopy(self.pairs[index])
        if self.transform:
            el['img'] = self.transform(el['img'])
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
        map_file = join(self.data_dir, split + "_filter.lst")
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
