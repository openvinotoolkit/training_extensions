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

import argparse
import yaml
import os.path
from functools import partial

from tqdm import tqdm

import torch
from im2latex.data.utils import collate_fn, create_list_of_transforms, get_timestamp
from im2latex.data.vocab import read_vocab
from im2latex.datasets.im2latex_dataset import Im2LatexDataset
from im2latex.models.im2latex_model import Im2latexModel
from torch.utils.data import DataLoader
from tools.evaluation_tools import Im2latexRenderBasedMetric

spaces = [r'\,', r'\>', r'\;', r'\:', r'\quad', r'\qquad', '~']


def ends_with_space(string):
    """If string end with one of the latex spaces (given the above),
    returns True and index of this space, else False and None

    Args:
        string (str): input string with possible spaces

    Returns:
        Tuple(bool, int) string ends with space, index of the space
    """
    for idx, space in enumerate(spaces):
        if string.endswith(space):
            return True, idx
    return False, None


def postprocess_prediction(pred_phrase_str):
    """Deletes usual space in the end of the string and then checks
    if string ends with latex space. If yes, deletes latex space.
    Deletion of spaces is performed because, even though spaces in the end are invisible,
    they affect on rendering the formula, making it more tight to the left

    Args:
        pred_phrase_str (str): input string

    Returns:
        str: postprocessed string
    """
    pred_phrase_str = pred_phrase_str.rstrip()
    ends, idx = ends_with_space(pred_phrase_str)
    while ends:
        pred_phrase_str = pred_phrase_str[:len(pred_phrase_str) - len(spaces[idx])]
        pred_phrase_str = pred_phrase_str.rstrip()
        ends, idx = ends_with_space(pred_phrase_str)
    return pred_phrase_str


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.model_path = config.get('model_path')
        self.val_path = config.get('val_path')
        self.vocab = read_vocab(config.get('vocab_path'))
        self.val_transforms_list = config.get('val_transforms_list')
        self.split = config.get('split_file', 'validate')
        self.print_freq = config.get('print_freq', 16)
        self.load_dataset()
        self.model = Im2latexModel(config.get('backbone_type', 'resnet'), config.get(
            'backbone_config'), len(self.vocab), config.get('head', {}))
        self.device = config.get('device', 'cpu')
        if self.model_path is not None:
            self.model.load_weights(self.model_path, map_location=self.device)

        self.model = self.model.to(self.device)
        self.time = get_timestamp()

    def load_dataset(self):

        val_dataset = Im2LatexDataset(self.val_path, self.split)
        batch_transform_val = create_list_of_transforms(self.val_transforms_list)
        self.val_loader = DataLoader(
            val_dataset,
            collate_fn=partial(collate_fn, self.vocab.sign2id,
                               batch_transform=batch_transform_val),
            num_workers=os.cpu_count())

    def validate(self):
        self.model.eval()
        print("Validation started")
        annotations = []
        predictions = []
        metric = Im2latexRenderBasedMetric()
        with torch.no_grad():
            for img_name, imgs, training_gt, loss_computation_gt in tqdm(self.val_loader):
                imgs = imgs.to(self.device)
                training_gt = training_gt.to(self.device)
                loss_computation_gt = loss_computation_gt.to(self.device)
                _, pred = self.model(imgs)
                gold_phrase_str = self.vocab.construct_phrase(loss_computation_gt[0])
                pred_phrase_str = self.vocab.construct_phrase(pred[0])
                pred_phrase_str = postprocess_prediction(pred_phrase_str)
                annotations.append((gold_phrase_str, img_name[0]))
                predictions.append((pred_phrase_str, img_name[0]))
        res = metric.evaluate(annotations, predictions)
        return res


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--config')
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        test_config = config.get("eval")
        common_config = config.get("common")
        test_config.update(common_config)
    validator = Evaluator(test_config)
    result = validator.validate()
    print("Im2latex metric is: {}".format(result))
