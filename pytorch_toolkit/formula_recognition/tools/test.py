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
from im2latex.datasets.im2latex_dataset import (BatchRandomSampler,
                                                Im2LatexDataset)
from im2latex.models.im2latex_model import Im2latexModel
from torch.utils.data import DataLoader
from tools.evaluation_tools import Im2latexRenderBasedMetric


class Evaluator:
    def __init__(self, config):
        self.config = config
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(root_dir, config.get('model_path'))
        self.val_path = os.path.join(root_dir, config.get('val_path'))
        self.vocab = read_vocab(os.path.join(root_dir, config.get('vocab_path')))
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
        val_sampler = BatchRandomSampler(dataset=val_dataset, batch_size=1)
        batch_transform_val = create_list_of_transforms(self.val_transforms_list)
        self.val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=partial(collate_fn, self.vocab.sign2id,
                               batch_transform=batch_transform_val),
            num_workers=os.cpu_count())

    def validate(self):
        self.model.eval()
        print("Validation started")
        annotations = []
        predictions = []
        metric = Im2latexRenderBasedMetric()
        formula_acc = 0
        with open("debug_metric.txt", 'w') as output_file:

            with torch.no_grad():
                for img_name, imgs, training_gt, loss_computation_gt in tqdm(self.val_loader, initial=1):
                    imgs = imgs.to(self.device)
                    training_gt = training_gt.to(self.device)
                    loss_computation_gt = loss_computation_gt.to(self.device)
                    _, pred = self.model(imgs)
                    gold_phrase_str = self.vocab.construct_phrase(loss_computation_gt[0])
                    pred_phrase_str = self.vocab.construct_phrase(pred[0], max_len=1 + len(gold_phrase_str.split()))

                    annotations.append((gold_phrase_str, img_name[0]))
                    predictions.append((pred_phrase_str, img_name[0]))
                    output_file.write(img_name[0] + '\t' +
                                      gold_phrase_str + '\t' +
                                      pred_phrase_str + '\t' + '10' + '\t' + '10' + '\n')
                    if pred_phrase_str == gold_phrase_str:
                        formula_acc += 1
        avg_formula_acc = formula_acc / len(self.val_loader)
        print("Average formula acc", avg_formula_acc)
        res = metric.evaluate(annotations, predictions)
        return res


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--config')
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader).get("eval")
    validator = Evaluator(config)
    result = validator.validate()
    print("Im2latex metric is: {}".format(result))
