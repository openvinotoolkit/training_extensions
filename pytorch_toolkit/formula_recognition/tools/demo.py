import argparse
import json
import os.path
from collections import OrderedDict

import cv2 as cv
import torch
import torch.optim as optim
from im2latex.data.utils import create_list_of_transforms
from im2latex.data.vocab import read_vocab
from im2latex.models.im2latex_model import Im2latexModel



class Im2latexDemo():
    def __init__(self, config):
        self.config = config
        self.model_path = config.get('model_path')
        self.vocab = read_vocab(config.get('vocab_path'))
        self.transform = create_list_of_transforms(config.get('transforms_list'))
        self.model = Im2latexModel(config.get('backbone_type'), config.get(
            'backbone_config'), len(self.vocab), config.get('head'))
        if self.model_path is not None:
            self.model.load_weights(self.model_path)

        self.device = config.get('device', 'cpu')
        self.model = self.model.to(self.device)

    def __call__(self, img):
        img = self.transform(img)
        img = img[0].unsqueeze(0)
        img = img.to(self.device)
        _, targets = self.model(img)
        return self.vocab.construct_phrase(targets[0])


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--config')
    args.add_argument('--input')
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f, object_pairs_hook=OrderedDict)
    model = Im2latexDemo(config)
    if not os.path.isdir(args.input):
        args.input = [args.input]
    else:
        args.input = os.listdir(args.input)
    for inp in args.input:
        print(os.path.abspath(inp))
        input_image = cv.imread(os.path.abspath(inp))
        assert input_image is not None, "Error reading image, please, check input path"
        print("Predicted formula for {} is \n{}".format(os.path.abspath(inp), model(input_image)))
