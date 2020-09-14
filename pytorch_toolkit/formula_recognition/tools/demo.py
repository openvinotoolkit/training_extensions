import argparse
import yaml
import os.path

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
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    model = Im2latexDemo(config)
    for inp in config.get('input_images'):
        input_image = cv.imread(os.path.abspath(inp), cv.IMREAD_COLOR)
        assert input_image is not None, "Error reading image, please, check input path"
        print("Predicted formula for {} is \n{}".format(os.path.abspath(inp), model(input_image)))
