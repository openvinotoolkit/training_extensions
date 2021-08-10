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
import os.path

import cv2 as cv
from text_recognition.data.utils import create_list_of_transforms, ctc_greedy_search
from text_recognition.data.vocab import read_vocab
from text_recognition.models.model import TextRecognitionModel
from text_recognition.utils.evaluation_utils import render_routine, check_environment
from text_recognition.utils.get_config import get_config
import torch


class TextRecognitionDemo:
    def __init__(self, config):
        self.config = config
        self.model_path = config.get('model_path')
        self.vocab = read_vocab(config.get('vocab_path'))
        self.transform = create_list_of_transforms(config.get('transforms_list'))
        self.use_ctc = self.config.get('use_ctc')
        self.model = TextRecognitionModel(config.get('backbone_config'), len(
            self.vocab), config.get('head', {}), config.get('transformation', {}))
        if self.model_path is not None:
            self.model.load_weights(self.model_path, map_location=config.get('map_location', 'cpu'))
        self.model.eval()
        self.device = config.get('device', 'cpu')
        self.model = self.model.to(self.device)

    def __call__(self, img):
        img = self.transform(img)
        img = img[0].unsqueeze(0)
        img = img.to(self.device)
        logits, pred = self.model(img)
        if self.use_ctc:
            pred = torch.nn.functional.log_softmax(logits.detach(), dim=2)
            pred = ctc_greedy_search(pred, 0)

        return self.vocab.construct_phrase(pred[0], ignore_end_token=self.use_ctc)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--config')
    args.add_argument('-i', '--input', help='Path to a folder with images or path to an image files', required=True)
    return args.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    demo_config = get_config(arguments.config, section='demo')
    demo = TextRecognitionDemo(demo_config)
    try:
        check_environment()
    except EnvironmentError:
        print('Warning: cannot render image because some render tools are not installed')
        print('Check that pdflatex, ghostscript and imagemagick are installed')
        print('For details, please, refer to README.md')
        render_images = False
    else:
        render_images = True
    if os.path.isdir(arguments.input):
        inputs = sorted(os.path.join(arguments.input, inp)
                        for inp in os.listdir(arguments.input))
    else:
        inputs = [arguments.input]
    for inp in inputs:
        input_image = cv.imread(inp, cv.IMREAD_COLOR)
        if demo_config.get('backbone_config').get('one_ch_first_conv'):
            input_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
        if demo_config.get('model_resolution'):
            input_image = cv.resize(input_image, demo_config.get('model_resolution')[::-1])
        assert input_image is not None, 'Error reading image {}, please, check input path'.format(inp)
        recognized_formula = demo(input_image)
        cv.imshow('Input image', input_image)
        print(recognized_formula)
        line_for_render = (recognized_formula, 'output.png', './')
        if render_images:
            render_routine(line_for_render)
            rendered_formula = cv.imread('output.png', cv.IMREAD_UNCHANGED)
            cv.imshow('Predicted formula', rendered_formula)
            cv.waitKey(0)
            if os.path.exists('output.png'):
                os.remove('output.png')
