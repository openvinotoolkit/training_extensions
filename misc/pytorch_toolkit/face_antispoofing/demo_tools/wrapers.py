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

import inspect
import os.path as osp
import sys

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn

from .ie_tools import load_ie_model
current_dir = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = osp.dirname(current_dir)
sys.path.insert(0, parent_dir)
import utils

class FaceDetector:
    """Wrapper class for face detector"""
    def __init__(self, model_path, conf=.6, device='CPU', ext_path=''):
        self.net = load_ie_model(model_path, device, None, ext_path)
        self.confidence = conf
        self.expand_ratio = (1.1, 1.05)

    def get_detections(self, frame):
        """Returns all detections on frame"""
        _, _, h, w = self.net.get_input_shape().shape
        out = self.net.forward(cv.resize(frame, (w, h)))
        detections = self.__decode_detections(out, frame.shape)
        return detections

    def __decode_detections(self, out, frame_shape):
        """Decodes raw SSD output"""
        detections = []

        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > self.confidence:
                left = int(max(detection[3], 0) * frame_shape[1])
                top = int(max(detection[4], 0) * frame_shape[0])
                right = int(max(detection[5], 0) * frame_shape[1])
                bottom = int(max(detection[6], 0) * frame_shape[0])
                if self.expand_ratio != (1., 1.):
                    w = (right - left)
                    h = (bottom - top)
                    dw = w * (self.expand_ratio[0] - 1.) / 2
                    dh = h * (self.expand_ratio[1] - 1.) / 2
                    left = max(int(left - dw), 0)
                    right = int(right + dw)
                    top = max(int(top - dh), 0)
                    bottom = int(bottom + dh)

                detections.append(((left, top, right, bottom), confidence))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)
        return detections

class VectorCNN:
    """Wrapper class for a nework returning a vector"""
    def __init__(self, model_path, device='CPU', switch_rb=False):
        self.net = load_ie_model(model_path, device, None, '', switch_rb=switch_rb)

    def forward(self, batch):
        """Performs forward of the underlying network on a given batch"""
        outputs = [self.net.forward(frame) for frame in batch]
        return outputs

class TorchCNN:
    '''Wrapper for torch model'''
    def __init__(self, model, checkpoint_path, config, device='cpu'):
        self.model = model
        if config.data_parallel.use_parallel:
            self.model = nn.DataParallel(self.model, **config.data_parallel.parallel_params)
        utils.load_checkpoint(checkpoint_path, self.model, map_location=device, strict=True)
        self.config = config

    def preprocessing(self, images):
        ''' making image preprocessing for pytorch pipeline '''
        mean = np.array(object=self.config.img_norm_cfg.mean).reshape((3,1,1))
        std = np.array(object=self.config.img_norm_cfg.std).reshape((3,1,1))
        height, width = list(self.config.resize.values())
        preprocessed_imges = []
        for img in images:
            img = cv.resize(img, (height, width) , interpolation=cv.INTER_CUBIC)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            img = img/255
            img = (img - mean)/std
            preprocessed_imges.append(img)
        return torch.tensor(preprocessed_imges, dtype=torch.float32)

    def forward(self, batch):
        batch = self.preprocessing(batch)
        self.model.eval()
        model1 = (self.model.module
                  if self.config.data_parallel.use_parallel
                  else self.model)
        with torch.no_grad():
            output = model1.forward_to_onnx(batch)
            return output.detach().numpy()
