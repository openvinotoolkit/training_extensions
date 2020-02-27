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

import logging

import numpy as np

from segmentoly.rcnn.openvino_net import OpenVINONet


class TextMaskRCNNOpenVINO(OpenVINONet):
    def __init__(self, detector_model_path, encoder_model_path, decoder_model_path, *args,
                 **kwargs):
        super(TextMaskRCNNOpenVINO, self).__init__(detector_model_path[:-3] + 'xml',
                                                   detector_model_path[:-3] + 'bin',
                                                   *args, **kwargs)
        required_input_keys = {'im_data', 'im_info'}
        assert required_input_keys == set(self.net.inputs.keys())
        required_output_keys = {'boxes', 'scores', 'classes', 'raw_masks', 'text_features'}
        assert required_output_keys.issubset(self.net.outputs.keys())

        self.n, self.c, self.h, self.w = self.net.inputs['im_data'].shape
        assert self.n == 1, 'Only batch 1 is supported.'

        self.text_encoder = OpenVINONet(encoder_model_path[:-3] + 'xml',
                                        encoder_model_path[:-3] + 'bin')

        self.text_decoder = OpenVINONet(decoder_model_path[:-3] + 'xml',
                                        decoder_model_path[:-3] + 'bin')

    def __call__(self, im_data, im_info, **kwargs):
        logging.warning('Please do not use this code for performance evaluation purposes. '
                        'This is for accuracy validation only. If you would like to see how fast it '
                        'is please refer to https://github.com/opencv/open_model_zoo/blob/develop/'
                        'demos/python_demos/text_spotting_demo/README.md.')

        im_data = im_data[0].cpu().numpy()
        im_info = im_info[0].cpu().numpy()
        if (self.h - im_data.shape[1] < 0) or (self.w - im_data.shape[2] < 0):
            raise ValueError('Input image should resolution of {}x{} or less, '
                             'got {}x{}.'.format(self.w, self.h, im_data.shape[2],
                                                 im_data.shape[1]))
        im_data = np.pad(im_data, ((0, 0),
                                   (0, self.h - im_data.shape[1]),
                                   (0, self.w - im_data.shape[2])),
                         mode='constant', constant_values=0).reshape(1, self.c, self.h, self.w)
        im_info = im_info.reshape(1, *im_info.shape)
        output = super().__call__(dict(im_data=im_data, im_info=im_info))

        classes = output['classes']
        valid_detections_mask = classes > 0
        classes = classes[valid_detections_mask]
        boxes = output['boxes'][valid_detections_mask]
        scores = output['scores'][valid_detections_mask]
        masks = output['raw_masks'][valid_detections_mask]
        text_features = output['text_features'][valid_detections_mask, :, :]

        texts = []
        for feature in text_features:
            feature = self.text_encoder({'input': feature})['output']
            feature = np.reshape(feature, (feature.shape[0], feature.shape[1], -1))
            feature = np.transpose(feature, (0, 2, 1))

            hidden = np.zeros([1, 1, 256])
            prev_symbol = np.zeros((1,))

            eos = 1
            max_seq_len = 28
            vocab_size = 38
            per_feature_outputs = np.zeros([max_seq_len, vocab_size])

            for i in range(max_seq_len):
                o = self.text_decoder(
                    {'prev_symbol': prev_symbol, 'prev_hidden': hidden, 'encoder_outputs': feature})
                output = o['output']
                per_feature_outputs[i] = output
                prev_symbol = np.argmax(output, axis=1)
                if prev_symbol == eos:
                    break
                hidden = o['hidden']

            texts.append(per_feature_outputs)

        texts = np.array(texts)

        return boxes, classes, scores, np.full(len(classes), 0, dtype=np.int32), masks, texts
