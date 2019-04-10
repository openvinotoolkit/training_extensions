"""
 Copyright (c) 2019 Intel Corporation

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

from ..utils.blob import to_numpy
from ..utils.profile import PerformanceCounters


class OpenVINONet(object):

    def __init__(self, xml_file_path, bin_file_path, device='CPU',
                 plugin_dir=None, cpu_extension_lib_path=None, collect_perf_counters=False):

        from openvino.inference_engine import IENetwork, IEPlugin

        logging.info('Creating {} plugin...'.format(device))
        self.plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
        if cpu_extension_lib_path and 'CPU' in device:
            logging.info('Adding CPU extensions...')
            self.plugin.add_cpu_extension(cpu_extension_lib_path)

        # Read IR
        logging.info('Reading network from IR...')
        self.net = IENetwork(model=xml_file_path, weights=bin_file_path)

        if self.plugin.device == 'CPU':
            logging.info('Check that all layers are supported...')
            supported_layers = self.plugin.get_supported_layers(self.net)
            not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                unsupported_info = '\n\t'.join('{} ({} with params {})'.format(layer_id,
                                                                               self.net.layers[layer_id].type,
                                                                               str(self.net.layers[layer_id].params))
                                               for layer_id in not_supported_layers)
                logging.warning('Following layers are not supported '
                                'by the plugin for specified device {}:'
                                '\n\t{}'.format(self.plugin.device, unsupported_info))
                logging.warning('Please try to specify cpu extensions library path.')
                raise ValueError('Some of the layers are not supported.')

        logging.info('Loading network to plugin...')
        self.exec_net = self.plugin.load(network=self.net, num_requests=1)

        self.perf_counters = None
        if collect_perf_counters:
            self.perf_counters = PerformanceCounters()

    def __call__(self, inputs):
        outputs = self.exec_net.infer(inputs)
        if self.perf_counters:
            perf_counters = self.exec_net.requests[0].get_perf_counts()
            self.perf_counters.update(perf_counters)
        return outputs

    def print_performance_counters(self):
        if self.perf_counters:
            self.perf_counters.print()

    def __del__(self):
        del self.net
        del self.exec_net
        del self.plugin
            
            
class MaskRCNNOpenVINO(OpenVINONet):
    def __init__(self, *args, **kwargs):
        super(MaskRCNNOpenVINO, self).__init__(*args, **kwargs)
        required_input_keys = {'im_data', 'im_info'}
        assert required_input_keys == set(self.net.inputs.keys())
        required_output_keys = {'boxes', 'scores', 'classes', 'raw_masks'}
        assert required_output_keys.issubset(self.net.outputs.keys())

        self.n, self.c, self.h, self.w = self.net.inputs['im_data'].shape
        assert self.n == 1, 'Only batch 1 is supported.'

    def __call__(self, im_data, im_info):
        im_data = to_numpy(im_data[0])
        im_info = to_numpy(im_info[0])
        if (self.h - im_data.shape[1] < 0) or (self.w - im_data.shape[2] < 0):
            raise ValueError('Input image should resolution of {}x{} or less, '
                             'got {}x{}.'.format(self.w, self.h, im_data.shape[2], im_data.shape[1]))
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
        return boxes, classes, scores, np.full(len(classes), 0, dtype=np.int32), masks
