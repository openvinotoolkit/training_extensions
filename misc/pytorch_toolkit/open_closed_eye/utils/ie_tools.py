"""
 Copyright (c) 2018-2020 Intel Corporation
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

import sys
import os

import glog as log
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin  # pylint: disable=import-error,E0611


class IEModel:
    """Class for inference of models in the Inference Engine format"""
    def __init__(self, exec_net, inputs_info, input_key, output_key):
        self.net = exec_net
        self.inputs_info = inputs_info
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, img):
        """Performs forward pass of the wrapped IE model"""
        res = self.net.infer(inputs={self.input_key: np.expand_dims(img.transpose(2, 0, 1), axis=0)})
        return np.copy(res[self.output_key])

    def get_input_shape(self):
        """Returns an input shape of the wrapped IE model"""
        return self.inputs_info[self.input_key]


def load_ie_model(model_xml, device, plugin_dir, cpu_extension=''):
    """Loads a model in the Inference Engine format"""
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
    if cpu_extension and 'CPU' in device:
        plugin.add_cpu_extension(cpu_extension)
    # Read IR
    log.info("Loading network files:\n\t%s\n\t%s", model_xml, model_bin)
    net = IENetwork(model=model_xml, weights=model_bin)

    if "CPU" in plugin.device:
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if not_supported_layers:
            log.error("Following layers are not supported by the plugin for specified device %s:\n %s",
                      plugin.device, ', '.join(not_supported_layers))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.inputs.keys()) == 1, "Checker supports only single input topologies"
    assert len(net.outputs) == 1, "Checker supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = 1

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=net)
    model = IEModel(exec_net, net.inputs, input_blob, out_blob)
    del net
    return model
