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

import math
import os
import subprocess

import numpy as np
import torch
import torch.onnx
from im2latex.data.vocab import START_TOKEN, read_vocab
from im2latex.models.im2latex_model import Im2latexModel
from tools.utils.common import (DECODER_INPUTS, DECODER_OUTPUTS,
                                ENCODER_INPUTS, ENCODER_OUTPUTS)

OPENVINO_DIR = '/opt/intel/openvino'

FEATURES_SHAPE = 1, 20, 175, 512
HIDDEN_SHAPE = 1, 512
CONTEXT_SHAPE = 1, 512
OUTPUT_SHAPE = 1, 256


class Exporter:
    def __init__(self, config):
        self.config = config
        self.model_path = config.get('model_path')
        self.vocab = read_vocab(config.get('vocab_path'))
        self.model = Im2latexModel(config.get('backbone_type', 'resnet'), config.get(
            'backbone_config'), len(self.vocab), config.get('head', {}))
        self.model.eval()
        if self.model_path is not None:
            self.model.load_weights(self.model_path)
        self.img_for_export = torch.rand(self.config.get("input_shape_decoder"))
        self.encoder = self.model.get_encoder_wrapper(self.model)
        self.encoder.eval()
        self.decoder = self.model.get_decoder_wrapper(self.model)
        self.decoder.eval()

    def export_encoder(self):
        encoder_inputs = self.config.get("encoder_input_names", ENCODER_INPUTS).split(',')
        encoder_outputs = self.config.get("encoder_output_names", ENCODER_OUTPUTS).split(',')
        torch.onnx.export(self.encoder, self.img_for_export, self.config.get("res_encoder_name"),
                          opset_version=11,
                          input_names=encoder_inputs,
                          output_names=encoder_outputs,
                          dynamic_axes={encoder_inputs[0]:
                                        {0: 'batch', 1: "channels", 2: "height", 3: "width"},
                                        encoder_outputs[0]: {0: 'batch', 1: 'H', 2: 'W'},
                                        },
                          )

    def export_decoder(self):
        tgt = np.array([[START_TOKEN]] * 1)
        decoder_inputs = self.config.get("decoder_input_names", DECODER_INPUTS).split(',')
        decoder_outputs = self.config.get("decoder_output_names", DECODER_OUTPUTS).split(',')
        row_enc_out = torch.rand(FEATURES_SHAPE)
        hidden = torch.randn(HIDDEN_SHAPE)
        context = torch.rand(CONTEXT_SHAPE)
        output = torch.rand(OUTPUT_SHAPE)

        torch.onnx.export(self.decoder,
                          (hidden,
                           context,
                           output,
                           row_enc_out,
                           torch.tensor(tgt, dtype=torch.long)),
                          self.config.get("res_decoder_name"),
                          opset_version=11,
                          input_names=decoder_inputs,
                          output_names=decoder_outputs,
                          dynamic_axes={decoder_inputs[3]: {  # row_enc_out name should be here
                              0: 'batch', 1: 'H', 2: 'W'}}
                          )

    def export_encoder_ir(self):
        input_model = self.config.get("res_encoder_name")
        input_shape = self.config.get("input_shape_decoder")
        output_names = self.config.get("encoder_output_names", ENCODER_OUTPUTS)
        export_command = f"""{OPENVINO_DIR}/bin/setupvars.sh && \
        python {OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py \
        --framework onnx \
        --input_model {input_model} \
        --input_shape "{input_shape}" \
        --output "{output_names}" \
        --reverse_input_channels \
        --scale_values 'imgs[255,255,255]'"""
        if self.config.get('verbose_export'):
            print(export_command)
        subprocess.run(export_command,
                       shell=True, check=True
                       )

    def export_decoder_ir(self):
        input_shape_decoder = self.config.get("input_shape_decoder")
        output_h, output_w = input_shape_decoder[2] / 32, input_shape_decoder[3] / 32
        if self.config['backbone_config']['disable_layer_4']:
            output_h, output_w = output_h * 2, output_w * 2
        if self.config['backbone_config']['disable_layer_3']:
            output_h, output_w = output_h * 2, output_w * 2
        output_h, output_w = math.ceil(output_h), math.ceil(output_w)
        input_shape = [[1, self.config.get('head', {}).get("decoder_hidden_size", 512)],
                       [1, self.config.get('head', {}).get('decoder_hidden_size', 512)],
                       [1, self.config.get('head', {}).get('encoder_hidden_size', 256)],
                       [1, output_h, output_w, self.config.get('head', {}).get('decoder_hidden_size', 512)], [1, 1]]
        input_shape = "{}, {}, {}, {}, {}".format(*input_shape)
        input_model = self.config.get('res_decoder_name')
        input_names = self.config.get("decoder_input_names", DECODER_INPUTS)
        output_names = self.config.get("decoder_output_names", DECODER_OUTPUTS)
        export_command = f"""{OPENVINO_DIR}/bin/setupvars.sh &&
        python {OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py \
        --framework onnx \
        --input_model {input_model} \
        --input {input_names} \
        --input_shape '{input_shape}' \
        --output {output_names}"""
        if self.config.get('verbose_export'):
            print(export_command)
        subprocess.run(export_command,
                       shell=True, check=True
                       )

    def export_to_onnx_model_if_not_yet(self, model, model_type):
        """Wrapper for _export_model_if_not_yet. Exports model to ONNX

        Args:
            model (str): Path to the model file
            model_type (str): Encoder or decoder
        """
        self._export_model_if_not_yet(model, model_type, ir=False)

    def export_to_ir_model_if_not_yet(self, model, model_type):
        """Wrapper for _export_model_if_not_yet. Exports model to ONNX and to OpenVINO IR

        Args:
            model (str): Path to the model file
            model_type (str): Encoder or decoder
        """
        self._export_model_if_not_yet(model, model_type, ir=False)
        self._export_model_if_not_yet(model, model_type, ir=True)

    def _export_model_if_not_yet(self, model, model_type, ir=False):
        """Checks if given model file exists and if not runs model export

        Args:
            model (str): Path to the model file
            model_type (str): encoder or decoder
            ir (bool, optional): Export to OpenVINO IR. Defaults to False.
        """
        assert model_type in ('encoder', 'decoder')
        result_model_exists = os.path.exists(model)
        if ir:
            model_xml = model.replace(".onnx", '.xml')
            model_bin = model.replace(".onnx", '.bin')
            result_model_exists = os.path.exists(model_xml) and os.path.exists(model_bin)
        if not result_model_exists:
            print(f"Model {model} does not exists, exporting it...")
            ir_suffix = "_ir" if ir else ""
            export_function_name = f"export_{model_type}{ir_suffix}"
            getattr(self, export_function_name)()
