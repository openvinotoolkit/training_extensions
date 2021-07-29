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

import os
import subprocess

import torch
import torch.onnx
from text_recognition.data.vocab import read_vocab
from text_recognition.models.model import TextRecognitionModel
from text_recognition.utils.common import (DECODER_INPUTS, DECODER_OUTPUTS,
                                           ENCODER_INPUTS, ENCODER_OUTPUTS)


FEATURES_SHAPE = 1, 20, 175, 512
HIDDEN_SHAPE = 1, 512
CONTEXT_SHAPE = 1, 512
OUTPUT_SHAPE = 1, 256
TGT_SHAPE = 1, 1

OPSET_VERSION = 11
LOG_LEVEL = 'ERROR'


class Exporter:
    def __init__(self, config):
        self.config = config
        self.model_path = config.get('model_path')
        self.vocab = read_vocab(config.get('vocab_path'))
        self.use_ctc = self.config.get('use_ctc')
        self.out_size = len(self.vocab) + 1 if self.use_ctc else len(self.vocab)
        self.model = TextRecognitionModel(config.get('backbone_config'), self.out_size,
                                          config.get('head', {}), config.get('transformation', {}))
        self.model.eval()
        if self.model_path is not None:
            self.model.load_weights(self.model_path)
        self.img_for_export = torch.rand(self.config.get('input_shape_encoder', self.config.get('input_shape')))
        if not self.use_ctc:
            self.encoder = self.model.get_encoder_wrapper(self.model)
            self.encoder.eval()
            self.decoder = self.model.get_decoder_wrapper(self.model)
            self.decoder.eval()

    def export_complete_model(self):
        model_inputs = [self.config.get('model_input_names')]
        model_outputs = self.config.get('model_output_names').split(',')
        print(f"Saving model to {self.config.get('res_model_name')}")
        res_path = os.path.join(os.path.split(self.model_path)[0], self.config.get('res_model_name'))
        torch.onnx.export(self.model, self.img_for_export, res_path,
                          opset_version=11, input_names=model_inputs, output_names=model_outputs,
                          dynamic_axes={model_inputs[0]: {0: 'batch', 1: 'channels', 2: 'height', 3: 'width'},
                                        model_outputs[0]: {0: 'batch', 1: 'max_len', 2: 'vocab_len'},
                                        })

    def export_encoder(self):
        encoder_inputs = self.config.get('encoder_input_names', ENCODER_INPUTS).split(',')
        encoder_outputs = self.config.get('encoder_output_names', ENCODER_OUTPUTS).split(',')
        res_encoder_path = os.path.join(os.path.split(self.model_path)[0], self.config.get('res_encoder_name'))
        torch.onnx.export(self.encoder, self.img_for_export, res_encoder_path,
                          opset_version=OPSET_VERSION,
                          input_names=encoder_inputs,
                          output_names=encoder_outputs,
                          )

    def export_decoder(self):
        decoder_inputs = self.config.get('decoder_input_names', DECODER_INPUTS).split(',')
        decoder_outputs = self.config.get('decoder_output_names', DECODER_OUTPUTS).split(',')
        input_shapes = self.config.get('decoder_input_shapes', [
                                       HIDDEN_SHAPE, CONTEXT_SHAPE, OUTPUT_SHAPE, FEATURES_SHAPE, TGT_SHAPE])
        inputs = [torch.rand(shape) for shape in input_shapes]
        res_decoder_path = os.path.join(os.path.split(self.model_path)[0], self.config.get('res_decoder_name'))
        torch.onnx.export(self.decoder,
                          inputs,
                          res_decoder_path,
                          opset_version=OPSET_VERSION,
                          input_names=decoder_inputs,
                          output_names=decoder_outputs
                          )

    def export_complete_model_ir(self):
        input_model = os.path.join(os.path.split(self.model_path)[0], self.config.get('res_model_name'))
        input_shape = self.config.get('input_shape')
        output_names = self.config.get('model_output_names')
        output_dir = os.path.split(self.model_path)[0]
        export_command = f"""mo \
        --framework onnx \
        --input_model {input_model} \
        --input_shape "{input_shape}" \
        --output "{output_names}" \
        --log_level={LOG_LEVEL} \
        --output_dir {output_dir} \
        --scale_values 'imgs[255]'"""
        if self.config.get('verbose_export'):
            print(export_command)
        subprocess.run(export_command, shell=True, check=True)

    def export_encoder_ir(self):
        input_model = os.path.join(os.path.split(self.model_path)[0], self.config.get('res_encoder_name'))
        input_shape = self.config.get('input_shape_encoder')
        num_channels = input_shape[1]
        scale_values = '[255]' if num_channels == 1 else '[255,255,255]'
        reverse_channels = '' if num_channels == 1 else '--reverse_input_channels'
        input_names = self.config.get("encoder_input_names", ENCODER_INPUTS)
        output_names = self.config.get('encoder_output_names', ENCODER_OUTPUTS)
        output_dir = os.path.split(self.model_path)[0]
        export_command = f"""mo \
        --framework onnx \
        --input_model {input_model} \
        --input_shape "{input_shape}" \
        --output "{output_names}" \
        {reverse_channels} \
        --log_level={LOG_LEVEL} \
        --output_dir {output_dir} \
        --scale_values '{input_names}{scale_values}'"""
        if self.config.get('verbose_export'):
            print(export_command)
        subprocess.run(export_command, shell=True, check=True)

    def export_decoder_ir(self):
        input_shape_decoder = self.config.get('decoder_input_shapes', [
            HIDDEN_SHAPE, CONTEXT_SHAPE, OUTPUT_SHAPE, FEATURES_SHAPE, TGT_SHAPE])
        input_shape_decoder = ', '.join(str(shape) for shape in input_shape_decoder)
        input_model = os.path.join(os.path.split(self.model_path)[0], self.config.get('res_decoder_name'))
        input_names = self.config.get('decoder_input_names', DECODER_INPUTS)
        output_names = self.config.get('decoder_output_names', DECODER_OUTPUTS)
        output_dir = os.path.split(self.model_path)[0]
        export_command = f"""mo \
        --framework onnx \
        --input_model {input_model} \
        --input '{input_names}' \
        --input_shape '{input_shape_decoder}' \
        --log_level={LOG_LEVEL} \
        --output_dir {output_dir} \
        --output '{output_names}'"""
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

        export_function_template = 'export_{}{}'
        if not self.use_ctc:
            assert model_type in ('encoder', 'decoder')

        result_model_exists = os.path.exists(model)
        if ir:
            model_xml = model.replace('.onnx', '.xml')
            model_bin = model.replace('.onnx', '.bin')
            result_model_exists = os.path.exists(model_xml) and os.path.exists(model_bin)
        if not result_model_exists:
            print(f'Model {model} does not exists, exporting it...')
            ir_suffix = '_ir' if ir else ''
            if not self.use_ctc:
                export_function_name = export_function_template.format(model_type, ir_suffix)
            else:
                export_function_name = export_function_template.format('complete_model', ir_suffix)
            getattr(self, export_function_name)()
