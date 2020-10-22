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
import math
import os
import subprocess

import cv2 as cv
import numpy as np
import onnxruntime
import torch
import torch.onnx
from openvino.inference_engine import IECore
from torchvision.transforms import ToTensor

from im2latex.data.utils import create_list_of_transforms
from im2latex.data.vocab import END_TOKEN, START_TOKEN, read_vocab
from im2latex.models.im2latex_model import Im2latexModel
from tools.utils.get_config import get_config

ENCODER_INPUTS = "imgs"
ENCODER_OUTPUTS = "row_enc_out,hidden,context,init_0"
DECODER_INPUTS = "dec_st_h,dec_st_c,output_prev,row_enc_out,tgt"
DECODER_OUTPUTS = "dec_st_h_t,dec_st_c_t,output,logit"

OPENVINO_DIR = '/opt/intel/openvino'

FEATURES_SHAPE = 1, 20, 175, 512
HIDDEN_SHAPE = 1, 512
CONTEXT_SHAPE = 1, 512
OUTPUT_SHAPE = 1, 256


def read_net(model_xml, ie):
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    model = ie.read_network(model_xml, model_bin)

    return model


class ONNXExporter:
    def __init__(self, config):
        self.config = config
        self.model_path = config.get('model_path')
        self.vocab = read_vocab(config.get('vocab_path'))
        self.transform = create_list_of_transforms(config.get('transforms_list'))
        self.transform_for_ir = create_list_of_transforms(config.get('transforms_list'), ovino_ir=True)
        self.model = Im2latexModel(config.get('backbone_type', 'resnet'), config.get(
            'backbone_config'), len(self.vocab), config.get('head', {}))
        self.model.eval()
        if self.model_path is not None:
            self.model.load_weights(self.model_path)
        self.img_for_export = torch.rand(self.config.get("input_shape_decoder"))
        img_for_test = cv.imread(self.config.get("test_image"))
        assert img_for_test is not None, f"Check path to the image for test: {self.config.get('test_image')}"
        dst_h, dst_w = self.config.get('input_shape_decoder')[2], self.config.get('input_shape_decoder')[3]
        img_for_test = cv.resize(img_for_test, dsize=(dst_w, dst_h))
        self.img_for_test = np.expand_dims(np.transpose(img_for_test, (2, 0, 1)), axis=0)
        self.img = (ToTensor()(self.img_for_test.squeeze(0))).unsqueeze(0).permute(0, 2, 3, 1)
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

    def run_encoder(self):
        encoder_onnx = onnxruntime.InferenceSession(self.config.get("res_encoder_name"))
        encoder_outputs = get_onnx_outputs(encoder_onnx)
        encoder_input = get_onnx_inputs(encoder_onnx)[0]
        return encoder_onnx.run(encoder_outputs, {
            encoder_input: np.array(self.img, dtype=np.float32)
        })

    def export_decoder(self):
        tgt = np.array([[START_TOKEN]] * self.img.size(0))
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

    def run_decoder(self, hidden, context, output, row_enc_out):
        decoder_onnx = onnxruntime.InferenceSession(self.config.get("res_decoder_name"))
        decoder_inputs = get_onnx_inputs(decoder_onnx)
        decoder_outputs = get_onnx_outputs(decoder_onnx)
        logits = []
        logit = None
        for _ in range(self.model.head.max_len):
            if logit is not None:
                tgt = np.reshape(np.argmax(logit, axis=1), (self.img.size(0), 1)).astype(np.long)
            else:
                tgt = np.array([[START_TOKEN]] * self.img.size(0))
            if tgt[0][0] == END_TOKEN:
                break
            hidden, context, output, logit = decoder_onnx.run(
                decoder_outputs,
                {
                    decoder_inputs[0]: hidden,
                    decoder_inputs[1]: context,
                    decoder_inputs[2]: output,
                    decoder_inputs[3]: row_enc_out,
                    decoder_inputs[4]: tgt
                })
            logits.append(logit)
        return np.argmax(np.array(logits).squeeze(1), axis=1)

    def export_encoder_ir(self):
        export_command = f"""{OPENVINO_DIR}/bin/setupvars.sh && \
        python {OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py \
        --framework onnx \
        --input_model {self.config.get("res_encoder_name")} \
        --input_shape "{self.config.get("input_shape_decoder")}" \
        --output "{self.config.get("encoder_output_names", ENCODER_OUTPUTS)}" \
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
        export_command = f"""{OPENVINO_DIR}/bin/setupvars.sh &&
        python {OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py \
        --framework onnx \
        --input_model {self.config.get('res_decoder_name')} \
        --input {self.config.get("decoder_input_names", DECODER_INPUTS)} \
        --input_shape '{str(input_shape)}' \
        --output {self.config.get("decoder_output_names", DECODER_OUTPUTS)}"""
        if self.config.get('verbose_export'):
            print(export_command)
        subprocess.run(export_command,
                       shell=True, check=True
                       )

    def run_ir_model(self):
        ie = IECore()

        encoder = read_net(self.config.get("res_encoder_name").replace(".onnx", ".xml"), ie)
        dec_step = read_net(self.config.get("res_decoder_name").replace(".onnx", ".xml"), ie)

        exec_net_encoder = ie.load_network(network=encoder, device_name="CPU")
        exec_net_decoder = ie.load_network(network=dec_step, device_name="CPU")
        enc_res = exec_net_encoder.infer(inputs={self.config.get(
            "encoder_input_names", ENCODER_INPUTS).split(",")[0]: self.img_for_test})
        enc_out_names = self.config.get("encoder_output_names", ENCODER_OUTPUTS).split(",")
        ir_row_enc_out = enc_res[enc_out_names[0]]
        dec_states_h = enc_res[enc_out_names[1]]
        dec_states_c = enc_res[enc_out_names[2]]
        output = enc_res[enc_out_names[3]]
        dec_in_names = self.config.get("decoder_input_names", DECODER_INPUTS).split(",")
        dec_out_names = self.config.get("decoder_output_names", DECODER_OUTPUTS).split(",")
        tgt = np.array([[START_TOKEN]] * self.img.size(0))
        logits = []
        for _ in range(self.model.head.max_len):
            dec_res = exec_net_decoder.infer(inputs={
                dec_in_names[0]: dec_states_h,
                dec_in_names[1]: dec_states_c,
                dec_in_names[2]: output,
                dec_in_names[3]: ir_row_enc_out,
                dec_in_names[4]: tgt
            }
            )

            dec_states_h = dec_res[dec_out_names[0]]
            dec_states_c = dec_res[dec_out_names[1]]
            output = dec_res[dec_out_names[2]]
            logit = dec_res[dec_out_names[3]]
            logits.append(logit)

            tgt = np.reshape(np.argmax(logit, axis=1), (self.img.size(0), 1)).astype(np.long)
            if tgt[0][0] == END_TOKEN:
                break

        return np.argmax(np.array(logits).squeeze(1), axis=1)


def get_onnx_inputs(model):
    names = []
    for inp in model.get_inputs():
        names.append(inp.name)
    return names


def get_onnx_outputs(model):
    names = []
    for out in model.get_outputs():
        names.append(out.name)
    return names


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--config')
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_config = get_config(args.config, section='export')
    with torch.no_grad():
        exporter = ONNXExporter(export_config)
        exporter.export_encoder()
        row_enc_out, h, c, O_t = exporter.run_encoder()
        exporter.export_decoder()
        targets_onnx = exporter.run_decoder(h, c, O_t, row_enc_out).astype(np.int32)
        pred_onnx = exporter.vocab.construct_phrase(targets_onnx)
        _, targets_pytorch = exporter.model(exporter.img)
        pred_pytorch = exporter.vocab.construct_phrase(targets_pytorch[0])
        print("Predicted with PyTorch: {}".format(pred_pytorch))
        print("Predicted with ONNX: {}".format(pred_onnx))
        print("Predicted with ONNX is equal to PyTorch: {}".format(pred_onnx == pred_pytorch))
        if export_config.get("export_ir"):
            exporter.export_encoder_ir()
            exporter.export_decoder_ir()
            targets_ir = exporter.run_ir_model()
            ir_pred = exporter.vocab.construct_phrase(targets_ir)
            print("Predicted with OpenvinoIR is equal to PyTorch: {}".format(ir_pred == pred_pytorch))
            print("Predicted with OpenvinoIR: {}".format(ir_pred))
            print("Predicted with PyTorch: {}".format(pred_pytorch))
