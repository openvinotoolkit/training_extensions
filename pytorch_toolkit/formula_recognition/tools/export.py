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
import subprocess
import os
import math

import cv2 as cv
import numpy as np
import onnxruntime
import torch
import torch.onnx
import torch.optim as optim
import yaml

from openvino.inference_engine import IECore

from im2latex.data.utils import create_list_of_transforms
from im2latex.data.vocab import START_TOKEN, END_TOKEN, read_vocab
from im2latex.models.im2latex_model import Im2latexModel


def read_net(model_xml, ie, device):
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    model = ie.read_network(model_xml, model_bin)

    return model


class ONNXExporter():
    def __init__(self, config):
        self.config = config
        self.model_path = os.path.join(os.path.abspath("./"), config.get('model_path'))
        self.vocab = read_vocab(os.path.join(os.path.abspath("./"), config.get('vocab_path')))
        self.transform = create_list_of_transforms(config.get('transforms_list'))
        self.transform_for_ir = create_list_of_transforms(config.get('transforms_list'), ovino_ir=True)
        self.model = Im2latexModel(config.get('backbone_type'), config.get(
            'backbone_config'), len(self.vocab), config.get('head', {}))
        if self.model_path is not None:
            self.model.load_weights(self.model_path, old_model=config.get("old_model"))

        self.input = os.path.join(os.path.abspath("./"), config.get("dummy_input"))
        self.img = self.read_and_preprocess_img(self.transform)
        self.img_for_ir = self.read_and_preprocess_img_for_ir(self.transform_for_ir)
        self.encoder = self.model.get_encoder_wrapper(self.model)
        self.decoder = self.model.get_decoder_wrapper(self.model)

    def read_and_preprocess_img(self, transform):
        img = cv.imread(self.input)
        img = transform(img)
        img = img[0].unsqueeze(0)
        return img


    def read_and_preprocess_img_for_ir(self, transform):
        img = cv.imread(self.input)
        img = transform(img)
        img = np.expand_dims(img[0], axis=0)
        return np.transpose(img, (0, 3, 1, 2))

    def export_encoder(self):
        torch.onnx.export(self.encoder, self.img, self.config.get("res_encoder_name"),
                          opset_version=11,
                          input_names=self.config.get("encoder_input_names").split(','),
                          dynamic_axes={self.config.get("encoder_input_names").split(',')[0]:
                                        {0: 'batch', 1: "channels", 2: "height", 3: "width"},
                                        self.config.get("encoder_output_names").split(',')[0]: {0: 'batch', 1: 'H', 2: 'W'},
                                        self.config.get("encoder_output_names").split(',')[1]: {1: 'B', 2: "H"},
                                        self.config.get("encoder_output_names").split(',')[2]: {1: 'B', 2: "H"}
                                        },
                          output_names=self.config.get("encoder_output_names").split(','))

    def run_encoder(self):
        encoder_onnx = onnxruntime.InferenceSession(self.config.get("res_encoder_name"))
        encoder_outputs = get_onnx_outputs(encoder_onnx)
        encoder_input = get_onnx_inputs(encoder_onnx)[0]
        return encoder_onnx.run(encoder_outputs, {
            encoder_input: np.array(self.img).astype(np.float32)
        })

    def export_decoder(self, row_enc_out, h, c, output):
        tgt = np.array([[START_TOKEN]] * self.img.size(0))
        torch.onnx.export(self.decoder,
                          (torch.tensor(h),
                           torch.tensor(c),
                           torch.tensor(output),
                           torch.tensor(row_enc_out),
                           torch.tensor(tgt, dtype=torch.long)),
                          self.config.get("res_decoder_name"),
                          opset_version=11,
                          input_names=self.config.get("decoder_input_names").split(','),
                          output_names=self.config.get("decoder_output_names").split(','),
                          dynamic_axes={self.config.get("decoder_input_names").split(',')[0]: {  # row_enc_out name should be here
                              0: 'batch', 1: 'H', 2: 'W'}}
                          )

    def run_decoder(self, hidden, context, output, row_enc_out):
        decoder_onnx = onnxruntime.InferenceSession(self.config.get("res_decoder_name"))
        decoder_inputs = get_onnx_inputs(decoder_onnx)
        decoder_outputs = get_onnx_outputs(decoder_onnx)
        tgt = np.array([[START_TOKEN]] * self.img.size(0))
        dec_states_out_h, dec_states_out_c, O_t_out_val, logit_val = decoder_onnx.run(
            decoder_outputs,
            {
                decoder_inputs[0]: hidden,
                decoder_inputs[1]: context,
                decoder_inputs[2]: output,
                decoder_inputs[3]: row_enc_out,
                decoder_inputs[4]: tgt
            })
        logits = []
        logits.append(logit_val)
        for _ in range(self.model.head.max_len):
            tgt = np.reshape(np.argmax(logit_val, axis=1), (self.img.size(0), 1)).astype(np.long)
            if tgt[0][0] == END_TOKEN:
                break
            dec_states_out_h, dec_states_out_c, O_t_out_val, logit_val = decoder_onnx.run(
                decoder_outputs,
                {
                    decoder_inputs[0]: dec_states_out_h,
                    decoder_inputs[1]: dec_states_out_c,
                    decoder_inputs[2]: O_t_out_val,
                    decoder_inputs[3]: row_enc_out,
                    decoder_inputs[4]: tgt
                })
            logits.append(logit_val)
        return np.argmax(np.array(logits).squeeze(1), axis=1)

    def export_encoder_ir(self):
        # --output 'row_enc_out,hidden,context,init_0' --reverse_input_channels  --scale_values "imgs[255,255,255]"
        if self.config.get('verbose_export'):
            print(f'/opt/intel/openvino/bin/setupvars.sh && '
                  f'python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py '
                  f'--framework onnx '
                  f'--input_model {self.config.get("res_encoder_name")} '
                  f'--input_shape "{self.config.get("input_shape_decoder")}" '
                  f'--output "{self.config.get("encoder_output_names")}" '
                  f'--reverse_input_channels '
                  f'--scale_values "imgs[255,255,255]"')
        subprocess.run(f'/opt/intel/openvino/bin/setupvars.sh && '
                       f'python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py '
                       f'--framework onnx '
                       f'--input_model {self.config.get("res_encoder_name")} '
                       f'--input_shape "{self.config.get("input_shape_decoder")}" '
                       f'--output {self.config.get("encoder_output_names")} '
                       f'--reverse_input_channels '
                       f'--scale_values "imgs[255,255,255]"',
                       shell=True, check=True
                       )

    def export_decoder_ir(self):
        input_shape_decoder = self.config.get("input_shape_decoder")
        #TODO: resolve output H& W
        print('*'*20)
        print(input_shape_decoder[2] / 8)
        print(input_shape_decoder[3] / 8 )
        print('*'*20)
        output_h, output_w = math.ceil(input_shape_decoder[2] / 8), math.ceil(input_shape_decoder[3] / 8)
        input_shape = [[1, self.config.get('head', {}).get("decoder_hidden_size", 512)],
                       [1, self.config.get('head', {}).get('decoder_hidden_size', 512)],
                       [1, self.config.get('head', {}).get('encoder_hidden_size', 256)],
                       [1, output_h, output_w, self.config.get('head', {}).get('decoder_hidden_size', 512)], [1, 1]]
        input_shape = "{}, {}, {}, {}, {}".format(*input_shape)
        if self.config.get('verbose_export'):
            print(f'/opt/intel/openvino/bin/setupvars.sh && '
                  f'python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py '
                  f'--framework onnx '
                  f'--input_model {self.config.get("res_decoder_name")} '
                  f'--input "{self.config.get("decoder_input_names")}" '
                  f'--input_shape "{str(input_shape)}" '
                  f'--output "{self.config.get("decoder_output_names")}"')
        subprocess.run(f'/opt/intel/openvino/bin/setupvars.sh && '
                       f'python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py '
                       f'--framework onnx '
                       f'--input_model {self.config.get("res_decoder_name")} '
                       f'--input "{self.config.get("decoder_input_names")}" '
                       f'--input_shape "{str(input_shape)}" '
                       f'--output "{self.config.get("decoder_output_names")}"',
                       shell=True, check=True
                       )

    def run_ir_model(self):
        ie = IECore()

        encoder = read_net(self.config.get("res_encoder_name").replace(".onnx", ".xml"), ie, 'cpu')
        dec_step = read_net(self.config.get("res_decoder_name").replace(".onnx", ".xml"), ie, 'cpu')

        exec_net_encoder = ie.load_network(network=encoder, device_name="CPU")
        exec_net_decoder = ie.load_network(network=dec_step, device_name="CPU")
        enc_res = exec_net_encoder.infer(inputs={self.config.get("encoder_input_names").split(",")[0]: self.img_for_ir})
        enc_out_names = self.config.get("encoder_output_names").split(",")
        row_enc_out = enc_res[enc_out_names[0]]
        dec_states_h = enc_res[enc_out_names[1]]
        dec_states_c = enc_res[enc_out_names[2]]
        output = enc_res[enc_out_names[3]]
        dec_in_names = self.config.get("decoder_input_names").split(",")
        dec_out_names = self.config.get("decoder_output_names").split(",")
        tgt = np.array([[START_TOKEN]] * self.img.size(0))
        logits = []
        for _ in range(self.model.head.max_len):
            dec_res = exec_net_decoder.infer(inputs={
                dec_in_names[0]: dec_states_h,
                dec_in_names[1]: dec_states_c,
                dec_in_names[2]: output,
                dec_in_names[3]: row_enc_out,
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
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    exporter = ONNXExporter(config)
    exporter.export_encoder()
    row_enc_out, h, c, O_t = exporter.run_encoder()
    exporter.export_decoder(row_enc_out, h, c, O_t)
    targets_onnx = exporter.run_decoder(h, c, O_t, row_enc_out).astype(np.int32)
    pred_onnx = exporter.vocab.construct_phrase(targets_onnx)
    _, targets_pytorch = exporter.model(exporter.img)
    pred_pytorch = exporter.vocab.construct_phrase(targets_pytorch[0])
    print("Predicted with ONNX is equal to PyTorch: {}".format(pred_onnx == pred_pytorch))
    if config.get("export_ir"):
        exporter.export_encoder_ir()
        exporter.export_decoder_ir()
        targets_ir = exporter.run_ir_model()
        ir_pred = exporter.vocab.construct_phrase(targets_ir)
        print("Predicted with OpenvinoIR is equal to PyTorch: {}".format(ir_pred == pred_pytorch))
