import argparse

import cv2 as cv
import numpy as np
import onnxruntime
import torch
import torch.onnx
import torch.optim as optim
import yaml

from im2latex.data.utils import create_list_of_transforms
from im2latex.data.vocab import START_TOKEN, END_TOKEN, read_vocab
from im2latex.models.im2latex_model import Im2latexModel


class ONNXExporter():
    def __init__(self, config):
        self.config = config
        self.model_path = config.get('model_path')
        self.vocab = read_vocab(config.get('vocab_path'))
        self.transform = create_list_of_transforms(config.get('transforms_list'))
        self.model = Im2latexModel(config.get('backbone_type'), config.get(
            'backbone_config'), len(self.vocab), config.get('head'))
        if self.model_path is not None:
            self.model.load_weights(self.model_path, old_model=config.get("old_model"))

        self.input = config.get("dummy_input")
        self.img = self.read_and_preprocess_img()
        self.encoder = self.model.get_encoder_wrapper(self.model)
        self.decoder = self.model.get_decoder_wrapper(self.model)

    def read_and_preprocess_img(self):
        img = cv.imread(self.input)
        img = self.transform(img)
        img = img[0].unsqueeze(0)
        return img

    def export_encoder(self):
        self.img = self.read_and_preprocess_img()
        torch.onnx.export(self.encoder, self.img, self.config.get("res_encoder_name"),
                          opset_version=11,
                          input_names=['imgs'],
                          dynamic_axes={'imgs': {0: 'batch', 1: "channels", 2: "height", 3: "width"},
                                        'row_enc_out': {0: 'batch', 1: 'H', 2: 'W'},
                                        'hidden': {1: 'B', 2: "H"},
                                        'context': {1: 'B', 2: "H"},
                                        'init_0': {}
                                        },
                          output_names=['row_enc_out', 'hidden', 'context', 'init_0'])
        return True

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
                          input_names=['dec_st_h', 'dec_st_c',
                                       'output_prev', 'row_enc_out', 'tgt'],
                          output_names=['dec_st_h_t',
                                        'dec_st_c_t', 'output', 'logit'],
                          dynamic_axes={'row_enc_out': {
                              0: 'batch', 1: 'H', 2: 'W'}}
                          )
        return True

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
    pred_onnx = exporter.vocab.construct_phrase(exporter.run_decoder(h, c, O_t, row_enc_out).astype(np.int32))
    _, targets = exporter.model(exporter.img)
    pred_pytorch = exporter.vocab.construct_phrase(targets[0])
    print("Predicted with ONNX equal to PyTorch: {}".format(pred_onnx == pred_pytorch))
