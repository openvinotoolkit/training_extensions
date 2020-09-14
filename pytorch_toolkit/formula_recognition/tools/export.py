import argparse
import yaml
import numpy as np

import cv2 as cv
import torch
import torch.optim as optim
from im2latex.data.utils import create_list_of_transforms
from im2latex.data.vocab import read_vocab
from im2latex.models.im2latex_model import Im2latexModel
import onnxruntime
import torch.onnx
from im2latex.data.vocab import START_TOKEN, END_TOKEN


class ONNXExporter():
    def __init__(self, config):
        self.config = config
        self.model_path = config.get('model_path')
        self.vocab = read_vocab(config.get('vocab_path'))
        self.transform = create_list_of_transforms(config.get('transforms_list'))
        self.model = Im2latexModel(config.get('backbone_type'), config.get(
            'backbone_config'), len(self.vocab), config.get('head'))
        if self.model_path is not None:
            self.model.load_weights(self.model_path)

        self.input = config.get("dummy_input")

        self.encoder = self.model.get_encoder_wrapper(self.model)
        self.decoder = self.model.get_decoder_wrapper(self.model)

    def read_and_preprocess_img(self):
        img = cv.imread(self.input)
        img = self.transform(img)
        img = img[0].unsqueeze(0)
        return img


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--config')
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    exporter = ONNXExporter(config)
    encoder = exporter.encoder
    decoder = exporter.decoder
    img = exporter.read_and_preprocess_img()
    torch.onnx.export(encoder, img, config.get("res_encoder_name"),
                      opset_version=11,
                      input_names=['imgs'],
                      dynamic_axes={'imgs': {0: 'batch', 1: "channels", 2: "height", 3: "width"},
                                    'row_enc_out': {0: 'batch', 1: 'H', 2: 'W'},
                                    'hidden': {1: 'B', 2: "H"},
                                    'context': {1: 'B', 2: "H"},
                                    'init_0': {}
                                    },
                      output_names=['row_enc_out', 'hidden', 'context', 'init_0'])
    sess_pt1 = onnxruntime.InferenceSession(config.get("res_encoder_name"))
    imgs_name = sess_pt1.get_inputs()[0].name
    row_enc_out_name = sess_pt1.get_outputs()[0].name
    h_name = sess_pt1.get_outputs()[1].name
    c_name = sess_pt1.get_outputs()[2].name
    init_0_name = sess_pt1.get_outputs()[3].name

    row_enc_out, h, c, O_t = sess_pt1.run([row_enc_out_name, h_name, c_name, init_0_name],
                                          {imgs_name: np.array(
                                              img).astype(np.float32)},
                                          )[:]
    tgt = torch.tensor([[START_TOKEN]] * img.size(0))
    torch.onnx.export(decoder,
                        (torch.tensor(h),
                        torch.tensor(c),
                        torch.tensor(O_t),
                        torch.tensor(row_enc_out),
                        torch.tensor(tgt, dtype=torch.long)),
                        config.get("res_decoder_name"),
                        opset_version=11,
                        input_names=['dec_st_h', 'dec_st_c',
                                    'output_prev', 'row_enc_out', 'tgt'],
                        output_names=['dec_st_h_t',
                                    'dec_st_c_t', 'output', 'logit'],
                        dynamic_axes={row_enc_out_name: {
                            0: 'batch', 1: 'H', 2: 'W'}}
                        )