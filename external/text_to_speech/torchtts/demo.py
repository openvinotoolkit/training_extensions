# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import logging as log
import os.path as osp
from argparse import ArgumentParser, SUPPRESS
from time import perf_counter

import numpy as np
from openvino.runtime import PartialShape, Core

try:
    from text_preprocessing import text_to_sequence, cmudict, intersperse
except:
    from .text_preprocessing import text_to_sequence, cmudict, intersperse


def check_input_name(model, input_tensor_name):
    try:
        model.input(input_tensor_name)
        return True
    except RuntimeError:
        return False


class IEModel:
    def __init__(self, ie, device):
        self.ie = ie
        self.device = device

    def load_network(self, model_xml):
        model_bin_name = ".".join(osp.basename(model_xml).split('.')[:-1]) + ".bin"
        model_bin = osp.join(osp.dirname(model_xml), model_bin_name)
        log.info('Reading AcousticGAN model {}'.format(model_xml))
        model = self.ie.read_model(model=model_xml, weights=model_bin)
        return model

    def create_infer_request(self, model, path=None):
        compiled_model = self.ie.compile_model(model, device_name=self.device)
        if path is not None:
            log.info('The AcousticGAN model {} is loaded to {}'.format(path, self.device))
        return compiled_model.create_infer_request()


class Encoder(IEModel):
    def __init__(self, model_encoder, ie, device='CPU'):
        super().__init__(ie, device)
        self.encoder = self.load_network(model_encoder)
        self.request = self.create_infer_request(self.encoder, model_encoder)
        self.enc_input_data_name = "seq"
        self.enc_input_mask_name = "seq_len"

    def preprocess(self, seq, seq_len=None):
        if len(seq.shape) == 1:
            seq = np.array(seq)[None, :]
        if seq_len is None:
            seq_len = np.array([seq.shape[1]])

        model_shape = self.encoder.input(self.enc_input_data_name).shape[1]
        if model_shape != seq.shape[1]:
            new_shape = {self.enc_input_data_name: PartialShape(seq.shape), self.enc_input_mask_name: PartialShape([1])}
            self.encoder.reshape(new_shape)
            self.request = self.create_infer_request(self.encoder)

        return {self.enc_input_data_name: seq, self.enc_input_mask_name: seq_len}

    def infer(self, data):
        self.request.infer(data)


class Decoder(IEModel):
    def __init__(self, model_decoder, ie, device='CPU'):
        super().__init__(ie, device)
        self.decoder = self.load_network(model_decoder)
        self.request = self.create_infer_request(self.decoder, model_decoder)

        self.dec_input_data_name = "z"
        self.dec_input_mask_name = "z_mask"

    def preprocess(self, z, z_mask):
        model_shape = list(self.decoder.input(self.dec_input_data_name).shape)
        if model_shape[-1] != z.shape[-1]:
            self.decoder.reshape({self.dec_input_data_name: PartialShape(z.shape),
                                  self.dec_input_mask_name: PartialShape(z_mask.shape)})
            self.request = self.create_infer_request(self.decoder)

        return {self.dec_input_data_name: z, self.dec_input_mask_name: z_mask}

    def infer(self, data):
        self.request.infer(data)


class AcousticGANIE:
    def __init__(self, model_encoder, model_decoder, ie=None, device='CPU', verbose=False):
        self.verbose = verbose
        self.device = device

        if ie is None:
            ie = Core()
        self.ie = ie

        self.cmudict = cmudict.CMUDict(osp.join(osp.dirname(osp.realpath(__file__)), 'text_preprocessing/cmu_dictionary'))
        self.encoder = Encoder(model_encoder, ie, device)
        self.decoder = Decoder(model_decoder, ie, device)

    def seq_to_indexes(self, text):
        res = text_to_sequence(text, dictionary=self.cmudict)
        if self.verbose:
            log.debug(res)
        return res

    @staticmethod
    def sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = np.arange(max_length, dtype=length.dtype)
        return np.expand_dims(x, 0) < np.expand_dims(length, 1)

    @staticmethod
    def generate_path(duration, mask):
        """
        duration: [b, t_text]
        mask: [b, t_text, t_mel]
        """

        b, t_x, t_y = mask.shape  # batch size, text size, mel size
        cum_duration = np.cumsum(duration, 1)

        cum_duration_flat = cum_duration.flatten()  # view(b * t_x)
        path = AcousticGANIE.sequence_mask(cum_duration_flat, t_y).astype(mask.dtype)

        path = path.reshape(b, t_x, t_y)
        path = path - np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]

        path = path * mask

        return path

    def gen_decoder_in(self, alpha=1.0, offset=0.3):
        x_mask = self.encoder.request.get_tensor("x_mask").data[:]
        x_res = self.encoder.request.get_tensor("x_res").data[:]
        logw = self.encoder.request.get_tensor("log_dur").data[:]

        w = (np.exp(logw) + offset) * x_mask
        w_ceil = np.ceil(w) * alpha

        mel_lengths = np.clip(np.sum(w_ceil, axis=(1, 2)), a_min=1, a_max=None).astype(dtype=np.long)

        z_mask = np.expand_dims(self.sequence_mask(mel_lengths), 1).astype(x_mask.dtype)
        attn_mask = np.expand_dims(x_mask, -1) * np.expand_dims(z_mask, 2)

        attn = self.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1))
        attn = np.expand_dims(attn, 1)
        attn = attn.squeeze(1).transpose(0, 2, 1)
        z = np.matmul(attn, x_res.transpose(0, 2, 1)).transpose(0, 2, 1)  # [b, t', t], [b, t, d] -> [b, d, t']

        return z, z_mask

    def forward(self, text, alpha=1.0, **kwargs):
        seq = self.seq_to_indexes(text)
        seq = np.array(seq)

        encoder_in = self.encoder.preprocess(seq)
        self.encoder.infer(encoder_in)
        decoder_in = self.decoder.preprocess(*self.gen_decoder_in(alpha))
        self.decoder.request.infer(decoder_in)

        res = self.decoder.request.get_tensor("mel").data[:]
        res = res * 6.0 - 6.0
        return res


def main(args):
    """
    Main function that is used to run demo.
    """
    log.info('OpenVINO Inference Engine')

    model_encoder = osp.join(args.model, 'encoder.xml')
    model_decoder = osp.join(args.model, 'decoder.xml')
    model = AcousticGANIE(model_encoder, model_decoder, device=args.device)

    start_time = perf_counter()
    mel_spectrogram = model.forward(args.input)
    latency = (perf_counter() - start_time) * 1e3
    log.info("Metrics report:")
    log.info("\tLatency: {:.1f} ms".format(latency))

    try:
        from matplotlib import pyplot as plt
        if len(mel_spectrogram.shape) == 3:
            mel_spectrogram = mel_spectrogram[0, :, :]
        fig, (ax1) = plt.subplots(1)
        ax1.imshow(mel_spectrogram)
        plt.show()
    except ModuleNotFoundError:
        print("module 'matplotlib' for demo is not installed")



if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    parser.add_argument('-m', '--model', help='Optional. Path to an .xml file with a trained model.', default=None)
    parser.add_argument('-i', '--input', help="Input text", default="Hello, my name is demo script. "
                                                                    "I am ready for converting to audio.")
    parser.add_argument('-d', '--device', default='CPU',
                        help="Optional. Specify the target device to infer on, for example: "
                             "CPU, GPU, HDDL, MYRIAD or HETERO. "
                             "The demo will look for a suitable IE plugin for this device. Default value is CPU.")
    args = parser.parse_args()
    main(args)