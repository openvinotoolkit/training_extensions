#!/usr/bin/env python3
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from argparse import ArgumentParser, SUPPRESS
import logging as log
from time import perf_counter
import typing
import sys
import numpy as np
import wave
from openvino.runtime import get_version
from .utils import create_model

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def load_audio_wav(audio_path: str) -> typing.Tuple[np.array, int]:
    """
    Load audio file in .wav format

    Arguments:
        audio_path (str): Path to audio file in .wav format.

    Returns:
        audio (np.array): Waveform.
        sampling_rate (int): Sampling rate.
    """
    with wave.open(audio_path, 'rb') as wave_read:
        channel_num, sample_width, sampling_rate, pcm_length, compression_type, _ = wave_read.getparams()
        assert sample_width == 2, "Only 16-bit WAV PCM supported"
        assert compression_type == 'NONE', "Only linear PCM WAV files supported"
        assert channel_num == 1, "Only mono WAV PCM supported"
        assert sampling_rate == 16000, "Only 16 KHz audio supported"
        audio = np.frombuffer(wave_read.readframes(pcm_length * channel_num), dtype=np.int16).reshape((pcm_length, channel_num))
    return audio.flatten(), sampling_rate


def main(args):
    """
    Main function that is used to run demo.
    """
    log.info('OpenVINO Inference Engine')
    log.info('\tbuild: {}'.format(get_version()))
    engine = create_model(
        model_path = args.model,
        vocab_path = args.vocab,
        device = args.device
    )
    audio, sampling_rate = load_audio_wav(args.input)
    start_time = perf_counter()
    text = engine(audio, sampling_rate)
    latency = (perf_counter() - start_time) * 1e3
    log.info("Metrics report:")
    log.info("\tLatency: {:.1f} ms".format(latency))
    print(text)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    parser.add_argument('-m', '--model', help='Optional. Path to an .xml file with a trained model.', default=None)
    parser.add_argument('-i', '--input', help="Required. Path to an audio file in WAV PCM 16 kHz mono format", required=True)
    parser.add_argument('-v', '--vocab', help="Optional. Path to vocabulary file in .json format", default=None)
    parser.add_argument('-d', '--device', default='CPU',
                        help="Optional. Specify the target device to infer on, for example: "
                             "CPU, GPU, HDDL, MYRIAD or HETERO. "
                             "The demo will look for a suitable IE plugin for this device. Default value is CPU.")
    args = parser.parse_args()
    main(args)
