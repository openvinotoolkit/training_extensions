# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# OpenVINO Toolkit Open Mozel Zoo 2021.4:
# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file contains modified code snippets from demos/speech_recognition_quartznet_demo/python/speech_recognition_quartznet_demo.py
# from https://github.com/openvinotoolkit/open_model_zoo tag 2021.4
#
from typing import Tuple, Dict, List
import json
# Librosa workaround is based on demos/speech_recognition_quartznet_demo/python/speech_recognition_quartznet_demo.py
# from https://github.com/openvinotoolkit/open_model_zoo tag 2021.4:
# Workaround to import librosa on Linux without installed libsndfile.so
try:
    import librosa
except OSError:
    import sys
    import types
    sys.modules['soundfile'] = types.ModuleType('fake_soundfile')
    import librosa
    del sys.modules['soundfile']

import numpy as np
import scipy
import wave

from openvino.runtime import Core, PartialShape


class Vocab:
    """ Vocabulary.

    Arguments:
        vocab (str or list): Alphabet.
        space_symbol (str): Space symbol in BPE format..
        pad_id (int): Index of padding token.
        bos_id (int): Index of 'begin of sentence' token.
        eos_id (int): Index of 'end of sentence' token.
        unk_id (int): Index of unknown token.
    """
    def __init__(
            self,
            vocab: str or List[str] = " abcdefghijklmnopqrstuvwxyz'-",
            space_symbol: str = " ",
            pad_id: int = 29,
            bos_id: int = 29,
            eos_id: int = 29,
            unk_id: int = 29
    ):
        self.vocab = vocab
        self.space_symbol = space_symbol
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id

    def __len__(self) -> int:
        """ Returns length of vocabulary.

        Returns:
            int: length of vocabulary.
        """
        return len(self.vocab)

    def __getitem__(self, idx: int) -> str:
        """ Returns token from vocabulary by index.

        Arguments:
            idx (int): Index of token from vocabulary.

        Returns:
            str: Token from vocabulary.
        """
        assert idx < len(self.vocab)
        return self.vocab[idx]

    def remove_special_symbols(self, s: str) -> str:
        """ Remove special tokens from input string.
        Special tokens: pad_id, bos_id, eos_id, unk_id, space_symbol.

        Arguments:
            s (str): Input string.

        Returns:
            str: Output string.
        """
        for t in [self.pad_id, self.bos_id, self.eos_id, self.unk_id]:
            s = s.replace(self.vocab[t], "")
        s = s.replace(self.space_symbol, " ")
        if len(s) and s[0] == " ":
            s = s[1:]
        return s


class QuartzNet:
    """ QuartzNet inferencer.

    Arguments:
        model_path (str): path to model's .xml file.
        vocab_path (str): path to vocab in .json fromat.
        device (str): inference device.
        pad_to (int): pad sequence to.
    """
    def __init__(self, model_path: str, vocab_path: str, device: str = "CPU", pad_to: int = 16):
        self.pad_to = pad_to
        self.vocab = Vocab(**json.load(open(vocab_path)))
        self.core = Core()
        self.model = self.core.read_model(model_path)
        if len(self.model.inputs) != 1:
            raise RuntimeError('QuartzNet must have one input')
        if len(self.model.outputs) != 1:
            raise RuntimeError('QuartzNet must have one output')
        self.input_tensor_name = self.model.inputs[0].get_any_name()
        self.n_mels = self.model.inputs[0].shape[1]
        self.device = device
        self.infer_request = self.core.compile_model(self.model, device).create_infer_request()

    def __call__(
            self,
            audio: np.array,
            sampling_rate: int = 16000,
            remove_special_symbols: bool = True
    ) -> str:
        """ Main inferencer method.

        Arguments:
            audio (np.array): Waveform.
            sampling_rate (int): sampling rate of input audio (default = 16000).
            remove_special_symbols (bool): remove special symbols after decoding (default = True).

        Returns:
            text (str): recognized text.
        """
        melspec = self._audio_to_melspec(audio, sampling_rate)
        probs = self._infer(melspec)
        text = self._ctc_greedy_decode(probs, remove_special_symbols)
        return text

    # This method is based on demos/speech_recognition_quartznet_demo/python/speech_recognition_quartznet_demo.py
    # from https://github.com/openvinotoolkit/open_model_zoo tag 2021.4
    def _audio_to_melspec(self, audio: np.array, sampling_rate: int) -> np.ndarray:
        """ Convert waveform to MEL-spectrogram.

        Arguments:
            audio (np.array): Waveform.
            sampling_rate (int): sampling rate of input audio (default = 16000).

        Returns:
            melspec (np.array): MEL-spectrogram.
        """
        assert sampling_rate == 16000, "Only 16 KHz audio supported"
        preemph = 0.97
        preemphased = np.concatenate([audio[:1], audio[1:] - preemph * audio[:-1].astype(np.float32)])

        win_length = round(sampling_rate * 0.02)
        spec = np.abs(librosa.core.spectrum.stft(preemphased, n_fft=512, hop_length=round(sampling_rate * 0.01),
            win_length=win_length, center=True, window=scipy.signal.windows.hann(win_length), pad_mode='reflect'))
        mel_basis = librosa.filters.mel(sampling_rate, 512, n_mels=self.n_mels, fmin=0.0, fmax=8000.0, norm='slaney', htk=False)
        log_melspectrum = np.log(np.dot(mel_basis, np.power(spec, 2)) + 2 ** -24)

        normalized = (log_melspectrum - log_melspectrum.mean(1)[:, None]) / (log_melspectrum.std(1)[:, None] + 1e-5)
        remainder = normalized.shape[1] % self.pad_to
        if remainder != 0:
            return np.pad(normalized, ((0, 0), (0, self.pad_to - remainder)))[None]
        return normalized[None]

    def _reshape(self, input_shape: Tuple[int]):
        """ Reshape network.

        Arguments:
            input_shape (Tuple[int]): shape of input melspec.
        """
        if self.model.inputs[0].shape != input_shape:
            self.model.reshape({self.input_tensor_name: PartialShape(input_shape)})
        self.infer_request = self.core.compile_model(self.model, self.device).create_infer_request()

    def _infer(self, melspec: np.ndarray) -> np.ndarray:
        """ Run network inference.

        Arguments:
            melspec (np.ndarray): input MEL-spectrogram.

        Returns:
            preds (np.ndarray): output predictions.
        """
        self._reshape(melspec.shape)
        input_data = {self.input_tensor_name: melspec}
        return next(iter(self.infer_request.infer(input_data).values()))

    # This method is based on demos/speech_recognition_quartznet_demo/python/speech_recognition_quartznet_demo.py
    # from https://github.com/openvinotoolkit/open_model_zoo tag 2021.4
    def _ctc_greedy_decode(self, pred: np.ndarray, remove_special_symbols: str = True) -> str:
        """ Greedy CTC decoder.

        Arguments:
            preds (np.ndarray): output predictions.
            remove_special_symbols (bool): remove special symbols after decoding (default = True).

        Returns:
            text (str): recognized text.
        """
        prev_id = blank_id = self.vocab.pad_id
        transcription = []
        for idx in pred[0].argmax(1):
            if prev_id != idx != blank_id:
                transcription.append(self.vocab[idx])
            prev_id = idx
        out = ''.join(transcription)
        if remove_special_symbols:
            out = self.vocab.remove_special_symbols(out)
        return out
