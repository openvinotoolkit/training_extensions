# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
import string
import sys
from typing import List, Callable, Optional, Tuple, Dict
from pathlib import Path
import torch
import torch.nn as nn
import torchaudio


def format_text(
        text: str,
        to_lowercase: bool = True,
        strip: bool = True,
        punctuation: str = string.punctuation + '—–«»−…‑'
) -> str:
    if to_lowercase:
        text = text.lower()
    if strip:
        text = text.strip()
    if punctuation is not None:
        text = text.translate(str.maketrans('', '', punctuation))
    return text


def parse_librispeech_dataset(path: str, ext_audio: str = ".flac", ext_text: str = ".trans.txt", preprocess_text: bool = True) -> Dict:
    """Parse dataset folder in Librispeech format to internal dataset format"""
    def get_transcription(path_text, speaker_id, chapter_id, utterance_id):
        fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
        with open(path_text) as ft:
            for line in ft:
                fileid_text, utterance = line.strip().split(" ", 1)
                if fileid_audio == fileid_text:
                    break
            else:
                # Translation not found
                raise FileNotFoundError("Translation not found for " + fileid_audio)
        return utterance

    dataset = []
    walker = sorted(str(p.stem) for p in Path(path).glob('*/*/*' + ext_audio))
    for fileid in walker:
        speaker_id, chapter_id, utterance_id = fileid.split("-")

        file_text = speaker_id + "-" + chapter_id + ext_text
        file_text = os.path.join(path, speaker_id, chapter_id, file_text)

        fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
        file_audio = fileid_audio + ext_audio
        file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)
        sample = {
            "audio_path": file_audio,
            "text": get_transcription(file_text, speaker_id, chapter_id, utterance_id),
            "speaker_id": speaker_id,
            "chapter_id": chapter_id,
            "utterance_id": utterance_id
        }
        if preprocess_text:
            sample["text"] = format_text(sample["text"])
        dataset.append(sample)
    return dataset


def load_audio(path: str):
    """Load audio file."""
    waveform, sample_rate = torchaudio.load(path)
    return waveform, sample_rate


class AudioDataset(torch.utils.data.Dataset):
    """Base class for Speech To Text Dataset in PyTorch."""
    def __init__(
            self,
            samples: List[Dict],
            load_audio: bool = True,
            load_text: bool = True,
            audio_transforms: Callable = None,
            tokenizer: Callable = None,
            **kwargs
    ):
        self.samples = samples
        self.load_audio = load_audio
        self.load_text = load_text
        self.audio_transforms = audio_transforms
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio, _ = load_audio(self.samples[idx]["audio_path"]) if self.load_audio else None
        text = self.samples[idx]["text"] if self.load_text else None
        sample_rate = 16000 # TODO: add dynamic sample rate support
        if self.audio_transforms is not None:
            audio, sample_rate = self.audio_transforms(audio, sample_rate)
        if self.tokenizer is not None and self.load_text:
            text = self.tokenizer.encode(text)
        return {
            "audio": audio,
            "sample_rate": sample_rate,
            "text": text
        }

    @staticmethod
    def get_collate_fn(audio_pad_id: int, text_pad_id: int):
        def _collate_fn(batch):
            out = {
                "audio": [], "audio_lengths": [],
                "text": [], "text_lengths": [],
            }
            for sample in batch:
                if sample["audio"] is not None:
                    audio = sample["audio"].squeeze(0).transpose(0, 1)
                    out["audio"].append(audio)
                    out["audio_lengths"].append(audio.shape[0])
                if sample["text"] is not None:
                    out["text"].append(sample["text"])
                    out["text_lengths"].append(sample["text"].shape[0])
            if len(out["audio"]):
                out["audio"] = nn.utils.rnn.pad_sequence(out["audio"], batch_first=True, padding_value=audio_pad_id)
                out["audio"] = out["audio"].transpose(1, 2)
                out["audio_lengths"] = torch.LongTensor(out["audio_lengths"])
            if len(out["text"]):
                out["text"] = nn.utils.rnn.pad_sequence(out["text"], batch_first=True, padding_value=text_pad_id)
                out["text_lengths"] = torch.LongTensor(out["text_lengths"])
            return out
        return _collate_fn
