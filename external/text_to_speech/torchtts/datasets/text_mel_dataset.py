# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
from pathlib import Path
import random
import math

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import torch.distributed as dist

from torchtts.text_preprocessing import text_to_sequence, intersperse, cmudict, random_fill
from .tacotronstft import TacotronSTFT
from scipy.io.wavfile import read


def parse_ljspeech_dataset(csv_path, data_root):
    dataset = []
    with open(csv_path, 'r', encoding="utf8", errors="ignore") as f:
        for line in f:
            datas = line.strip().split('|')
            audio_path = f'{data_root}/wavs/{datas[0]}.wav'
            text = datas[-1]
            dataset.append({"audio_path": audio_path,
                            "text": text})
    return dataset


def get_tts_datasets(cfg, max_mel_len=1000):
    if hasattr(cfg, "test_ann_file") and cfg.test_ann_file is not None:
        return TTSDatasetWithSTFT(Path(cfg.test_data_root), cfg.test_ann_file, cfg)
    if hasattr(cfg, "train_ann_file"):
        train_dataset = TTSDatasetWithSTFT(Path(cfg.train_data_root), cfg.train_ann_file, cfg, max_mel_len,
                                           add_noise=True)
        val_dataset = TTSDatasetWithSTFT(Path(cfg.val_data_root), cfg.val_ann_file, cfg, max_mel_len)
    else:
        path = Path(cfg.training_path)
        train_dataset = TTSDatasetWithSTFT(path, path / 'metadata_train.csv', cfg, max_mel_len, add_noise=True)
        val_dataset = TTSDatasetWithSTFT(path, path / 'metadata_val.csv', cfg, max_mel_len)

    return train_dataset, val_dataset


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


class TTSDatasetWithSTFT(Dataset):
    def __init__(self, path: Path=None, csv_path: Path=None,
                 dataset_items=None, cfg=None,
                 add_noise=False, items_type=None):
        self.path = path
        self.metadata = []
        self.text_dict = {}

        self.add_noise = add_noise

        if dataset_items is not None:
            self.metadata = []
            for item in dataset_items:
                if items_type is not None and item.subset != items_type:
                    continue
                audio_path = item.media.metadata
                text = str(item.media.numpy)
                self.metadata.append({"audio_path": audio_path, "text": text})
        else:
            self.metadata = parse_ljspeech_dataset(csv_path, path)

        print('Read {} records for TTS dataset. Add noise {}'.format(len(self.metadata), self.add_noise))

        self.cmudict = None
        if cfg is None:
            self.add_blank = True
            self.cmudict = cmudict.CMUDict()
            self.stft = TacotronSTFT()
            self.max_wav_value = 32768.0
        else:
            self.add_blank = getattr(self.cfg, "add_blank", False)
            if not(getattr(cfg, "cmudict_path", None)) is dict:
                self.cmudict = cmudict.CMUDict(cfg.cmudict_path)
            self.max_wav_value = cfg.max_wav_value
            self.stft = TacotronSTFT(
                filter_length=cfg.filter_length, hop_length=cfg.hop_length, win_length=cfg.win_length,
                n_mel_channels=cfg.n_mel_channels, sampling_rate=cfg.sampling_rate, mel_fmin=cfg.mel_fmin,
                mel_fmax=cfg.mel_fmax)

    def __getitem__(self, index):
        item = self.metadata[index]
        x = text_to_sequence(item["text"], dictionary=self.cmudict)
        if self.add_noise:
            x = random_fill(x, p=0.05)
        if self.add_blank:
            x = intersperse(x)

        filename = str(item["audio_path"])
        mel = self.get_mel(filename)

        mel = (mel + 6) / 6.0

        x = torch.IntTensor(x)
        return x, mel

    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        if self.add_noise:
            audio = audio + torch.rand_like(audio)
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        return melspec

    def __len__(self):
        return len(self.metadata)

    def get_mel_lengths(self):
        res = []
        for item in self.metadata:
            sampling_rate, data = read(item["audio_path"])
            res.append(data.shape[-1] / self.stft.hop_length)
        return res


def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')


def pad2d(x, max_len):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode='constant', constant_values=-1.0)


def collate_tts(batch):
    char_lens = [len(x[0]) for x in batch]
    max_x_len = max(char_lens)

    chars = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars)

    mel_lens = [x[1].shape[-1] for x in batch]
    max_spec_len = max(mel_lens) + 1

    mel = [pad2d(x[1], max_spec_len) for x in batch]
    mel = np.stack(mel)

    chars = torch.tensor(chars).long()
    mel = torch.tensor(mel)
    char_lens = torch.tensor(char_lens).long()
    mel_lens = torch.tensor(mel_lens).long()

    return chars, char_lens, mel, mel_lens


def round_down(num, divisor):
    return num - (num % divisor)


class TrainTTSDatasetSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size, bin_size, distributed=True): # by default bin_size = 3 * batch_size
        _, self.idx = torch.sort(torch.tensor(data_source.get_mel_lengths()).long())
        self.batch_size = batch_size
        self.num_samples = len(data_source)
        self.bin_size = bin_size
        self.distributed = distributed

        assert self.bin_size % self.batch_size == 0

        self.num_replicas = 1
        if torch.cuda.is_available():
            self.num_replicas = max(1, torch.cuda.device_count())
        self.num_samples = int(math.ceil(len(self.idx) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        idx = self.idx.numpy()
        bins = []

        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            random.shuffle(this_bin)
            bins += [this_bin]

        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)

        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])

        # Divide data to each GPU
        if self.distributed:
            total_size = round_down(len(binned_idx), self.batch_size * dist.get_world_size())
            start_index = int((dist.get_rank()) / dist.get_world_size() * total_size)
            end_index = int((dist.get_rank() + 1) / dist.get_world_size() * total_size)
            self.num_samples = end_index - start_index

            start_idx = dist.get_rank() * self.batch_size
            steps = self.num_samples // self.batch_size
            offset = dist.get_world_size() * self.batch_size
            idxs = []
            for i in range(steps):
                idxs.extend(binned_idx[start_idx:start_idx+self.batch_size])
                start_idx += offset
            return iter(idxs)
        else:
            total_size = round_down(len(binned_idx), self.batch_size)
            return iter(binned_idx[:total_size])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
