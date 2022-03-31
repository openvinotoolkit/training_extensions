# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
from subprocess import run, DEVNULL, CalledProcessError  # nosec - disable B404:import-subprocess check

import json
import torch
from pytorch_lightning.callbacks import Callback
from addict import Dict
from collections import OrderedDict

from torchtts.datasets import collate_tts, TrainTTSDatasetSampler

def load_cfg(path):
    with open(path) as stream:
        cfg = Dict(json.load(stream))
    return cfg


def load_weights(target, source_state):
    new_dict = OrderedDict()
    for k, v in target.state_dict().items():
        if k in source_state and v.size() == source_state[k].size():
            new_dict[k] = source_state[k]
        elif k in source_state and v.size() != source_state[k].size():
            print(f"src: {source_state[k].size()}, tgt: {v.size()}")
            new_dict[k] = v
        else:
            print(f"key {k} not loaded...")
            new_dict[k] = v
    target.load_state_dict(new_dict)


class StopCallback(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def on_batch_end(self, trainer, pl_module):
        trainer.should_stop = self.should_stop

    def stop(self):
        self.should_stop = True

    def reset(self):
        self.should_stop = False


def extract_annotation(dataset):
    annotation = []
    for sample in dataset:
        text = sample["text"]
        if dataset.tokenizer is not None:
            text = dataset.tokenizer.decode(text)[0]
        annotation.append(text)
    return annotation


def build_dataloader(dataset, batch_size, num_workers, shuffle, train=False, distributed=False):
    train_sampler = None
    if train:
        train_sampler = TrainTTSDatasetSampler(dataset, batch_size, 3 * batch_size, distributed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_tts,
        shuffle=shuffle,
        sampler=train_sampler
    )


def export_ir(onnx_model_path, input_shape=None,
                optimized_model_dir='./ir_model', data_type='FP32'):
    def get_mo_cmd():
        for mo_cmd in ('mo', 'mo.py'):
            try:
                run([mo_cmd, '-h'], stdout=DEVNULL, stderr=DEVNULL, shell=False, check=True)
                return mo_cmd
            except CalledProcessError:
                pass
        raise RuntimeError('OpenVINO Model Optimizer is not found or configured improperly')

    mo_cmd = get_mo_cmd()

    command_line = [mo_cmd, f'--input_model={onnx_model_path}',
                    f'--output_dir={optimized_model_dir}',
                    '--data_type', f'{data_type}']
    if input_shape:
        command_line.extend(['--input_shape', f"{input_shape}"])

    # run() will raise a ValueError in case of an embedded NUL character
    run(command_line, shell=False, check=True)


def find_file(dir, filename):
    for root, dirs, files in os.walk((os.path.normpath(dir)), topdown=False):
        for name in files:
            if filename in name:
                return os.path.join(root, name)
    return None
