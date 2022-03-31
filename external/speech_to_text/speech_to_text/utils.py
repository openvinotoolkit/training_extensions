# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file contains code snippets from torchvision v0.9.2 https://github.com/pytorch/vision/blob/v0.9.2/references/detection/utils.py
# Copyright (c) Soumith Chintala 2016
# SPDX-License-Identifier: BSD-3-Clause
#
import pickle  # nosec - disable B403:import-pickle check
import operator
from functools import reduce
from subprocess import run, DEVNULL, CalledProcessError  # nosec - disable B404:import-subprocess check
from collections import OrderedDict
import json
import torch
import torch.distributed as dist
from addict import Dict
import speech_to_text.transforms as transforms
import speech_to_text.metrics as metrics
from speech_to_text.datasets import AudioDataset, parse_librispeech_dataset


def build_dataset(data_path, ext_audio: str = ".flac", ext_text: str = ".trans.txt", load_audio=True, load_text=True, **kwargs):
    if not isinstance(data_path, list):
        data_path = [data_path]
    samples = []
    for path in data_path:
        samples.extend(parse_librispeech_dataset(path, ext_audio, ext_text))
    return AudioDataset(samples, load_audio=load_audio, load_text=load_text)


def build_audio_transforms(cfg):
    def build_transform(name, params):
        if hasattr(transforms, name):
            return getattr(transforms, name)(**params)
        else:
            return None
    out = []
    for p in cfg:
        t = build_transform(p["name"], p["params"])
        if t is not None:
            out.append(t)
    return transforms.AudioCompose(out)


def build_metrics(names):
    out = []
    for name in names:
        if hasattr(metrics, name):
            out.append(getattr(metrics, name)())
    return metrics.MetricAggregator(out)


def build_tokenizer(data_path: str, model_path: str, vocab_size: int) -> transforms.TextTokenizerYTTM:
    transforms.TextTokenizerYTTM.train(
        data_path=data_path,
        model_path=model_path,
        vocab_size=vocab_size
    )
    return transforms.TextTokenizerYTTM(model_path=model_path, vocab_size=vocab_size)


def build_dataloader(dataset, tokenizer, audio_transforms_cfg, batch_size, num_workers, shuffle):
    dataset.tokenizer = tokenizer
    dataset.audio_transforms = build_audio_transforms(audio_transforms_cfg)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        collate_fn = AudioDataset.get_collate_fn(
            audio_pad_id = 0,
            text_pad_id = tokenizer.pad_id if tokenizer is not None else 0
        ),
        shuffle = shuffle
    )


def extract_annotation(dataset):
    annotation = []
    for sample in dataset:
        text = sample["text"]
        if dataset.tokenizer is not None:
            text = dataset.tokenizer.decode(text)[0]
        annotation.append(text)
    return annotation


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


def export_ir(onnx_model_path, optimized_model_dir='./ir_model', input_shape=None, data_type='FP32'):
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


###################################
### Multi-GPU reduction
###################################

# Borrowed from torchvision v0.9.2
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


# Borrowed from torchvision v0.9.2
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


# Borrowed from torchvision v0.9.2
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []

    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        # 'buffer' contains trusted data received from our other instances through all_gather().
        data_list.append(pickle.loads(buffer))  # nosec - disable B301:pickle check

    return data_list
