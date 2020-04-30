"""
 Copyright (c) 2019 Intel Corporation
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

import os

import torch

from nncf.checkpoint_loading import load_state


def save_checkpoint(model, optimizer, epoch, miou, compression_scheduler, config):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The model to save.
    - optimizer (``torch.optim``): The optimizer state to save.
    - epoch (``int``): The current epoch for the model.
    - miou (``float``): The mean IoU obtained by the model.
    - compression_scheduler: The compression scheduler associated with the model
    - config: Model config".

    Returns:
        The path to the saved checkpoint.
    """
    name = config.name
    save_dir = config.checkpoint_save_dir

    assert os.path.isdir(
        save_dir), "The directory \"{0}\" doesn't exist.".format(save_dir)

    # Save model
    checkpoint_path = os.path.join(save_dir, name) + "_last.pth"

    checkpoint = {
        'epoch': epoch,
        'miou': miou,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': compression_scheduler.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(model, model_path, device_name, optimizer=None, compression_scheduler=None):
    """Loads the model from a specified directory with a specified name

    Keyword arguments:
    - model (``nn.Module``): The stored model state is copied to this model
    instance.
    - model_path: The model filename.
    - device_name: Device name for the model to be loaded into.
    - is_ddp: If true, model will be treated as a DistributedDataParallel instance
              and the actual model will be loaded into model.module
    - optimizer (``torch.optim``): The stored optimizer state is copied to this
    optimizer instance.
    - compression_ctrl: The compression scheduler for the saved state
                        to be loaded into

    Returns:
    The ``model``, ``optimizer``, epoch, mean IoU and ``compression_scheduler``, loaded from the
    checkpoint.

    """
    assert os.path.isfile(
        model_path), "The model file \"{0}\" doesn't exist.".format(model_path)

    # Load the stored model parameters to the model instance
    checkpoint = torch.load(model_path, map_location=device_name)
    load_state(model, checkpoint['state_dict'], is_resume=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    miou = checkpoint['miou']

    if "scheduler" in checkpoint and compression_scheduler is not None:
        compression_scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, epoch, miou, compression_scheduler
