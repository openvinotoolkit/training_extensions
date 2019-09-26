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

import torch
import torch.multiprocessing as mp


class ExecutionMode:
    CPU_ONLY = "cpu_only"
    SINGLE_GPU = "single_gpu"

    # Multiple nodes, each with 1 process utilising all local GPUs
    DISTRIBUTED = "distributed"

    # Multiple nodes, each with 1 process for each local GPU
    MULTIPROCESSING_DISTRIBUTED = "multiprocessing_distributed"

    # Single node with 1 process utilising all local GPUs
    GPU_DATAPARALLEL = "gpu_dataparallel"


def get_execution_mode(config):
    if config.cpu_only:
        return ExecutionMode.CPU_ONLY
    if config.gpu_id is not None:
        return ExecutionMode.SINGLE_GPU
    if config.multiprocessing_distributed:
        return ExecutionMode.MULTIPROCESSING_DISTRIBUTED
    if config.world_size > 1:
        return ExecutionMode.DISTRIBUTED
    return ExecutionMode.GPU_DATAPARALLEL


def get_device(config):
    if config.execution_mode == ExecutionMode.CPU_ONLY:
        return "cpu"
    if config.current_gpu is not None:
        return "cuda:{}".format(config.current_gpu)

    return "cuda"


def prepare_model_for_execution(model, config):
    model_without_dp = model

    # TODO: enable this. SyncBatchNorm only works with GPU-tensors, so it cannot
    # be moved to create_compressed_model, but if we do the conversion here, then
    # the dynamic graph becomes incorrect.
    #
    # if config.distributed:
    #     try:
    #         from torch.nn import SyncBatchNorm
    #         model = SyncBatchNorm.convert_sync_batchnorm(model)
    #     except ImportError:
    #         print("Current PyTorch version does not support SyncBatchNorm!")

    model.to(config.device)

    if config.execution_mode == ExecutionMode.MULTIPROCESSING_DISTRIBUTED:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(config.current_gpu)
        model = torch.nn.parallel.distributed.DistributedDataParallel(model, device_ids=[config.current_gpu])
        model_without_dp = model.module

    if config.execution_mode == ExecutionMode.DISTRIBUTED:
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)
        model_without_dp = model.module

    if config.execution_mode == ExecutionMode.SINGLE_GPU:
        torch.cuda.set_device(config.current_gpu)

    if config.execution_mode == ExecutionMode.GPU_DATAPARALLEL:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model)
        model_without_dp = model.module

    return model, model_without_dp


def start_worker(main_worker, config):
    if config.execution_mode == ExecutionMode.CPU_ONLY:
        main_worker(current_gpu=None, config=config)
        return

    if config.execution_mode == ExecutionMode.SINGLE_GPU:
        main_worker(current_gpu=config.gpu_id, config=config)
        return

    if config.execution_mode == ExecutionMode.GPU_DATAPARALLEL:
        main_worker(current_gpu=None, config=config)
        return

    if config.execution_mode == ExecutionMode.MULTIPROCESSING_DISTRIBUTED:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.ngpus_per_node = torch.cuda.device_count()
        config.world_size = config.ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=config.ngpus_per_node, args=(config,))
