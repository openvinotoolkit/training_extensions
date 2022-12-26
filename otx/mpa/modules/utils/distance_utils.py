# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import subprocess
import warnings

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.metrics import average_precision_score
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch import nn


def distance_metric(query, gallery, metric='euclidean'):
    if gallery is None:
        gallery = query
    m = len(query)
    n = len(gallery)
    x = np.reshape(query, (m, -1))
    y = np.reshape(gallery, (n, -1))
    if metric == 'euclidean':
        dist = euclidean_distances(x, y)
    elif metric == 'cosine':
        dist = cosine_distances(x, y)
    else:
        raise KeyError("Unsupported distance metric:", metric)

    return dist


def mean_ap(distmat, query_ids, gallery_ids):
    distmat = distmat
    m, n = distmat.shape
    if gallery_ids is None:
        gallery_ids = query_ids
        cut = True
    else:
        cut = False

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    if cut:
        indices = indices[:, 1:]
        matches = matches[:, 1:]
    # Compute AP for each query
    aps = []
    for i in range(m):
        y_true = matches[i]
        y_score = -distmat[i][indices[i]]
        if not np.any(y_true):
            if y_score[0] < -0.5:
                aps.append(1)
            else:
                aps.append(0)
        else:
            aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        # raise RuntimeError("No valid query")
        return 0
    return np.mean(aps)


def calculate_cmc(distmat, query_ids, gallery_ids, topk=100, first_match_break=True):
    if gallery_ids is None:
        gallery_ids = query_ids
    distmat = distmat.cpu().numpy()
    m, n = distmat.shape

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        if not np.any(matches[i]):
            continue
        repeat = 1
        for _ in range(repeat):
            index = np.nonzero(matches[i])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk:
                    break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        # raise RuntimeError("No valid query")
        return [0]
    return ret.cumsum() / num_valid_queries


def init_dist(args, backend="nccl"):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    if not dist.is_available():
        args.launcher = "none"

    if args.launcher == "pytorch":
        # DDP
        init_dist_pytorch(args, backend)
        return True

    elif args.launcher == "slurm":
        # DDP
        init_dist_slurm(args, backend)
        return True

    elif args.launcher == "none":
        # DataParallel or single GPU
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            args.total_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        else:
            args.total_gpus = torch.cuda.device_count()
        if args.total_gpus > 1:
            warnings.warn(
                "It is highly recommended to use DistributedDataParallel by setting "
                "args.launcher as 'slurm' or 'pytorch'."
            )
        return False

    else:
        raise ValueError("Invalid launcher type: {}".format(args.launcher))


def get_dist_info():
    try:
        # data distributed parallel
        return dist.get_rank(), dist.get_world_size(), True
    except Exception:
        return 0, 1, False


def init_dist_pytorch(args, backend="nccl"):
    args.rank = int(os.environ["LOCAL_RANK"])
    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        args.ngpus_per_node = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        args.ngpus_per_node = torch.cuda.device_count()
    assert args.ngpus_per_node > 0, "CUDA is not supported"
    args.gpu = args.rank
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend=backend)
    args.total_gpus = dist.get_world_size()
    args.world_size = args.total_gpus


def init_dist_slurm(args, backend="nccl"):
    args.rank = int(os.environ["SLURM_PROCID"])
    args.world_size = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        args.ngpus_per_node = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        args.ngpus_per_node = torch.cuda.device_count()
    assert args.ngpus_per_node > 0, "CUDA is not supported"
    args.gpu = args.rank % args.ngpus_per_node
    torch.cuda.set_device(args.gpu)
    addr = subprocess.getoutput(
        "scontrol show hostname {} | head -n1".format(node_list)
    )
    os.environ["MASTER_PORT"] = str(args.tcp_port)
    os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["RANK"] = str(args.rank)
    dist.init_process_group(backend=backend)
    args.total_gpus = dist.get_world_size()


def simple_group_split(world_size, rank, num_groups):
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(dist.new_group(rank_list[i]))
    group_size = world_size // num_groups
    print(
        "Rank no.{} start sync BN on the process group of {}".format(
            rank, rank_list[rank // group_size]
        )
    )
    return groups[rank // group_size]


def convert_sync_bn(model, process_group=None):
    for _, (child_name, child) in enumerate(model.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            if isinstance(child, nn.modules.instancenorm._InstanceNorm):
                continue
            m = nn.SyncBatchNorm.convert_sync_batchnorm(child, process_group)
            m.weight.requires_grad_(child.weight.requires_grad)
            m.bias.requires_grad_(child.bias.requires_grad)
            m.to(next(child.parameters()).device)
            setattr(model, child_name, m)
        else:
            convert_sync_bn(child, process_group)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


@torch.no_grad()
def broadcast_tensor(x, src, gpu=None):
    rank, world_size, is_dist = get_dist_info()
    if not is_dist:
        return x

    container = torch.empty_like(x).cuda(gpu)
    if rank == src:
        container.data.copy_(x)
    dist.broadcast(container, src)
    return container.cpu()


@torch.no_grad()
def broadcast_value(x, src, gpu=None):
    rank, world_size, is_dist = get_dist_info()
    if not is_dist:
        return x

    container = torch.Tensor([0.0]).cuda(gpu)
    if rank == src:
        tensor_x = torch.Tensor([x])
        container.data.copy_(tensor_x)
    dist.broadcast(container, src)
    return container.cpu()[0].item()


@torch.no_grad()
def all_gather_tensor(x, gpu=None, save_memory=False):
    rank, world_size, is_dist = get_dist_info()

    if not is_dist:
        return x

    if not save_memory:
        # all gather features in parallel
        # cost more GPU memory but less time
        # x = x.cuda(gpu)
        x_gather = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(x_gather, x, async_op=False)
        x_gather = torch.cat(x_gather, dim=0)
    else:
        # broadcast features in sequence
        # cost more time but less GPU memory
        container = torch.empty_like(x).cuda(gpu)
        x_gather = []
        for k in range(world_size):
            container.data.copy_(x)
            print("gathering features from rank no.{}".format(k))
            dist.broadcast(container, k)
            x_gather.append(container.cpu())
        x_gather = torch.cat(x_gather, dim=0)
        # return cpu tensor

    return x_gather
