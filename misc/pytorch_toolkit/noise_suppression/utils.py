"""
 Copyright (c) 2021 Intel Corporation

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

import logging
import torch
import socket
import os
import pickle

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)
logger = logging.getLogger('{} utils'.format(os.getpid()))
def printlog(*args):
    logger.info(' '.join([str(v) for v in args]))

def get_shape(x):
    return list(x.shape)
    #this is to get more compact and human readable but unresizable onnx model
    #return [int(s) for s in x.shape]

sync_model_count = -1
def sync_models(rank, model):
    if rank < 0:
        return
    world_size = torch.distributed.get_world_size()
    # sync model in different process

    with torch.no_grad():
        for p in model.parameters():
            torch.distributed.all_reduce(p.data, op=torch.distributed.ReduceOp.SUM)
            if p.data.is_floating_point():
                p.data /= float(world_size)
            else:
                p.data = p.data // world_size

def sync_grads(rank, named_params, report_no_grad_params=True, grad_count=1):
    if rank < 0 and grad_count==1:
        return

    world_size = torch.distributed.get_world_size() if rank>-1 else 1
    #average all grads
    for n, p in named_params:
        if p.grad is None:
            if report_no_grad_params and rank in [-1, 0]:
                logger.warning("{} param has no grad but in the optimization list".format(n))
            continue
        if world_size>1:
            torch.distributed.all_reduce(p.grad.data, op=torch.distributed.ReduceOp.SUM)
        p.grad.data /= float(world_size*grad_count)


def get_free_port():
    for port in range(100):
        port += 12355
        sock = socket.socket()
        res = sock.connect_ex(('localhost', port))
        sock.close()
        if res != 0:
            printlog("port {} is free".format(port))
            return port
        printlog("port {} is used".format(port))
    return None


def brodcast_data(rank, data, rank_src=0):
    if rank<0:
        return data

    device = torch.device("cuda:{}".format(rank))

    # send starts to other processes
    if rank == rank_src:
        data_buf = pickle.dumps(data)
        data_storage = torch.ByteStorage.from_buffer(data_buf)
        data_tensor = torch.ByteTensor(data_storage).to(device)
        size_tensor = torch.tensor([data_tensor.numel()], dtype=torch.long, device=device)
        printlog("Pack data by {} process and sent to other processes {} bytes".format(rank_src, data_tensor.numel()))
    else:
        size_tensor = torch.tensor([0], dtype=torch.long, device=device)

    torch.distributed.broadcast(size_tensor, rank_src)

    if rank != rank_src:
        data_tensor = torch.empty((size_tensor.item(),), dtype=torch.uint8, device=device)

    torch.distributed.broadcast(data_tensor, rank_src)

    if rank != rank_src:
        printlog(
            "Receive data and unpack {} bytes by {} process".format(size_tensor.item(), rank))
        data_buf = data_tensor.cpu().numpy().tobytes()
        data = pickle.loads(data_buf)
    torch.distributed.barrier()

    return data
