"""
 Copyright (c) 2020 Intel Corporation
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

import json
import tokens_bert
import logging
import multiprocessing
import time
import pickle
import torch
import socket
import os

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)
logger = logging.getLogger('{} utils'.format(os.getpid()))
def printlog(*args):
    logger.info(' '.join([str(v) for v in args]))

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

def sync_grads(rank, named_params, report_no_grad_params=True):
    if rank < 0:
        return
    world_size = torch.distributed.get_world_size()
    #average all grads
    for n, p in named_params:
        if p.grad is None:
            if report_no_grad_params and rank in [-1, 0]:
                logger.warning("{} param has no grad but in the optimization list".format(n))
            continue
        torch.distributed.all_reduce(p.grad.data, op=torch.distributed.ReduceOp.SUM)
        p.grad.data /= float(world_size)

def encode_squad_article(article, vocab, do_lower_case):
    def encode_txt(txt):
        if do_lower_case:
            txt = txt.lower()
        return tokens_bert.text_to_tokens(txt, vocab)

    for par in article['paragraphs']:
        par['context_enc'], par['context_enc_pos'] = encode_txt(par['context'])
        for qa in par['qas']:
            qa['question_enc'], qa['question_enc_pos'] = encode_txt(qa['question'])

    return article


def squad_read_and_encode(rank, device, squad_file, tokenizer):
    if rank in [-1, 0]:
        #read and encode squad
        with open(squad_file, 'rt') as inp:
            squad = json.load(inp)

        N = len(squad['data'])

        t0 = time.time()
        printlog("Encode Squad {} articles from {} file ...".format(N, squad_file))
        with multiprocessing.Pool() as pool:
            squad['data'] = pool.starmap(
                encode_squad_article,
                zip(squad['data'], [tokenizer.vocab] * N, [tokenizer.basic_tokenizer.do_lower_case]*N)
            )
        t1 = time.time()
        printlog("Encoded Squad {} articles by {} sec".format(N, t1-t0))

    if rank > -1:
        #send squad to other processes
        if rank == 0:
            squad_buf = pickle.dumps(squad)
            squad_storage = torch.ByteStorage.from_buffer(squad_buf)
            squad_tensor = torch.ByteTensor(squad_storage).to(device)
            size_tensor = torch.tensor([squad_tensor.numel()], dtype=torch.long, device=device)
            printlog("Pack squad articles and sent to other processes {} bytes".format(squad_tensor.numel()))
        else:
            size_tensor = torch.tensor([0], dtype=torch.long, device=device)

        torch.distributed.broadcast(size_tensor, 0)

        if rank > 0:
            squad_tensor = torch.empty((size_tensor.item(),), dtype=torch.uint8, device=device)

        torch.distributed.broadcast(squad_tensor, 0)

        if rank > 0:
            printlog("Receive squad articles and unpack {} bytes by {} process".format(size_tensor.item(), rank))
            squad_buf = squad_tensor.cpu().numpy().tobytes()
            squad = pickle.loads(squad_buf)
        torch.distributed.barrier()

    return squad



def get_free_port():
    for port in range(100):
        port += 12355
        sock = socket.socket()
        res = sock.connect_ex(('localhost', port))
        sock.close()
        if res != 0:
            printlog("port {} is free".format(port))
            return port
        else:
            printlog("port {} is used".format(port))
    return None

#prepare parameters groups for optimizer and named list of all optimized parameters
def make_param_groups(rank, model, freeze_list, lr, lr_fq, fq_tune_only, model_controller):
    # split parameters to FQ and the rest
    params_fq = set()
    if model_controller is not None:
        for k, v in model_controller.all_quantizations.items():
            for n, p in v.named_parameters():
                params_fq.add(p)

    #create total list of named parameters to optimization
    named_params = []
    if isinstance(freeze_list,str):
        freeze_list = freeze_list.split(',') if freeze_list else []
    for n, p in model.named_parameters():
        if n.lower()!="none" and any(fn in n for fn in freeze_list):
            if rank in [-1, 0]:
                logger.warning("rank {} {} param is frozen and excluded from tune".format(rank,n))
            continue

        if fq_tune_only and p not in params_fq:
            continue

        #for unknown reason nncf define internal flags as integer Parameters
        #these flags are not for optimization so just filter them out
        if p in params_fq and not p.data.is_floating_point():
            continue

        named_params.append((n,p))

    #split filtered named_params into 2 groups
    #1 for FQ parameters
    #2 for the rest parameters
    groups = []
    #add FQ group
    params = [p for n, p in named_params if p in params_fq]
    if params:
        groups.append({'params': params, 'lr': lr_fq})
    #add rest group
    params = [p for n, p in named_params if p not in params_fq]
    if params:
        groups.append({'params': params,  'lr': lr})
    del params
    return named_params, groups

