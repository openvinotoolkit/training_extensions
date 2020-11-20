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

import sys
import collections
import argparse
import logging
import os
import timeit
import time
import utils
import numpy as np


logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)
logger = logging.getLogger('{} train_qcemb'.format(os.getpid()))
def printlog(*args):
    logger.info(' '.join([str(v) for v in args]))

import torch
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler,TensorDataset)

QUANTIZATION = any('nncf_config' in t for t in sys.argv[1:])
printlog('QUANTIZATION', QUANTIZATION)
if QUANTIZATION:
    printlog('import nncf')
    import nncf

import math

from torch.optim.lr_scheduler import LambdaLR
from transformers.modeling_bert import BertPooler
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW
from transformers import AutoConfig
from model_bert_pack import BertPacked


def get_modules(net, net_name=None):
    modules = []
    for n, m in net._modules.items():
        m_name = n if net_name == None else (net_name + "." + n)
        modules.append({
            'name_full': m_name,
            'module': m,
            'name': n,
            'module_parent': net})
        modules += get_modules(m, m_name)
    return modules

class BertModelEMB(BertModel):
    def __init__(self, config):
        super().__init__(config)
        modules = get_modules(self)
        for m_desc in modules:
            m = m_desc['module']
            if isinstance(m, BertPooler):
                class Scale(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.embscale = torch.nn.Parameter(torch.Tensor(1).fill_(0))
                    def forward(self, input):
                        return input*self.embscale.exp()
                m.activation = Scale()
    def forward(self,*args, **kwargs):

        output_hidden_states_old = {}
        if 'output_hidden_states' in kwargs:
            for m in self.modules():
                if hasattr(m, 'output_hidden_states'):
                    output_hidden_states_old[m] = m.output_hidden_states
                    m.output_hidden_states = kwargs['output_hidden_states']

        if len(output_hidden_states_old)>0:
            #remove to support old transformers version
            del kwargs['output_hidden_states']
        outputs = super().forward(*args, **kwargs)

        for m,v in output_hidden_states_old.items():
            m.output_hidden_states = v

        return outputs[1:]

def get_inputs(batch, device):
    res = {
        'input_ids':       batch[0].to(device),
        'attention_mask':  batch[1].to(device),
        'token_type_ids':  batch[2].to(device)
    }
    return res

def create_squad_qcemb_dataset(rank, device, squad_file, tokenizer, max_seq_length_q, max_seq_length_c):

    squad = utils.squad_read_and_encode(rank, device, squad_file, tokenizer)

    pad = [tokenizer.vocab["[PAD]"]]
    cls = [tokenizer.vocab["[CLS]"]]
    sep = [tokenizer.vocab["[SEP]"]]

    q_texts   = []
    q_input_ids   = []
    q_input_mask  = []
    q_segment_ids = []
    q_context_ids = []
    qna_texts   = []
    qna_input_ids   = []
    qna_input_mask  = []
    qna_segment_ids = []
    qna_context_ids = []
    c_texts = []
    c_input_ids   = []
    c_input_mask  = []
    c_segment_ids = []


    def add_sample(ids, max_len,input_ids, input_mask, segment_ids):
        ids_len = min(max_len-2, len(ids))
        ids = ids[:ids_len]
        rest = max_len - (ids_len + 2)
        assert rest >= 0

        input_ids.append(cls + ids + sep + pad * rest)
        input_mask.append([1] * (1 + ids_len + 1) + pad * rest)
        segment_ids.append([0] * (1 + ids_len + 1) + pad * rest)

    for art_i, article in enumerate(squad['data']):
        for par_i, par in enumerate(article['paragraphs']):

            c_texts.append(par['context'])

            add_sample(
                par['context_enc'],
                max_seq_length_c,
                c_input_ids,
                c_input_mask,
                c_segment_ids)

            for qa_i, qa in enumerate(par['qas']):
                if qa['answers']:
                    q_context_ids.append(len(c_input_ids)-1)
                    q_texts.append(qa['question'])
                    add_sample(
                        qa['question_enc'],
                        max_seq_length_q,
                        q_input_ids,
                        q_input_mask,
                        q_segment_ids)
                else:
                    qna_context_ids.append(len(c_input_ids)-1)
                    qna_texts.append(qa['question'])
                    add_sample(
                        qa['question_enc'],
                        max_seq_length_q,
                        qna_input_ids,
                        qna_input_mask,
                        qna_segment_ids)

    dtype = torch.long
    q_dataset = TensorDataset(
        torch.tensor(q_input_ids, dtype=dtype),
        torch.tensor(q_input_mask, dtype=dtype),
        torch.tensor(q_segment_ids, dtype=dtype),
        torch.tensor(q_context_ids, dtype=dtype)
    )

    qna_dataset = TensorDataset(
        torch.tensor(qna_input_ids, dtype=dtype),
        torch.tensor(qna_input_mask, dtype=dtype),
        torch.tensor(qna_segment_ids, dtype=dtype),
        torch.tensor(qna_context_ids, dtype=dtype)
    )

    c_dataset = TensorDataset(
        torch.tensor(c_input_ids, dtype=dtype),
        torch.tensor(c_input_mask, dtype=dtype),
        torch.tensor(c_segment_ids, dtype=dtype)
    )

    class QCEMBDataSet():
        def __init__(self):
            self.q_dataset = q_dataset
            self.qna_dataset = qna_dataset
            self.c_dataset = c_dataset
            self.q_texts = q_texts
            self.c_texts = c_texts
            self.vocab = tokenizer.vocab

    return QCEMBDataSet()


train_count=-1
def train(rank, args, model, model_t, train_dataset_qc, test_dataset_qc, fq_tune_only, model_controller):
    """ Train the model """
    global train_count
    train_count += 1

    world_size = 1 if rank < 0 else torch.distributed.get_world_size()

    if rank in [-1, 0]:
        printlog("Train model", train_count)
        printlog(model)

    q_dataset = train_dataset_qc.q_dataset

    per_gpu_train_batch_size = args.per_gpu_train_batch_size
    train_batch_size = per_gpu_train_batch_size * world_size

    if fq_tune_only:
        gradient_accumulation_steps = 1
        num_train_epochs = 1
    else:
        gradient_accumulation_steps = args.total_train_batch_size // train_batch_size
        num_train_epochs = args.num_train_epochs

    if rank < 0:
        #single process take all
        q_sampler = RandomSampler(q_dataset)
        q_dataloader = DataLoader(q_dataset, sampler=q_sampler, batch_size=train_batch_size, num_workers=4)
    else:
        #special sampler that divide samples between processes
        q_sampler = torch.utils.data.distributed.DistributedSampler(q_dataset, rank=rank)
        q_dataloader = DataLoader(q_dataset, sampler=q_sampler, batch_size=per_gpu_train_batch_size)

    steps_total = int(len(q_dataloader) // gradient_accumulation_steps * num_train_epochs)

    # Prepare optimizer and schedule
    named_params, groups = utils.make_param_groups(
        rank,
        model,
        args.freeze_list, #list or str with subnames to define frozen parameters
        args.learning_rate, #learning rate for no FQ parameters
        0.01,# learning rate for FQ parameters
        fq_tune_only,#true if only FQ parameters will be optimized
        model_controller)

    optimizer = AdamW(groups,eps=1e-08,lr=args.learning_rate,weight_decay=0)

    def lr_lambda(current_step):
        p = float(current_step) / float(steps_total)
        return 1 - p
    scheduler = LambdaLR(optimizer, lr_lambda)

    if rank in [-1, 0]:
        for n,p in named_params:
            printlog('param for tune',n)
        printlog("fq_tune_only", fq_tune_only)
        printlog("dataset size", len(q_dataset) )
        printlog("epoches", num_train_epochs )
        printlog("per_gpu_train_batch_size", per_gpu_train_batch_size )
        printlog("n_gpu", args.n_gpu )
        printlog("world_size", world_size )
        printlog("gradient_accumulation_steps", gradient_accumulation_steps )
        printlog("total train batch size", train_batch_size * gradient_accumulation_steps )
        printlog("steps_total",steps_total )

    global_step = 1
    model.zero_grad()
    indicators = collections.defaultdict(list)

    softplus = torch.nn.Softplus()

    loss_cfg = dict([t.split(':') for t in args.loss_cfg.split(',')])

    hnm_hist = {}

    for epoch in range(math.ceil(num_train_epochs)):
        indicators = collections.defaultdict(list)
        model.train()
        if model_t:
            model_t.train()
        if rank > -1:
            #set epoch to make different samples division betwen process for different epoches
            q_sampler.set_epoch(epoch)

        utils.sync_models(rank, model)
        for step, q_batch in enumerate(q_dataloader):
            epoch_fp = epoch + step/len(q_dataloader)
            if epoch_fp > num_train_epochs:
                break

            losses = []

            context_ids_pos = q_batch[3]
            q_inputs = get_inputs(q_batch, args.device)
            q_outputs = model(**q_inputs, output_hidden_states=(model_t is not None))
            q_vec = q_outputs[0]

            #get positive embeddings
            c_batch = train_dataset_qc.c_dataset[context_ids_pos.detach().data]
            c_inputs = get_inputs(c_batch, args.device)
            c_outputs = model(**c_inputs, output_hidden_states=(model_t is not None))
            c_vec_pos = c_outputs[0]

            if model_t is not None:
                q_emb_s, q_hidden_s = q_outputs
                c_emb_s, c_hidden_s = c_outputs
                with torch.no_grad():
                    q_emb_t, q_hidden_t = model_t(**q_inputs, output_hidden_states=True)
                    c_emb_t, c_hidden_t = model_t(**c_inputs, output_hidden_states=True)

                def align_and_loss_outputs(out_s, out_t):
                    if len(out_s) != len(out_t):
                        #the student and teacher outputs are not aligned. try to find teacher output for each student output
                        n_s, n_t = len(out_s), len(out_t)
                        out_t = [out_t[(i*(n_t-1))//(n_s-1)] for i in range(n_s)]
                    assert len(out_s) == len(out_t), "can not align number of outputs between student and teacher"
                    assert all(s[0] == s[1] for s in zip(out_s[0].shape, out_t[0].shape)), "output shapes for student and teacher are not the same"
                    return [(s - t.detach()).pow(2).mean() for s,t in zip(out_s, out_t)]

                l_q = align_and_loss_outputs(q_hidden_s,q_hidden_t)
                l_c = align_and_loss_outputs(c_hidden_s,c_hidden_t)

                emb_loss = loss_cfg.get('emb_loss','')
                if emb_loss == 'L2':
                    l_q.append((q_emb_s - q_emb_t.detach()).pow(2).mean())
                    l_c.append((c_emb_s - c_emb_t.detach()).pow(2).mean())
                elif emb_loss == 'L1':
                    l_q.append((q_emb_s - q_emb_t.detach()).abs().mean())
                    l_c.append((c_emb_s - c_emb_t.detach()).abs().mean())
                elif emb_loss.lower() not in ['','none','0','disable']:
                    raise Exception('emb_loss={} is unsupported'.format(emb_loss))


                losses.extend([args.supervision_weight*l for l in l_c+l_q])

            triplet_num = int(loss_cfg.get('triplet_num', 1))
            if fq_tune_only:
                triplet_num = 0

            if triplet_num>0:
                #disable grad to select negatives
                with torch.no_grad():
                    hnm_scores = []
                    hnm_idxs = []

                    #check that current step has no HNM conext vector
                    if global_step not in hnm_hist and args.hnm_num > 0:
                        #generate the new one

                        if world_size > 1 and (args.hnm_num % world_size) != 0:
                            #aligh hnm_num per each replica
                            hnm_plus = world_size - (args.hnm_num % world_size)
                            args.hnm_num += hnm_plus
                            logger.warning("rank {} args.hnm_num increased by {} from {} to {} to be the same after division by {} replicas.".format(
                                rank,
                                hnm_plus,
                                args.hnm_num-hnm_plus,
                                args.hnm_num,
                                world_size))

                        # generate random contexts to calc embedding
                        context_ids_all = torch.randint(
                            low=0,
                            high=len(train_dataset_qc.c_dataset),
                            size=[args.hnm_num])

                        if rank < 0: #single process take all
                            context_ids = context_ids_all
                        else:
                            #broadcast one sigle indicies to all processes
                            context_ids_all = context_ids_all.to(args.device)
                            torch.distributed.broadcast(context_ids_all, 0)
                            context_ids_all = context_ids_all.cpu()

                            #each process take only small part to calc embedding
                            s = ((rank+0) * args.hnm_num) // world_size
                            e = ((rank+1) * args.hnm_num) // world_size
                            context_ids = context_ids_all[s:e]

                        batch_size = min(args.hnm_batch_size, context_ids.shape[0])

                        s,e = 0,batch_size
                        c_outputs = []
                        while e>s:
                            idx = context_ids.detach()[s:e]
                            c_batch = train_dataset_qc.c_dataset[idx]
                            inputs = get_inputs(c_batch, args.device)
                            outputs = model(**inputs, output_hidden_states=False)
                            c_outputs.append( outputs[0] )
                            s,e = e,min(e+batch_size, context_ids.shape[0])

                        context_emb = torch.cat(c_outputs, dim=0)

                        if rank < 0:
                            # single process calculated all
                            context_emb_all = context_emb
                        else:
                            context_emb_list = [torch.zeros_like(context_emb) for _ in range(world_size)]
                            torch.distributed.all_gather(context_emb_list, context_emb)
                            context_emb_all = torch.cat(context_emb_list, dim=0)

                        hnm_hist[global_step] = (context_ids_all, context_emb_all)

                        #check history size and crop the oldest one
                        if len(hnm_hist) > args.hnm_hist_num:
                            del hnm_hist[min(hnm_hist.keys())]

                    #calc HNM scores for current question batch
                    for hist_step, (c_idx, c_vec) in hnm_hist.items():
                        w = args.hnm_hist_alpha ** (global_step - hist_step)
                        t1 = q_vec[:, None, :]
                        t2 = c_vec[None, :, :]
                        d = (t1 - t2)
                        score = -d.norm(2, dim=-1)
                        score = score * w

                        hnm_scores.append(score)
                        hnm_idxs.append(c_idx)

                    if hnm_scores:
                        #choose the hardest negative if we have scores
                        score = torch.cat(hnm_scores, dim=-1)
                        idx = torch.cat(hnm_idxs, dim=-1)
                        score = score.cpu()
                        pos_mask = (context_ids_pos[:,None] == idx[None,:]).to(dtype=score.dtype, device=score.device)
                        score = (1-pos_mask)*score + pos_mask*score.min() #make positive context with small score to avoid chose it as hard neg
                        hn_idx = score.argmax(dim=1, keepdim=True)

                        context_ids_neg = idx[hn_idx]
                    else:
                        #just random selection in case of no scores for HNM
                        size = (context_ids_pos.shape[0], 1)
                        context_ids_neg = torch.randint(0, len(train_dataset_qc.c_dataset)-1, size)
                        shift = (context_ids_neg >= context_ids_pos[:,None])
                        context_ids_neg = context_ids_neg + shift.to(dtype=context_ids_neg.dtype)

                d_pos = (q_vec - c_vec_pos).norm(2, dim=-1)
                # get negative embeddings and calc losses
                for neg_index in range(context_ids_neg.shape[1]):
                    ids = context_ids_neg[:,neg_index]
                    c_batch = train_dataset_qc.c_dataset[ids.detach()]
                    inputs = get_inputs(c_batch, args.device)

                    outputs = model(**inputs, output_hidden_states=False)
                    c_vec_neg = outputs[0]

                    for triplet_index in range(triplet_num):

                        if triplet_index == 0:
                            d_neg = (q_vec - c_vec_neg).norm(2, dim=-1)
                        if triplet_index == 1:
                            d_neg = (c_vec_pos - c_vec_neg).norm(2, dim=-1)

                        d_diff = d_pos - d_neg

                        indicators['dd'+str(triplet_index)].append([v.mean().item() for v in (d_pos, d_neg, d_diff)])

                        l = softplus(d_diff)
                        losses.append( l )

                        del d_neg
                del d_pos

                #average over batch
                losses = [l.mean() for l in losses]

            l = sum(losses)/len(losses)
            (l/gradient_accumulation_steps).backward()

            indicators['loss'].append(l.item())
            indicators['ll'].append([lll.item() for lll in losses])

            #del losses
            del l

            if (step + 1) % gradient_accumulation_steps == 0:

                utils.sync_grads(rank, named_params, report_no_grad_params=(global_step==1))
                torch.nn.utils.clip_grad_norm_([p for n, p in named_params], 1)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    # Log metrics
                    wall_time = epoch + step / len(q_dataloader)

                    lrp = ['{:.2f}'.format(i) for i in np.log10(scheduler.get_last_lr())]

                    str_out = "{} ep {:.2f} lrp {}".format(train_count, epoch_fp, " ".join(lrp))

                    for k,v in indicators.items():
                        v = np.array(v)
                        if len(v.shape)==1:
                            v = v[:,None]

                        if rank>-1:
                            #sync indicators
                            vt = torch.tensor(v).to(args.device)
                            torch.distributed.all_reduce(vt, op=torch.distributed.ReduceOp.SUM)
                            v = vt.cpu().numpy() / float(world_size)

                        str_out += " {} {}".format(k," ".join(["{:.3f}".format(t) for t in v.mean(0)]))

                    if 'score' in locals():
                        str_out += " SS {}".format(list(score.shape))

                    if 'time_last' in locals():
                        dt_iter = (time.time() - time_last) / len(indicators['loss'])
                        dt_ep = dt_iter * len(q_dataloader)
                        str_out += " it {:.1f}s".format(dt_iter)
                        str_out += " ep {:.1f}m".format(dt_ep / (60))
                        str_out += " eta {:.1f}h".format(dt_ep * (num_train_epochs - epoch_fp) / (60 * 60))
                    time_last = time.time()

                    indicators = collections.defaultdict(list)
                    if rank in [-1, 0]:
                        logger.info(str_out)

        if rank in [-1, 0]:
            check_point_name = 'checkpoint-{:02}'.format(train_count)
            check_point_name = check_point_name + '-{:02}'.format(epoch + 1)
            result_s = evaluate(args, model.eval(), test_dataset_qc)
            for k,v in result_s.items():
                logger.info("{} {} {}".format(check_point_name, k, v))
        if rank>-1:
            torch.distributed.barrier()

def evaluate_embeddings(model, dataset_qc, eval_batch_size):

    def calc_vec(dataset, model):
        device = next(model.parameters()).device

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=eval_batch_size)

        printlog("samples num", len(dataset))
        printlog("eval_batch_size", eval_batch_size)
        start_time = timeit.default_timer()
        model.eval()
        vec = []
        for i,batch in enumerate(dataloader):
            with torch.no_grad():
                inputs = get_inputs(batch, device)
                outputs = model(**inputs, output_hidden_states=False)
                vec.append( outputs[0] )

        vec = torch.cat(vec)

        evalTime = timeit.default_timer() - start_time
        logger.info("Vectors calculation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
        return vec

    #run model for contexts and questions to get embeddings
    c_vec = calc_vec(dataset_qc.c_dataset, model)
    q_vec = calc_vec(dataset_qc.q_dataset, model)

    #calc distances from each question to all contexts
    count_top =  collections.OrderedDict([(k,0) for k in [1,5,10,100,200]])
    top_max = max(count_top.keys())

    for q_index, q_sample in enumerate(dataset_qc.q_dataset):
        context_ids_pos = q_sample[3]
        qv = q_vec[q_index]

        d = c_vec - qv.unsqueeze(0)
        p = -d.norm(2,dim=1)

        val, index = torch.topk(p,top_max,sorted=True)

        #count hits for each topN
        index = index.cpu()
        for top in count_top.keys():
            if context_ids_pos in index[0:top]:
                count_top[top] += 1

    result = collections.OrderedDict(
        [("top{}_neg{}".format(top, len(dataset_qc.c_dataset)-1), count_top[top] / len(dataset_qc.q_dataset)) for top in count_top.keys() ]
    )

    return result

def evaluate(args, model, dataset_qc):
    args.eval_batch_size = args.per_gpu_eval_batch_size
    return evaluate_embeddings(model, dataset_qc, args.eval_batch_size)

def main():
    parser = argparse.ArgumentParser()


    parser.add_argument("--squad_train_data", default=None, type=str, required=True,help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--squad_dev_data", default=None, type=str, required=True,help="SQuAD json for evaluation. E.g., dev-v1.1.json")
    parser.add_argument("--model_student", default="bert-large-uncased-whole-word-masking", type=str, required=False,help="Path to pre-trained model")
    parser.add_argument("--model_teacher", default="bert-large-uncased-whole-word-masking", type=str, required=False,help="Path to pre-trained model for supervision")
    parser.add_argument("--output_dir", default='bert-large-uncased-whole-word-masking-emb-squad', type=str, required=True,help="The output directory for embedding model")
    parser.add_argument("--max_seq_length_q", default=32, type=int,help="The maximum total input sequence length for question")
    parser.add_argument("--max_seq_length_c", default=384, type=int,help="The maximum total input sequence length for context")
    parser.add_argument("--total_train_batch_size", default=32, type=int,help="Batch size to make one optimization step.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,help="Batch size for one GPU inference on train stage.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,help="Batch size for one GPU inference on evaluation stage.")
    parser.add_argument("--learning_rate", default=5e-4, type=float, help="The initial learning rates for Adam.")
    parser.add_argument("--num_train_epochs", default=8.0, type=float,help="Number of epochs to train")
    parser.add_argument("--no_cuda", action='store_true',help="Disable GPU calculation")

    parser.add_argument("--hnm_batch_size", default=32, type=int,help="number of mined hard negatives for one gpu")
    parser.add_argument("--hnm_num", default=256, type=int,help="number of mined hard negatives for each optimization step.")
    parser.add_argument("--hnm_hist_num", default=32, type=int,help="number of mined hard negatives history optimization steps.")
    parser.add_argument("--hnm_hist_alpha", default=1.0, type=float,help="multiplier to increase distance for negatives for older steps.")

    parser.add_argument("--loss_cfg", default="", type=str, help="loss configuration.")
    parser.add_argument("--nncf_config", default=None, type=str, help="config json file for nncf quantization.")
    parser.add_argument("--freeze_list", default="", type=str,help="list of subnames to define model parameters that will not be tuned")
    parser.add_argument("--supervision_weight", default=0, type=float, required=False, help="set to more than 0 to use l2 loss between hidden states")


    args = parser.parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        args.n_gpu = torch.cuda.device_count()
    else:
        args.n_gpu = 0

    for k,v in sorted(vars(args).items(), key=lambda x:x[0]):
        printlog('parameter',k,v)

    if args.n_gpu > 1:
        port = utils.get_free_port()
        printlog("torch.multiprocessing.spawn is started")
        torch.multiprocessing.spawn(process, args=(args,port,), nprocs=args.n_gpu, join=True)
        printlog("torch.multiprocessing.spawn is finished")
    else:
        process(-1, args, 0)


def process(rank, args, port):
    #init multiprocess
    if rank<0:
        args.device = torch.device("cpu" if args.n_gpu < 1 else "cuda")
    else:
        # create default process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)
        torch.distributed.init_process_group("nccl", rank=rank, world_size=args.n_gpu)
        args.device = torch.device("cuda:{}".format(rank))
        torch.cuda.set_device(rank)

    if rank>0:
        #wait while process 0 load models
        torch.distributed.barrier()

    printlog("rank", rank, "load tokenizer", args.model_student)
    tokenizer = BertTokenizer.from_pretrained(args.model_student)

    printlog("rank", rank, "load model", args.model_student)
    config = AutoConfig.from_pretrained(args.model_student)
    if config.architectures and 'BertBasedClassPacked' in config.architectures:
        model = BertPacked(BertModelEMB).from_pretrained(args.model_student).to(args.device)
    else:
        model = BertModelEMB.from_pretrained(args.model_student).to(args.device)

    if args.supervision_weight > 0:
        model_t = BertModelEMB.from_pretrained(args.model_teacher).to(args.device)
    else:
        model_t = None

    if rank==0:
        #wait while other processes load models
        torch.distributed.barrier()

    #create train and evaluate datasets
    train_dataset_qc = create_squad_qcemb_dataset(
        rank,
        args.device,
        args.squad_train_data,
        tokenizer,
        args.max_seq_length_q,
        args.max_seq_length_c
    )

    test_dataset_qc = create_squad_qcemb_dataset(
        rank,
        args.device,
        args.squad_dev_data,
        tokenizer,
        args.max_seq_length_q,
        args.max_seq_length_c
    )

    if rank>=0:
        #lets sync after data loaded
        torch.distributed.barrier()

    model_controller = None
    if QUANTIZATION:

        if hasattr(model,'merge_'):
            #if model is packed, then merge some linera transformations before quantization
            model.merge_()

        if rank in [0,-1]:
            #evaluate before quntization
            model.eval()
            result = evaluate(args, model, test_dataset_qc)
            for n, v in result.items():
                logger.info("original {} - {}".format(n, v))
        if rank >= 0:
            torch.distributed.barrier()

        nncf_config = nncf.NNCFConfig.from_json(args.nncf_config)

        class SquadInitializingDataloader(nncf.initialization.InitializingDataLoader):
            def get_inputs(self, batch):
                return [], get_inputs(batch, args.device)

        train_dataloader = DataLoader(
            train_dataset_qc.c_dataset,
            sampler=RandomSampler(train_dataset_qc.c_dataset),
            batch_size=args.per_gpu_train_batch_size)

        initializing_data_loader = SquadInitializingDataloader(train_dataloader)
        init_range = nncf.initialization.QuantizationRangeInitArgs(initializing_data_loader)
        nncf_config.register_extra_structs([init_range])
        model_controller, model = nncf.create_compressed_model(model, nncf_config, dump_graphs=True)
        if rank>-1:
            model_controller.distributed()
            utils.sync_models(rank, model)

        if rank in [-1, 0]:
            #evaluate pure initialized int8 model
            model.eval()
            result = evaluate(args, model, test_dataset_qc)
            for n, v in result.items():
                logger.info("int8 {} - {}".format(n, v))

        if rank>-1:
            #lets sync after quantization
            torch.distributed.barrier()

        #tune FQ parameters only
        train(rank, args, model, model_t, train_dataset_qc, test_dataset_qc, fq_tune_only=True, model_controller=model_controller)

    #tune whole quantized model
    train(rank, args, model, model_t, train_dataset_qc, test_dataset_qc, fq_tune_only=False, model_controller=model_controller )

    if rank in [-1, 0]:
        #save and evaluate result
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        model.eval()

        #get sample to pass for onnx generation
        with torch.no_grad():
            torch.onnx.export(
                model,
                tuple(torch.zeros((1, args.max_seq_length_c), dtype=torch.long, device=args.device) for t in range(4)),
                os.path.join(args.output_dir, "model.onnx"),
                verbose=False,
                enable_onnx_checker=False,
                opset_version=10,
                input_names=['input_ids', 'attention_mask', 'token_type_ids', 'position_ids'],
                output_names=['embedding'])

        # Evaluate final model
        result = evaluate(args, model, test_dataset_qc)
        for n, v in result.items():
            logger.info("{} - {}".format(n, v))
        logger.info("checkpoint final result {}".format(result))

if __name__ == "__main__":
    main()
