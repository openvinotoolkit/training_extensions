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
import os
import random
import importlib
import time
import numpy as np
import math
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)
logger = logging.getLogger('{} train_qa'.format(os.getpid()))
def printlog(*args):
    logger.info(' '.join([str(v) for v in args]))

import torch
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler)

#nncf has to be imported right after torch
QUANTIZATION = any('nncf_config' in t for t in sys.argv[1:])
printlog('QUANTIZATION', QUANTIZATION)
if QUANTIZATION:
    printlog('import nncf')
    import nncf

from torch.optim.lr_scheduler import LambdaLR
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import AdamW
from transformers import AutoConfig
from model_bert_pack import BertPacked

import utils

#this function is needed to support old transformers model
def set_output_hidden_states(rank, model, flag):
    printlog('rank',rank,'set output_hidden_states={} and disable output_attentions'.format(flag))
    for m in model.modules():
        if hasattr(m,'output_hidden_states'):
            m.output_hidden_states = flag
        if hasattr(m, 'output_attentions'):
            m.output_attentions = False

def get_inputs(batch, device):
    return {
        'input_ids':       batch[0].to(device),
        'attention_mask':  batch[1].to(device),
        'token_type_ids':  batch[2].to(device)
    }

def get_targets(batch, device):
    return {
        'start_positions':  batch[4].to(device),
        'end_positions':    batch[5].to(device),
    }


def create_squad_qa_dataset(rank, device, squad_file, tokenizer, max_query_length, max_seq_length):

    squad = utils.squad_read_and_encode(rank, device, squad_file, tokenizer)

    samples_by_index = []
    for art_i, article in enumerate(squad['data']):
        for par_i, par in enumerate(article['paragraphs']):
            for qa_i, qa in enumerate(par['qas']):
                #paragraph context could be larger than max_context_length
                #then it have to be splitted with stride

                len_q = min(max_query_length, len(qa['question_enc']))
                len_c = len(par['context_enc'])
                max_context_length = max_seq_length - (len_q + 3)
                c_s, c_e = 0, min(max_context_length, len_c)
                c_stride = 128
                while c_e > c_s:
                    samples_by_index.append((art_i, par_i, qa_i, qa['id'], c_s, c_e))
                    # check that context window reach the end position
                    if c_e == len_c:
                        break
                    # move to next window position
                    c_s, c_e = c_s + c_stride, c_e + c_stride
                    #if end out of context then move window back
                    shift_left = max(0, c_e - len_c)
                    c_s, c_e = c_s - shift_left, c_e - shift_left
                    assert c_s >= 0, "start can be left of 0 only with window less than len but in this case we can not be here"

    class QADataset(torch.utils.data.Dataset):
        def get_squad(self):
            return squad

        def get_text_and_qid(self, sample_i, tok_s, tok_e):
            art_i, par_i, qa_i, qid, c_s, c_e = samples_by_index[sample_i]
            par = squad['data'][art_i]['paragraphs'][par_i]
            pos = par['context_enc_pos']
            s, e = pos[tok_s+c_s][0], pos[tok_e+c_s][1]
            return par['context'][s:e], qid

        def get_vocab(self):
            return tokenizer.vocab

        def __getitem__(self, sample_i):
            art_i, par_i, qa_i, qid, c_s, c_e = samples_by_index[sample_i]
            par = squad['data'][art_i]['paragraphs'][par_i]
            ids_c = par['context_enc'][c_s:c_e]
            pos_c = par['context_enc_pos'][c_s:c_e]
            ids_q = par['qas'][qa_i]['question_enc']

            len_q = min(max_query_length, len(ids_q))
            ids_q = ids_q[:len_q]
            len_c = len(ids_c)

            rest = max_seq_length - (len_q + len_c + 3)
            assert rest>=0

            pad = [tokenizer.vocab["[PAD]"]]
            cls = [tokenizer.vocab["[CLS]"]]
            sep = [tokenizer.vocab["[SEP]"]]

            input_ids = torch.tensor(cls + ids_q + sep + ids_c + sep + pad * rest, dtype=torch.long)
            input_mask = torch.tensor([1] * (1 + len_q + 1) + [1] * (len_c + 1) + pad * rest, dtype=torch.long)
            segment_ids = torch.tensor([0] * (1 + len_q + 1) + [1] * (len_c + 1) + pad * rest, dtype=torch.long)

            start_pos = 0
            end_pos = 0
            answers = par['qas'][qa_i]['answers']
            if answers:
                ans = random.choice(answers)
                s_i = ans['answer_start']
                e_i = s_i + len(ans['text'])-1
                start_pos_list = [i for i,p in enumerate(pos_c) if s_i>=p[0] and s_i<p[1]]
                end_pos_list   = [i for i,p in enumerate(pos_c) if e_i>=p[0] and e_i<p[1]]
                if start_pos_list and end_pos_list:
                    start_pos = start_pos_list[0] + (1 + len_q + 1)
                    end_pos   = end_pos_list[-1]  + (1 + len_q + 1)
            return (input_ids, input_mask, segment_ids, sample_i, start_pos, end_pos)

        def __len__(self):
            return len(samples_by_index)

    return QADataset()

train_count=-1
def train(rank, args, model, model_t, train_dataset_qa, test_dataset_qa, fq_tune_only, model_controller):
    """ Train the model """
    global train_count
    train_count += 1
    world_size = 1 if rank < 0 else torch.distributed.get_world_size()

    if rank in [-1, 0]:
        printlog("Train model",train_count)
        printlog(model)

    per_gpu_train_batch_size = args.per_gpu_train_batch_size
    train_batch_size = per_gpu_train_batch_size * world_size
    gradient_accumulation_steps = args.total_train_batch_size // train_batch_size
    num_train_epochs = args.num_train_epochs

    if fq_tune_only:
        gradient_accumulation_steps = 1
        num_train_epochs = 1

    if rank < 0:
        #single process take all samples
        sampler = RandomSampler(train_dataset_qa)
        dataloader = DataLoader(train_dataset_qa, sampler=sampler, batch_size=train_batch_size, num_workers=4)
    else:
        #special sampler that divide samples beween processes
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_qa, rank=rank)
        dataloader = DataLoader(train_dataset_qa, sampler=sampler, batch_size=per_gpu_train_batch_size)

    steps_total = int(len(dataloader) // gradient_accumulation_steps * num_train_epochs)

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
        printlog("dataset size", len(train_dataset_qa) )
        printlog("epoches", num_train_epochs )
        printlog("per_gpu_train_batch_size", per_gpu_train_batch_size )
        printlog("n_gpu", args.n_gpu )
        printlog("world_size", world_size )
        printlog("gradient_accumulation_steps", gradient_accumulation_steps )
        printlog("total train batch size", train_batch_size * gradient_accumulation_steps )
        printlog("steps_total",steps_total )

    global_step = 0
    model.zero_grad()
    indicators = collections.defaultdict(list)

    softplus = torch.nn.Softplus()

    loss_cfg = dict([t.split(':') for t in args.loss_cfg.split(',')]) if args.loss_cfg else dict()

    for epoch in range(math.ceil(num_train_epochs)):
        indicators = collections.defaultdict(list)
        model.train()
        set_output_hidden_states(rank, model, (model_t is not None))
        utils.sync_models(rank, model)
        if model_t is not None:
            set_output_hidden_states(rank, model_t, True)
            model_t.train()
        if rank > -1:
            #set epoch to make different samples division betwen process for different epoches
            sampler.set_epoch(epoch)

        for step, batch in enumerate(dataloader):
            epoch_fp = epoch + step/len(dataloader)
            if epoch_fp > num_train_epochs:
                break

            epoch_fp = epoch + step/len(dataloader)

            losses = []

            inputs = get_inputs(batch, args.device)
            targets = get_targets(batch, args.device)
            outputs = model(**inputs, **targets, output_hidden_states=(model_t is not None))
            losses.append(outputs[0])
            outputs = outputs[1:]

            if model_t is not None:
                with torch.no_grad():
                    outputs_t = model_t(**inputs, output_hidden_states=True)
                    hidden_t = outputs_t[2]
                    assert isinstance(hidden_t, (tuple,list)), "hidden states output is not detected right"
                    assert len(hidden_t) == model_t.config.num_hidden_layers+1, "hidden states output is not detected right"

                if args.kd_weight>0:
                    # Calculate knowladge distilation loss
                    kd_losses = []
                    for logit_s,logit_t in zip(outputs[0:2],outputs_t[0:2]):
                        T = 1
                        prob_t = torch.nn.functional.softmax(logit_t.detach() / T, dim=1)
                        logprob_s = torch.nn.functional.log_softmax(logit_s / T, dim=1)
                        kd_losses.append( -(logprob_s * prob_t).mean() * (T * T * prob_t.shape[1]) )
                    losses.append(args.kd_weight*sum(kd_losses)/len(kd_losses))


                hidden_s = outputs[2]
                assert isinstance(hidden_s, (tuple,list)), "hidden states output is not detected right"
                assert len(hidden_s) == model.config.num_hidden_layers+1, "hidden states output is not detected right"

                def align_and_loss_outputs(out_s, out_t):
                    if len(out_s) != len(out_t):
                        #the student and teacher outputs are not aligned. try to find teacher output for each student output
                        n_s, n_t = len(out_s), len(out_t)
                        out_t = [out_t[(i*(n_t-1))//(n_s-1)] for i in range(n_s)]
                    assert len(out_s) == len(out_t), "can not align number of outputs between student and teacher"
                    assert all(s[0] == s[1] for s in zip(out_s[0].shape, out_t[0].shape)), "output shapes for student and teacher are not the same"
                    return [(s - t.detach()).pow(2).mean() for s,t in zip(out_s, out_t)]

                sw_losses = align_and_loss_outputs(hidden_s,hidden_t)

                losses.extend([args.supervision_weight*l for l in sw_losses])

            #average over batch
            losses = [l.mean() for l in losses]

            l = sum(losses)/len(losses)
            indicators['loss'].append(l.item())
            indicators['ll'].append([lll.item() for lll in losses])

            (l/gradient_accumulation_steps).backward()

            del l

            if (step + 1) % gradient_accumulation_steps == 0:
                global_step += 1

                utils.sync_grads(rank, named_params, report_no_grad_params=(global_step==1))
                torch.nn.utils.clip_grad_norm_([p for n, p in named_params], 1)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()


                if global_step % 50 == 0:
                    # Log metrics
                    wall_time = epoch + step / len(dataloader)

                    lrp = " ".join(['{:.2f}'.format(t) for t in np.log10(scheduler.get_last_lr())])
                    str_out = "{} ep {:.2f} lrp {}".format(train_count, epoch_fp, lrp)

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


                    if 'time_last' in locals():
                        #estimate processing times
                        dt_iter = (time.time() - time_last) / len(indicators['loss'])
                        dt_ep = dt_iter * len(dataloader)
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
            model.eval()
            set_output_hidden_states(rank, model, False)
            result_s = evaluate(args, model, test_dataset_qa)
            for k,v in result_s.items():
                logger.info("{} {} {}".format(check_point_name, k, result_s[k]))
        if rank>-1:
            torch.distributed.barrier()

def evaluate(args, model, dataset_qa):

    return evaluate_qa(
        model,
        dataset_qa,
        args.per_gpu_eval_batch_size,
        args.squad_eval_script
    )

def evaluate_qa(model, squad_dataset, eval_batch_size, squad_eval_script ):
    max_answer_length = 30

    device = next(model.parameters()).device

    logger.info("eval_batch_size {}".format(eval_batch_size))

    data_loader = DataLoader(
        squad_dataset,
        sampler=SequentialSampler(squad_dataset),
        batch_size=eval_batch_size)

    logger.info("samples num {}".format(len(squad_dataset)))
    answers = collections.OrderedDict()
    answers_score = collections.OrderedDict()
    no_answers_score = collections.OrderedDict()

    for batch_i, batch in enumerate(data_loader):
        inputs = get_inputs(batch, device)

        with torch.no_grad():
            res = model(**inputs)
            score_s = torch.nn.functional.softmax(res[0], dim=-1).cpu().numpy()
            score_e = torch.nn.functional.softmax(res[1], dim=-1).cpu().numpy()

        for i in range(score_s.shape[0]):
            tokens, sample_i = batch[0][i], batch[3][i]
            ss, se = score_s[i], score_e[i]
            sep = squad_dataset.get_vocab()['[SEP]']

            # find product of all start-end combinations to find the best one
            sep_pos = [i for i,t in enumerate(tokens) if t == sep]
            c_slice = slice(sep_pos[0]+1, sep_pos[1])
            score_mat = np.matmul(
                ss[c_slice][:,None],
                se[c_slice][None,:]
            )
            # reset candidates with end before start
            score_mat = np.triu(score_mat)
            # reset long candidates (>max_answer_token_num)
            score_mat = np.tril(score_mat, max_answer_length - 1)
            # find the best start-end pair
            max_s, max_e = divmod(score_mat.flatten().argmax(), score_mat.shape[1])

            score = score_mat[max_s, max_e]
            pred, qid = squad_dataset.get_text_and_qid(sample_i, max_s, max_e)


            if qid not in answers or score > answers_score[qid]:
                answers[qid] = pred
                answers_score[qid] = score
                no_answers_score[qid] = (se[0] * ss[0]) / score

    dataset = squad_dataset.get_squad()['data']
    dataset_ver = squad_dataset.get_squad()['version']
    flag_squad_v2 = ('2' in dataset_ver.split('.')[0])
    logger.info("eval dataset ver {} flag_squad_v2 {}".format(dataset_ver,flag_squad_v2))

    #get evaluate squad script to get official numbers
    spec = importlib.util.spec_from_file_location('squad_evaluate', squad_eval_script)
    squad_evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(squad_evaluate)

    if hasattr(squad_evaluate,'evaluate'):
        logger.info("eval by v1 script {}".format((squad_eval_script)))
        if flag_squad_v2:
            msg = "evaluate script {} does not support squad2 dataset".format(squad_eval_script)
            logger.error(msg)
            raise Exception(msg)
        res = squad_evaluate.evaluate(dataset, answers)
    else:
        logger.info("eval by v2 script {}".format((squad_eval_script)))
        exact_raw, f1_raw = squad_evaluate.get_raw_scores(dataset, answers)

        if not flag_squad_v2:
            res = squad_evaluate.make_eval_dict(exact_raw, f1_raw)
        else:
            # find (se[0] * ss[0]) / score > 1
            qid_to_has_ans = squad_evaluate.make_qid_to_has_ans(dataset)
            exact_thresh = squad_evaluate.apply_no_ans_threshold(exact_raw, no_answers_score, qid_to_has_ans,1)
            f1_thresh = squad_evaluate.apply_no_ans_threshold(f1_raw, no_answers_score, qid_to_has_ans,1)
            res = squad_evaluate.make_eval_dict(exact_thresh, f1_thresh)

            # find the best threshold for (se[0] * ss[0]) / score
            squad_evaluate.find_all_best_thresh(res, answers, exact_raw, f1_raw, no_answers_score, qid_to_has_ans)
    return res



def main():
    parser = argparse.ArgumentParser()


    parser.add_argument("--squad_train_data", default=None, type=str, required=True,help="SQuAD json for training. (train-v1.1.json)")
    parser.add_argument("--squad_dev_data", default=None, type=str, required=True,help="SQuAD json for evaluation. (dev-v1.1.json)")
    parser.add_argument("--squad_eval_script", default=None, type=str, required=True,help="SQuAD evaluation script. (evaluate-v1.1.py)")
    parser.add_argument("--model_student", default="bert-large-uncased-whole-word-masking", type=str, required=False,help="Path to pre-trained model")
    parser.add_argument("--model_teacher", default="bert-large-uncased-whole-word-masking", type=str, required=False,help="Path to pre-trained model for supervision")
    parser.add_argument("--output_dir", default='bert-large-uncased-whole-word-masking-qa-squad', type=str, required=True,help="The output directory for embedding model")
    parser.add_argument("--total_train_batch_size", default=48, type=int,help="Batch size to make one optimization step.")
    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,help="Batch size per GPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int,help="Batch size per GPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rates for Adam.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,help="Number of epochs for one stage train")
    parser.add_argument("--no_cuda", action='store_true',help="Disable GPU calculation")
    parser.add_argument("--max_seq_length_q", default=64, type=int,help="The maximum total input sequence length for question")
    parser.add_argument("--max_seq_length_c", default=384, type=int,help="The maximum total input sequence length for context + question")

    parser.add_argument("--supervision_weight", default=0.02, type=float, required=False, help="set to more than 0 to use l2 loss between hidden states")
    parser.add_argument("--kd_weight", default=1, type=float, required=False, help="set to more than 0 to use kd loss between output logits")

    parser.add_argument("--loss_cfg", default="", type=str,help="loss type.")
    parser.add_argument("--nncf_config", default=None, type=str,help="config json file for quantization by nncf.")
    parser.add_argument("--freeze_list", default="", type=str,help="list of subnames to define parameters that will not be tuned")


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
        printlog("single process mode")
        process(-1, args, None)

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
        #wait while 0 process load models
        torch.distributed.barrier()

    printlog("rank", rank, "load tokenizer", args.model_student)
    tokenizer = BertTokenizer.from_pretrained(args.model_student)

    printlog("rank", rank, "load model", args.model_student)
    config = AutoConfig.from_pretrained(args.model_student)
    if config.architectures and 'BertBasedClassPacked' in config.architectures:
        model = BertPacked(BertForQuestionAnswering).from_pretrained(args.model_student).to(args.device)
    else:
        model = BertForQuestionAnswering.from_pretrained(args.model_student).to(args.device)

    if args.supervision_weight > 0:
        model_t = BertForQuestionAnswering.from_pretrained(args.model_teacher).to(args.device)
    else:
        model_t = None

    if rank==0:
        #release other process waiting
        torch.distributed.barrier()

    if rank>-1:
        #sync processes
        torch.distributed.barrier()

    #create train and evaluate datasets
    train_dataset_qa = create_squad_qa_dataset(
        rank,
        args.device,
        args.squad_train_data,
        tokenizer,
        args.max_seq_length_q,
        args.max_seq_length_c
    )

    test_dataset_qa = create_squad_qa_dataset(
        rank,
        args.device,
        args.squad_dev_data,
        tokenizer,
        args.max_seq_length_q,
        args.max_seq_length_c
    )

    if rank>-1:
        #lets sync after data loaded
        torch.distributed.barrier()

    model_controller = None
    if QUANTIZATION:

        if hasattr(model,'merge_'):
            #if model is packed then merge some linera transformations before quantization
            model.merge_()

        if rank in [0, -1]:
            # Evaluate
            model.eval()
            result = evaluate(args, model, test_dataset_qa)
            for n, v in result.items():
                logger.info("original {} - {}".format(n, v))

        if rank > -1:
            #lets sync after evaluation
            torch.distributed.barrier()

        nncf_config = nncf.NNCFConfig.from_json(args.nncf_config)

        class SquadInitializingDataloader(nncf.initialization.InitializingDataLoader):
            def get_inputs(self, batch):
                return [], get_inputs(batch, args.device)

        train_dataloader = DataLoader(
            train_dataset_qa,
            sampler=RandomSampler(train_dataset_qa),
            batch_size=args.per_gpu_train_batch_size)
        initializing_data_loader = SquadInitializingDataloader(train_dataloader)
        init_range = nncf.initialization.QuantizationRangeInitArgs(initializing_data_loader)
        nncf_config.register_extra_structs([init_range])
        print(nncf_config)

        model_controller, model = nncf.create_compressed_model(model, nncf_config, dump_graphs=True)
        if rank>-1:
            model_controller.distributed()


        if True and rank in [-1, 0]:
            model.eval()
            set_output_hidden_states(rank, model, False)
            result = evaluate(args, model, test_dataset_qa)
            for n, v in result.items():
                logger.info("quantized {} - {}".format(n, v))

        if rank > -1:
            #lets sync after quantization
            torch.distributed.barrier()

        #tune quntization layers parameters only
        train(
            rank,
            args,
            model, model_t,
            train_dataset_qa, test_dataset_qa,
            fq_tune_only=True,
            model_controller=model_controller)

    #tune all parameters
    train(
        rank,
        args,
        model, model_t,
        train_dataset_qa, test_dataset_qa,
        fq_tune_only=False,
        model_controller=model_controller)

    model.eval()
    set_output_hidden_states(rank, model, False)

    if rank in [-1, 0]:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        with torch.no_grad():
            # get sample to pass for onnx generation
            os.makedirs(args.output_dir, exist_ok=True)
            torch.onnx.export(
                model,
                tuple(torch.zeros((1, args.max_seq_length_c), dtype=torch.long, device=args.device) for t in range(4)),
                os.path.join(args.output_dir, "model.onnx"),
                verbose=False,
                enable_onnx_checker=False,
                opset_version=10,
                input_names=['input_ids', 'attention_mask', 'token_type_ids', 'position_ids'],
                output_names=['output_s', 'output_e'])

        # Evaluate
        result = evaluate(args, model, test_dataset_qa)
        for n, v in result.items():
            logger.info("{} - {}".format(n, v))
        logger.info("checkpoint {} result {}".format("final", result))

    if rank > -1:
        #lets sync after quantization
        torch.distributed.barrier()

if __name__ == "__main__":
    main()
