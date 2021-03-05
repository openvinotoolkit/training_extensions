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

import argparse
import logging
import os
import numpy as np
import torch
import math
import time
import utils

from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import LambdaLR

from transformers import BertForQuestionAnswering, BertTokenizer, AutoConfig
from transformers import AdamW

from model_bert_pack import BertPacked
from train_qcemb import BertModelEMB
from train_qcemb import evaluate_embeddings, create_squad_qcemb_dataset
from train_qa import evaluate_qa, create_squad_qa_dataset

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)
logger = logging.getLogger('{} pack_and_distill'.format(os.getpid()))
def printlog(*args):
    logger.info(' '.join([str(v) for v in args]))

def switch_to_train(rank, model):
    model.train()
    printlog('rank',rank, 'enable output_hidden_states and disable output_attentions for train')
    for m in model.modules():
        if hasattr(m,'output_hidden_states'):
            m.output_hidden_states = True
        if hasattr(m, 'output_attentions'):
            m.output_attentions = False
def switch_to_eval(rank, model):
    model.eval()
    printlog('rank', rank, 'disable output_hidden_states and output_attentions for evaluation')
    for m in model.modules():
        if hasattr(m,'output_hidden_states'):
            m.output_hidden_states = False
        if hasattr(m, 'output_attentions'):
            m.output_attentions = False


def save_model(args, model, tokenizer, out_name=None):
    output_dir = os.path.join(args.output_dir, out_name) if out_name else args.output_dir
    logger.info("Saving model checkpoint to %s", output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, 'training_args.txt'), 'wt') as out:
        for k,v in sorted(vars(args).items(),key=lambda x:x[0]):
            out.write("{} {}\n".format(k,v))

def check_model_type(model, class_type):
    if hasattr(model, 'module'):
        model = model.module
    return issubclass(model.__class__, class_type)

train_count=-1
def train(rank, args, tokenizer, train_dataset, test_dataset, model_s, model_t, params_to_tune, head_importance=None, loss_num=-1, tune_iter=0):
    """ Train the model """
    global train_count
    train_count += 1

    world_size = 1 if rank < 0 else torch.distributed.get_world_size()

    if rank in [-1,0]:
        printlog("Train stage: ", train_count )
        printlog(model_s)

    if head_importance is not None:
        head_mask = torch.ones(*list(head_importance.shape)).to(args.device)
        head_mask.requires_grad_(requires_grad=True)
    else:
        head_mask = None

    num_train_epochs = args.num_train_epochs
    if loss_num>0:
        num_train_epochs = 0.25 #short train for incremental loss

    per_gpu_train_batch_size = args.per_gpu_train_batch_size
    train_batch_size = per_gpu_train_batch_size * world_size

    #get total batch size and
    if tune_iter>0 and args.total_train_batch_size_for_tune:
        total_train_batch_size = args.total_train_batch_size_for_tune
    else:
        total_train_batch_size = args.total_train_batch_size
    gradient_accumulation_steps = total_train_batch_size // train_batch_size

    if tune_iter>0 and args.learning_rate_for_tune:
        learning_rate = args.learning_rate_for_tune
    else:
        learning_rate = args.learning_rate

    if check_model_type(model_s, BertModelEMB):
        #use 2 datasets for embedding question and context separatly
        if rank in [-1, 0]:
            printlog("dataset_q size", len(train_dataset.q_dataset))
            printlog("dataset_c size", len(train_dataset.c_dataset))
        datasets = [train_dataset.q_dataset, train_dataset.c_dataset]
    else:
        if rank in [-1, 0]:
            printlog("dataset size", len(train_dataset))
        datasets = [train_dataset]

    if rank>-1:
        #for distributed train use sample that take only part of samples for each process
        train_dataloaders = [
            DataLoader(
                dataset,
                sampler=torch.utils.data.distributed.DistributedSampler(dataset, rank=rank),
                batch_size=per_gpu_train_batch_size)
            for dataset in datasets
        ]
    else:
        train_dataloaders = [
            DataLoader(
                dataset,
                sampler=RandomSampler(dataset),
                batch_size=train_batch_size,
                num_workers = 4)
            for dataset in datasets
        ]

    steps_per_epoch = sum(len(d) for d in train_dataloaders)
    steps_total = int(steps_per_epoch // gradient_accumulation_steps * num_train_epochs)

    # Prepare optimizer and scheduler
    name_set = set()
    for n,p in model_s.named_parameters():
        if any(p is pp for pp in params_to_tune):
            name_set.add(n)
    named_params = [(n,p) for n,p in model_s.named_parameters() if n in name_set]

    if rank in [-1,0]:
        for n,p in named_params:
            printlog('param for tune',n)

    def new_optimizer():
        return AdamW([p for n, p in named_params], lr=learning_rate, eps=1e-08, weight_decay=0.0)
    optimizer = new_optimizer()
    
    def lr_lambda(current_step):
        p = float(current_step) / float(steps_total)
        warmup = 0.01
        if p<warmup:
            return p/warmup
        p = (p-warmup)/(1-warmup)
        return 1 if tune_iter==0 else max(1-p, 0)
    scheduler = LambdaLR(optimizer, lr_lambda)

    if rank in [-1, 0]:
        printlog("epoches", num_train_epochs )
        printlog("per_gpu_train_batch_size", per_gpu_train_batch_size )
        printlog("n_gpu", args.n_gpu )
        printlog("world_size", world_size )
        printlog("gradient_accumulation_steps", gradient_accumulation_steps )
        printlog("total train batch size", train_batch_size * gradient_accumulation_steps )
        printlog("steps_total",steps_total )

    restore_count = 0
    if rank in [-1, 0]:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    restore_file = os.path.join(args.output_dir,'last_good_state.pth')
    restore_loss = None

    losses_list = []

    global_step=0
    for epoch in range(math.ceil(num_train_epochs)):
        switch_to_train(rank, model_t)
        switch_to_train(rank, model_s)
        model_s.zero_grad()
        utils.sync_models(rank, model_s)

        time_last = time.time()
        for train_dataloader in train_dataloaders:
            printlog("rank", rank, "len(train_dataloader)", len(train_dataloader))
            if rank > -1:
                train_dataloader.sampler.set_epoch(epoch)

            if len(train_dataloaders)>1:
                # reset last loss to avoid restore due to dataset changing
                printlog("rank", rank, "reset restore_loss")
                restore_loss = None

            for step, batch in enumerate(train_dataloader):
                epoch_fp = epoch + step/len(train_dataloader)
                if epoch_fp > num_train_epochs:
                    break


                inputs = {
                    'input_ids':       batch[0].to(args.device),
                    'attention_mask':  batch[1].to(args.device),
                    'token_type_ids':  batch[2].to(args.device)
                }

                outputs_s = model_s(**inputs, head_mask=head_mask, output_hidden_states=True)
                losses = []

                with torch.no_grad():
                    outputs_t = model_t(**inputs, output_hidden_states=True)

                out_s, out_t = outputs_s[-1], outputs_t[-1]

                assert len(out_s) == model_s.config.num_hidden_layers+1, "can not find hidden states in student model outputs"
                assert len(out_t) == model_t.config.num_hidden_layers+1, "can not find hidden states in teacher model outputs"
                if len(out_s) != len(out_t):
                    #the student and teacher outputs are not aligned. try to find teacher output for each student output
                    n_s, n_t = len(out_s), len(out_t)
                    out_t = [out_t[(i*(n_t-1))//(n_s-1)] for i in range(n_s)]
                assert len(out_s) == len(out_t), "can not align number of outputs between student and teacher"
                assert all(s[0] == s[1] for s in zip(out_s[0].shape, out_t[0].shape)), "output shapes for student and teacher are not the same"

                out_pairs = list(zip(out_s, out_t))
                if loss_num > 0:
                    out_pairs = out_pairs[:loss_num]

                losses += [(s-t.detach()).pow(2).mean() for s, t in out_pairs]

                losses_list.append([l.item() for l in losses])

                if tune_iter==0:
                    loss = sum(losses) / len(losses)
                else:
                    weights = [args.loss_weight_alpha**i for i in range(len(losses))]
                    losses_w = [w*l for w,l in zip(weights,losses)]
                    loss = sum(losses_w) / sum(weights)


                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                del out_s
                del out_t
                del outputs_s
                del outputs_t

                if head_importance is not None:
                    #collect gradient statistics to find most valuable heads
                    head_mask.grad.detach_()
                    head_importance += (head_mask.grad.abs().detach() - head_importance)*0.001
                    head_mask.grad.zero_()

                if (step + 1) % gradient_accumulation_steps == 0:
                    global_step += 1

                    #sync gradients before calc step
                    utils.sync_grads(rank, named_params, global_step == 1)

                    torch.nn.utils.clip_grad_norm_([p for n, p in named_params], 1)
                    optimizer.step()
                    scheduler.step()

                    model_s.zero_grad()

                    if (step+1) % 50 == 0:
                        str_out = "{} ep {:.2f} lrp {:.2f} rc {:02}".format(train_count, epoch_fp, np.log10(scheduler.get_last_lr()[0]), restore_count)
                        ll = np.array(losses_list).mean(0)

                        if rank>-1:
                            #sync indicators
                            llt = torch.tensor(ll).to(args.device)
                            torch.distributed.all_reduce(llt, op=torch.distributed.ReduceOp.SUM)
                            ll = llt.cpu().numpy() / float(world_size)

                        loss = ll.mean()
                        str_out += " loss {:.4f}".format(loss)
                        losses_txt = ["{:.3f}".format(l) for l in ll]
                        if tune_iter > 0:
                            losses_txt = ["{:.2f}x".format(w) + lt for w,lt in zip(weights,losses_txt)]
                        str_out += " ll " + " ".join(losses_txt)

                        if time_last:
                            dt_iter = (time.time() - time_last) / len(losses_list)
                            dt_ep = dt_iter * steps_per_epoch
                            str_out += " it {:.1f}s".format(dt_iter)
                            str_out += " ep {:.1f}m".format(dt_ep / (60))
                            str_out += " eta {:.1f}h".format(dt_ep * (num_train_epochs - epoch_fp) / (60 * 60))
                        losses_list = []
                        time_last = time.time()
                        if rank in [-1, 0]:
                            logger.info(str_out)

                        if rank>-1:
                            #sync losses
                            loss_tensor = torch.tensor([loss], device=args.device)
                            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
                            loss = loss_tensor.item() / world_size

                        if restore_loss is None or loss < restore_loss*1.5:
                            #good result lets save it
                            restore_loss = loss

                            if rank in [-1, 0]:
                                torch.save({
                                    'model_state_dict': model_s.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict()
                                }, restore_file)
                            if rank>-1:
                                torch.distributed.barrier()
                        else:
                            #bad result lets restore
                            restore_count += 1
                            logger.info("rank {} restore #{} from {} with {} loss".format(rank, restore_count, restore_file, restore_loss))
                            checkpoint = torch.load(restore_file)
                            model_s.load_state_dict(checkpoint['model_state_dict'])
                            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                            optimizer = new_optimizer()
                            switch_to_train(rank, model_s)

        if loss_num <= 0:
            if rank in [-1, 0]:
                check_point_name = 'checkpoint-{:02}'.format(train_count)
                save_model(args, model_s, tokenizer, check_point_name)
                check_point_name = check_point_name+'-{:02}'.format(epoch+1)
                switch_to_eval(rank, model_s)
                result_s = evaluate(args, model_s, test_dataset)
                for k,v in result_s.items():
                    logger.info("{} {} {}".format(check_point_name, k, v))
            if rank>-1:
                torch.distributed.barrier()

    if rank in [-1, 0]:
        if os.path.exists(restore_file):
            os.remove(restore_file)

def evaluate(args, model, squad_dataset ):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    eval_batch_size = args.per_gpu_eval_batch_size
    logger.info("eval_batch_size {}".format(eval_batch_size))
    if check_model_type(model, BertModelEMB):
        return evaluate_embeddings(model, squad_dataset, eval_batch_size)
    else:
        return evaluate_qa(model, squad_dataset, eval_batch_size, args.squad_eval_script)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad_train_data", default=None, type=str, required=True,help="SQuAD json for training. (train-v1.1.json)")
    parser.add_argument("--squad_dev_data", default=None, type=str, required=True,help="SQuAD json for evaluation. (dev-v1.1.json)")
    parser.add_argument("--squad_eval_script", default=None, type=str, required=True,help="SQuAD evaluation script. (evaluate-v1.1.py)")
    parser.add_argument("--model_student", default=None, type=str, required=True,help="Path to pre-trained model")
    parser.add_argument("--model_teacher", default=None, type=str, required=True, help="Path to pre-trained model for supervision")
    parser.add_argument("--output_dir", default=None, type=str, required=True,help="The output directory for packed model")
    parser.add_argument("--max_seq_length_c", default=384, type=int,help="The maximum tokens for context")
    parser.add_argument("--max_seq_length_q", default=64, type=int,help="The maximum tokens for question")
    parser.add_argument("--total_train_batch_size", default=32, type=int,help="Batch size to make one optimization step")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,help="Batch size per GPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,help="Batch size per GPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-4, type=float, help="The learning rates for Adam.")
    parser.add_argument("--num_train_epochs", default=16.0, type=float,help="Number of epochs for one stage train")
    parser.add_argument("--no_cuda", action='store_true',help="Disable GPU calculation")

    parser.add_argument('--seed', type=int, default=42,help="seed for random generators")
    parser.add_argument("--pack_cfg", default="num_hidden_layers:12,ff_iter_num:4,num_attention_heads:8,hidden_size:512,pack_emb:1,hidden_act:orig", type=str,help="string for pack configuration")
    parser.add_argument("--loss_weight_alpha", default=1.5, type=float,help="alpha to define weights for losses on the final tune. w_i = alpha^i. higher weight for layers closer to output")
    parser.add_argument("--total_train_batch_size_for_tune", default=None, type=int,help="Batch size for one optimization step for final tune.")
    parser.add_argument("--learning_rate_for_tune", default=None, type=float, help="The initial learning rates for Adam for final model tune.")

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
        process(-1, args)


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
        torch.cuda.manual_seed_all(args.seed)

    #set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if rank>0:
        #wait while 0 process load models
        torch.distributed.barrier()

    printlog("rank", rank, "load tokenizer", args.model_teacher)
    tokenizer = BertTokenizer.from_pretrained(args.model_student)

    config = AutoConfig.from_pretrained(args.model_student)

    if hasattr(config,'pack_cfg') and 'base_class_name' in config.pack_cfg:
        #get model class from pach_cfg
        base_class_name = config.pack_cfg['base_class_name']
        printlog("rank", rank, "base_class_name to pack", base_class_name)
        Model = globals()[base_class_name]
    else:
        #get model class from architectures filed of config
        if config.architectures:
            assert len(config.architectures) == 1, "only single model is supported but {} has {}".format(args.model_student, config.architectures)
            Model = globals()[config.architectures[0]]
        else:
            Model = BertForQuestionAnswering

    printlog("rank", rank, "load teacher {} model from {}".format(Model.__name__, args.model_teacher))
    model_t = Model.from_pretrained(args.model_teacher)

    printlog("rank", rank, "load student {} model from {}".format(Model.__name__, args.model_student))
    model_s = BertPacked(Model).from_pretrained(args.model_student)


    if rank==0:
        #release other process waiting
        torch.distributed.barrier()

    if rank>-1:
        #sync processes
        torch.distributed.barrier()

    params_packed = []
    if hasattr(model_s.config,'pack_cfg'):
        logger.warning("rank {} !!!model already packed!!!".format(rank))
        logger.warning("rank {} !!!just continue distill the already packed model!!!".format(rank))
    else:
        pack_cfg = dict([t.split(':') for t in args.pack_cfg.split(',')])
        pack_cfg['pack_emb'] = True if eval(pack_cfg['pack_emb']) else False
        printlog("rank", rank, "pack model by", pack_cfg)
        params_packed = model_s.pack_(pack_cfg)

    model_s.to(args.device)
    model_t.to(args.device)

    utils.sync_models(rank, model_s)
    if rank in [-1, 0]:
        save_model(args, model_s, tokenizer)

    def wrap_dropout(net):
        #remove dropout
        class PASS(torch.nn.Module):
            def __init__(self, dropout):
                super().__init__()
                self.dropout = dropout
                self.dropout_enable = False
            def forward(self, x):
                return x
            def __repr__(self):
                return "PASS( dropout_enable {} for {} )".format(self.dropout_enable,self.dropout.__repr__())
        dropout_list = [(n,m,nn,mm) for n,m in net.named_modules() for nn,mm in m._modules.items() if isinstance(mm,torch.nn.Dropout)]
        for n,m,nn,mm in dropout_list:
            m._modules[nn] = PASS(mm)
            logger.info('rank {} {}.{} Dropout in warped by PASS'.format(rank,n,nn))

    logger.info('rank {} warp dropout for teacher model'.format(rank))
    wrap_dropout(model_t)

    logger.info('rank {} warp dropout for student model'.format(rank))
    wrap_dropout(model_s)

    #calculate current number of heads in student model
    bert_s = model_s.get_bert()
    n_layers, n_heads = bert_s.config.num_hidden_layers, bert_s.config.num_attention_heads
    if hasattr(bert_s.config, 'pruned_heads'):
        pruned_nums = [len(v) for v in model_s.config.pruned_heads.values()]
        if pruned_nums:
            n_heads -= min(pruned_nums)

    #load train and evaluation datasets
    if check_model_type(model_s, BertModelEMB):
        train_dataset = create_squad_qcemb_dataset(rank, args.device, args.squad_train_data, tokenizer, args.max_seq_length_q, args.max_seq_length_c)
        test_dataset = create_squad_qcemb_dataset(rank, args.device, args.squad_dev_data, tokenizer, args.max_seq_length_q, args.max_seq_length_c)
    else:
        train_dataset = create_squad_qa_dataset(rank, args.device, args.squad_train_data, tokenizer, args.max_seq_length_q, args.max_seq_length_c)
        test_dataset = create_squad_qa_dataset(rank, args.device, args.squad_dev_data, tokenizer, args.max_seq_length_q, args.max_seq_length_c)

    if rank in [-1,0]:
        switch_to_eval(rank, model_t)
        result_t = evaluate(args, model_t, test_dataset)
        for k,v in result_t.items():
            logger.info("{} teacher {}".format(k, v))
    if rank>-1:
        torch.distributed.barrier()

    params_emb = []
    for n, p in model_s.named_parameters():
        if any(p is pp for pp in params_packed) and 'embedding' in n:
            params_emb.append(p)

    if params_emb:
        params_inp = [p for n, p in model_s.named_parameters() if 'input_transform' in n ]

        #tune embeddings transformation
        params_tune = params_emb + params_inp
        loss_num = 1
        train(
            rank,
            args,
            tokenizer,
            train_dataset,
            test_dataset,
            model_s,
            model_t,
            params_tune,
            head_importance=None,
            loss_num=loss_num)

        #iterative add bert encoder blocks
        encoder = model_s.get_bert().encoder
        for l,t in zip(encoder.layer, encoder.output_transforms):
            params_tune.extend(l.parameters())
            params_tune.extend(t.parameters())
            loss_num += 1
            train(
                rank,
                args,
                tokenizer,
                train_dataset,
                test_dataset,
                model_s,
                model_t,
                params_tune,
                head_importance=None,
                loss_num=loss_num)

    if params_packed:
        #on the first stage the FF block only reduced and tuned
        #the number of self attention heads is the same

        #check that head prune is needed and run second train to tune the rest heads
        pack_head_num = int( model_s.config.pack_cfg.get('num_attention_heads',n_heads) )
        pack_heads_flag = (pack_head_num < n_heads)
        head_importance = torch.zeros(n_layers, n_heads).to(args.device) if pack_heads_flag else None

        params_ff = [p for n, p in model_s.named_parameters() if 'encoder.' in n and 'attention.' not in n]

        train(
            rank,
            args,
            tokenizer,
            train_dataset,
            test_dataset,
            model_s,
            model_t,
            params_packed + params_ff,
            head_importance=head_importance)

        if head_importance is not None and rank>-1:
            torch.distributed.all_reduce(head_importance.data, op=torch.distributed.ReduceOp.SUM)

        if pack_heads_flag:
            #reduce number of heads before move to the second stage and tune all model
            if rank in [-1, 0]:
                logger.info('head_importance')
                logger.info(head_importance)
                logger.info('heads_to_prune')

            #prune heads
            heads_to_prune = {}
            for l in range(n_layers):
                imp = head_importance[l].tolist()
                idx = list(sorted(range(n_heads), key=lambda x:imp[x]))
                heads_to_prune[l] = idx[:-pack_head_num]
                if rank in [-1, 0]:
                    logger.info( "layer {} heads_to_prune {}".format(l, heads_to_prune[l]) )
            model_s.prune_heads(heads_to_prune)
            utils.sync_models(rank, model_s)

        params_encoder = [p for n, p in model_s.named_parameters() if 'encoder.' in n]
        params_emb = [p for n, p in model_s.named_parameters() if 'embedding' in n and 'linear' in n]
        if params_emb:
            # if has linear then LayerNorm was trained
            params_emb += [p for n, p in model_s.named_parameters() if 'embedding' in n and 'LayerNorm' in n]
        train(
            rank,
            args,
            tokenizer,
            train_dataset,
            test_dataset,
            model_s,
            model_t,
            params_emb + params_encoder)

    params_encoder = [p for n, p in model_s.named_parameters() if 'encoder.' in n]
    params_emb = [p for n, p in model_s.named_parameters() if 'embedding' in n and 'linear' in n]
    if params_emb:
        #if has linear then LayerNorm was trained
        params_emb += [p for n, p in model_s.named_parameters() if 'embedding' in n and 'LayerNorm' in n]

    #final tune
    train(
        rank,
        args,
        tokenizer,
        train_dataset,
        test_dataset,
        model_s,
        model_t,
        params_emb + params_encoder,
        tune_iter=1)

    if rank in [-1,0]:
        save_model(args, model_s, tokenizer)

        logger.info('Evaluate student model')
        logger.info('Model for evaluation')
        logger.info(model_s)
        switch_to_eval(rank, model_s)
        result_s = evaluate(args, model_s, test_dataset)
        for k,v in result_s.items():
            logger.info("{} student {} teacher {}".format(k,v,result_t[k]))

        #merge some linear transformations into filters
        model_s.merge_()

        logger.info("student model")
        logger.info(model_s)
        result_s = evaluate(args, model_s, test_dataset)
        for k,v in result_s.items():
            logger.info("{} student {} after some operations are merged".format(k,v))

        #save to onnx
        if check_model_type(model_s, BertModelEMB):
            output_names = ['embedding']
        else:
            output_names = ['output_s', 'output_e']
        inputs = tuple(torch.zeros(args.max_seq_length_q, dtype=torch.long) for t in range(4))
        inputs = tuple(t.unsqueeze(0).to(args.device) for t in inputs)
        torch.onnx.export(
            model_s,
            inputs,
            os.path.join(args.output_dir, "model.onnx"),
            verbose=False,
            input_names=['input_ids', 'attention_mask', 'token_type_ids', 'position_ids'],
            output_names=output_names)



if __name__ == "__main__":
    main()
