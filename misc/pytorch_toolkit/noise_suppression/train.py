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

import collections
import logging
import os
import time
import argparse
import math
import random
import re

import numpy as np
import torch

import utils
import models
import dataset
from metrics import sisdr
from evaluate import evaluate_dir

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)
logger = logging.getLogger('{} train_poconetlike'.format(os.getpid()))
def printlog(*args):
    logger.info(' '.join([str(v) for v in args]))

EPS = torch.finfo(torch.float32).tiny

class CheckLossRaise():
    def __init__(self):
        self.loss_mean = 0
        self.loss_dev = 0
        self.loss_count = 0
        self.reset_count = 0

    def check(self, loss_val, named_params, optimizer):
        # check 3 sigma
        loss_sigma = self.loss_dev ** 0.5
        loss_raise = loss_val - self.loss_mean
        if self.loss_count < 10 or loss_raise < 3 * loss_sigma:
            #save params for future backup
            self.named_params_backup = [(n, p.detach().cpu()) for n, p in named_params]
        else:

            # reset params and optimizer
            self.reset_count += 1
            logger.info("{} > 3*{} RESET PARAMS {}".format(loss_raise,loss_sigma,self.reset_count))
            #restore params from backup
            for (n, p), (nb, pb) in zip(named_params, self.named_params_backup):
                assert n == nb
                p.data.copy_(pb.data)
            #reset optimizer
            optimizer.state = collections.defaultdict(dict)

        self.loss_mean += (0.1 if self.loss_count>0 else 1) * loss_raise
        self.loss_dev +=  (0.1 if self.loss_count>1 else 1) * (loss_raise ** 2 - self.loss_dev)
        self.loss_count += 1


class Train():
    def __init__(self, rank, args, model, dataset_train):
        self.rank = rank
        self.args = args
        self.world_size = 1 if self.rank < 0 else torch.distributed.get_world_size()
        self.time_last = None
        self.model = model

        #create datasetloader
        if self.rank < 0:
            # single process take all samples
            sampler = torch.utils.data.RandomSampler(dataset_train)
        else:
            # special sampler that divide samples between processes
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_train,
                rank=self.rank,
                drop_last=True,
                shuffle=True)

        self.dataloader = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler,
            batch_size=self.args.per_gpu_train_batch_size,
            num_workers=3)

        #calc some batch related parameters
        self.train_batch_size = self.args.per_gpu_train_batch_size * self.world_size
        self.gradient_accumulation_steps = max(1, self.args.total_train_batch_size // self.train_batch_size)
        self.total_train_batch_size = self.train_batch_size * self.gradient_accumulation_steps
        self.steps_total = int((len(self.dataloader) // self.gradient_accumulation_steps) * self.args.num_train_epochs)

        #get list of parameters from model
        self.named_params = list(model.named_parameters())

        #create optimizer
        params = [p for n, p in self.named_params]
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay)

        #create scheduler
        def lr_lambda(current_step):
            const_time = 0.5
            p = float(current_step) / (float(self.steps_total) + EPS)
            if p <= const_time:
                return 1
            p = (p - const_time) / (1 - const_time)
            return max(0, math.cos(math.pi * 0.5 * p))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        #print some train information
        if self.rank in [-1, 0]:
            printlog("Train model")
            printlog(model)
            for n, p in self.named_params:
                printlog('param for tune', n, list(p.shape))
            printlog("dataset_train size", len(dataset_train))
            printlog("epoches", self.args.num_train_epochs)
            printlog("per_gpu_train_batch_size", self.args.per_gpu_train_batch_size)
            printlog("n_gpu", self.args.n_gpu)
            printlog("world_size", self.world_size)
            printlog("gradient_accumulation_steps", self.gradient_accumulation_steps)
            printlog("total train batch size", self.total_train_batch_size)
            printlog("steps_total", self.steps_total)

    def aver_indicators(self):
        self.indicators_mean = {}
        for k, v in self.indicators.items():
            v = np.array(v)
            if len(v.shape) == 1:
                v = v[:, None]

            if self.rank > -1:
                # sync indicators
                world_size = torch.distributed.get_world_size()
                vt = torch.tensor(v).to(self.args.device)
                torch.distributed.all_reduce(vt, op=torch.distributed.ReduceOp.SUM)
                v = vt.cpu().numpy() / float(world_size)

            self.indicators_mean[k] = v.mean(0)

    def log_indicators(self, epoch_fp):
        # Log metrics
        str_out = ["ep {:.3f}".format(epoch_fp)]

        lr = self.scheduler.get_last_lr()
        str_out.append("lrp " + " ".join(['{:.2f}'.format(math.log10(t + EPS)) for t in lr]))

        # add all indicators into log string
        def tostr(v):
            return " ".join(["{:.3f}".format(t) for t in v])
        str_out.extend(["{} {}".format(k, tostr(v)) for k, v in self.indicators_mean.items()])

        str_out.append("RC {}".format(self.check_loss_raise.reset_count))

        # estimate processing times
        if self.time_last is not None:
            dt_iter = (time.time() - self.time_last) / len(self.indicators['loss'])
            dt_ep = dt_iter * len(self.dataloader)
            str_out.append("it {:.1f}s".format(dt_iter))
            str_out.append("ep {:.1f}m".format(dt_ep / (60)))
            str_out.append("eta {:.1f}h".format(dt_ep * (self.args.num_train_epochs - epoch_fp) / (60 * 60)))
        self.time_last = time.time()

        if self.rank in [-1, 0]:
            logger.info(" ".join(str_out))

    def save_and_eval_checkpoint(self, epoch):
        if self.rank in [-1, 0]:
            check_point_name = 'checkpoint-{:02}'.format(epoch)
            output_dir = os.path.join(self.args.output_dir, check_point_name)
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            evaluate_dir(self.args.eval_data, output_dir)

        if self.rank > -1:
            torch.distributed.barrier()

    def mix_signals(self, x_clean, x_noise):
        # mix clean and noise signals on GPU
        rms_noise = x_noise.pow(2).mean(-1, keepdim=True).sqrt()
        rms_clean = x_clean.pow(2).mean(-1, keepdim=True).sqrt()

        def rand_range(a, b):
            return a + (b - a) * torch.rand(rms_clean.shape, dtype=rms_clean.dtype, device=rms_clean.device)

        snr_db = rand_range(self.args.snr_min, self.args.snr_max)
        target_db = rand_range(self.args.target_min, self.args.target_max)

        def db_to_scale(db):
            return 10 ** (db / 20)

        # scale noise to get given SNR
        scale_noise = db_to_scale(-snr_db) * rms_clean / (rms_noise + EPS)

        x = x_clean + x_noise * scale_noise
        rms_x = x.pow(2).mean(-1, keepdim=True).sqrt()

        target_scale = db_to_scale(target_db) / (rms_x + EPS)
        x = x * target_scale
        x_clean = x_clean * target_scale

        # randomply clip signal
        t = torch.rand(target_scale.shape, dtype=target_scale.dtype, device=target_scale.device)
        clamp_mask = (t < self.args.clip_prob).float()
        # clamp_mask = clamp_mask.float()

        t = torch.rand(target_scale.shape, dtype=target_scale.dtype, device=target_scale.device)
        max_val = x.abs()
        max_val, _ = max_val.max(-1, keepdim=True)
        clamp_val = max_val * (1 - clamp_mask * t * 0.5)
        x = torch.max(x, 0 - clamp_val)
        x = torch.min(x, clamp_val)
        return x_clean, x_noise, x

    def loss(self, epoch_fp, y_clean, Y_clean, x_clean, X_clean):
        # calc losses
        losses = []

        def add_loss(name, val):
            self.indicators[name].append(val.item())
            losses.append(val)

        add_loss("Lri", 100 * torch.nn.functional.l1_loss(Y_clean, X_clean))

        add_loss("negsisdr", -sisdr(y_clean, x_clean).mean())

        if epoch_fp < 1:
            # [B,2,F,T] -> [2,B,F,T]
            Y = Y_clean.transpose(0, 1)
            X = X_clean.transpose(0, 1)
            norm_x = (X[0] * X[0] + X[1] * X[1] + EPS).sqrt().sum()
            norm_y = (Y[0] * Y[0] + Y[1] * Y[1] + EPS).sqrt()
            scale_x = (norm_x + EPS).reciprocal()
            scale_y = (norm_y + EPS).reciprocal()
            # [B, F, T]
            cosph = ((Y[0] * X[0] + Y[1] * X[1]) * scale_y).sum() * scale_x
            add_loss("cosph", -100 * cosph.mean())

        return sum(losses)

    def train(self, epoch_start):

        global_step = 0
        self.check_loss_raise = CheckLossRaise()
        for epoch in range(epoch_start, math.ceil(self.args.num_train_epochs)):
            self.indicators = collections.defaultdict(list)

            utils.sync_models(self.rank, self.model)

            self.model.train()

            self.model.zero_grad()
            grad_count = 0

            if self.rank > -1:
                #set epoch to make different samples division betwen proceses for different epoches
                self.dataloader.sampler.set_epoch(epoch)

            for step, batch in enumerate(self.dataloader):
                epoch_fp = epoch + step/len(self.dataloader)
                if epoch_fp > self.args.num_train_epochs:
                    break

                x_noise, x_clean = [t.to(self.args.device) for t in batch]

                #augment and mix signals
                x_clean, x_noise, x = self.mix_signals(x_clean, x_noise)

                #forward pass
                y_clean, Y_clean, _ = self.model(x)

                #calc specter for clean input signal
                tail_size = self.model.wnd_length - self.model.hop_length
                X_clean = self.model.encode(torch.nn.functional.pad(x_clean, (tail_size, 0)))

                # crop target and model output to align to each other
                sample_ahead = self.model.get_sample_ahead()
                spectre_ahead = self.model.ahead
                if sample_ahead > 0:
                    x = x[:, :-sample_ahead]
                    x_clean = x_clean[:, :-sample_ahead]
                    y_clean = y_clean[:, sample_ahead:]
                if spectre_ahead > 0:
                    Y_clean = Y_clean[:, :, :, spectre_ahead:]
                    X_clean = X_clean[:, :, :, :-spectre_ahead]

                loss = self.loss(epoch_fp, y_clean, Y_clean, x_clean, X_clean)
                self.indicators['loss'].append(loss.item())

                #calculate and accumulate gradients
                loss.backward()
                grad_count += 1

                #continue if not all gradients were accumulated
                if grad_count < self.gradient_accumulation_steps:
                    continue

                #make optimization step
                utils.sync_grads(self.rank, self.named_params, global_step==0, grad_count)
                self.optimizer.step()  # make optimization step
                self.scheduler.step()  # Update learning rate schedule
                global_step += 1

                self.model.zero_grad()
                grad_count = 0

                #make logs only after several steps
                if global_step % self.args.logacc != 0:
                    continue

                #average indicator over GPUs and iterations
                self.aver_indicators()

                #check that negsisdr suddenly raise
                #if high raise detected then model parameters are restored and optimizer is reset
                self.check_loss_raise.check(
                    self.indicators_mean["negsisdr"],
                    self.named_params,
                    self.optimizer
                )

                self.log_indicators(epoch_fp)

                self.indicators = collections.defaultdict(list)

            self.save_and_eval_checkpoint(epoch+1)

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

    #set seed
    np.random.seed(args.seed+rank)
    random.seed(args.seed+rank)
    torch.manual_seed(args.seed+rank)

    model = models.model_create(args.model_desc)
    model = model.to(args.device)

    size_to_read = model.get_sample_length_ceil(dataset.FREQ * args.size_to_read)
    printlog("size_to_read {} ".format(size_to_read))

    dataset_train = None
    # only one process scan folders then share info to other
    if rank in [-1, 0]:
        printlog("create train dataset from", args.dns_datasets)
        dataset_train = dataset.DNSDataset(args.dns_datasets, size_to_read=size_to_read )

    if rank>-1:
        #lets sync after data creation
        torch.distributed.barrier()
        dataset_train = utils.brodcast_data(rank, dataset_train)

    #tune params
    m = re.match(r'.*checkpoint-(\d+)', args.model_desc)
    epoch_start = int(m.group(1)) if m else 0

    train = Train(rank, args, model, dataset_train)
    train.train(epoch_start)

    if rank in [-1, 0]:
        #save model, eval model

        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)

        # save to onnx
        model.eval()
        with torch.no_grad():
            #get the smallest size
            size = model.get_sample_length_ceil(1)
            input_tensor = torch.zeros((1,size), dtype=torch.float, device=args.device)

            outputs = model(input_tensor)
            state = outputs[2]
            printlog("state size",sum(t.numel() for t in state),"float params")
            inputs = (input_tensor, state)
            input_names = ['input'] + ["inp_state_{:03}".format(i) for i in range(len(state))]
            output_names = ['output', 'Y']+ ["out_state_{:03}".format(i) for i in range(len(state))]

            #save inputs outputs sizes
            def rec(n,t):
                s = list(t.shape)
                return [
                    '- name: "{}"'.format(n),
                    '  shape: '+str(s),
                    '  layout: ' + ('[N, C, H, W]' if len(s)==4 else '[N, H, W]' if len(s)==3 else '[N, C]'),
                    '  precision: "FP32"',
                    '  type: const',
                    '  float_value: 0'
                ]

            with open(os.path.join(args.output_dir, "input.yml"), 'wt') as out:
                out.write("inputs:\n")
                for n, t in zip(input_names, [input_tensor] + state):
                    for s in rec(n, t):
                        out.write("  "+ s + "\n")
                out.write("outputs:\n")
                for n, t in zip(output_names, list(outputs[:2]) + list(outputs[2])):
                    for s in rec(n, t)[:-2]:
                        out.write("  "+ s + "\n")

            torch.onnx.export(
                model,
                inputs,
                os.path.join(args.output_dir, "model.onnx"),
                verbose=False,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names)

        evaluate_dir(args.eval_data, args.output_dir)

    if rank > -1:
        torch.distributed.barrier()

def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_desc",
        default="PoCoNetLikeModel",
        type=str,
        required=False,
        help="model directory or model json or model desc string")
    parser.add_argument(
        "--dns_datasets",
        type=str,
        default=None,
        required=True,
        help="DNS-Chalange datasets directory")
    parser.add_argument(
        "--eval_data",
        default=None,
        type=str,
        required=False,
        help="synthetic dataset to validate <dns-datasets>/ICASSP_dev_test_set/track_1/synthetic")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory for model")
    parser.add_argument(
        "--total_train_batch_size",
        default=128,
        type=int,
        help="Batch size to make one optimization step.")
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=6,
        type=int,
        help="Batch size per GPU for training.")
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rates for optimizer.")
    parser.add_argument(
        "--weight_decay",
        default=1e-2,
        type=float,
        help="The weight decay for optimizer")
    parser.add_argument(
        "--num_train_epochs",
        default=10.0,
        type=float,
        help="Number of epochs to train")
    parser.add_argument(
        "--no_cuda",
        action='store_true',
        help="Disable GPU calculation")
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed for different inittializations")
    parser.add_argument(
        "--logacc",
        default=50,
        type=int,
        help="Number of optimization steps before log")

    parser.add_argument(
        "--size_to_read",
        default=4.0,
        type=float,
        help="number of second in batch to train infer")

    parser.add_argument(
        "--snr_min",
        default=-20,
        type=float,
        help="Minimal SNR value (dB) for mixing clean signal and noise")
    parser.add_argument(
        "--snr_max",
        default=+10,
        type=float,
        help="Maximal SNR value (dB) for mixing clean signal and noise")
    parser.add_argument(
        "--target_min",
        default=-30,
        type=float,
        help="Minimal (dBFS) for input mixed signal")
    parser.add_argument(
        "--target_max",
        default=-5,
        type=float,
        help="Maximal (dBFS) for input mixed signal")
    parser.add_argument(
        "--clip_prob",
        default=0.1,
        type=float,
        help="Probability to clip input mixed signal")

    args = parser.parse_args(args)

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


if __name__ == "__main__":

    main()
