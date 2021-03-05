# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os
import shutil
import time
import gc

import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
from segthor import metrics


class TrainingState():
    def __init__(self):
        self.epoch = 0
        self.train_metric = dict()
        self.val_metric = dict()
        # number of processed batches
        self.global_step = 0
        self.best_val = 0
        self.optimizer_state = None
        self.cuda = True


#pylint: disable=R0913,R0914,R0915,W0212
class Trainer():
    def __init__(self, name, models_root, model=None, rewrite=False, connect_tb=True):

        self.model = model

        assert (isinstance(self.model, (list, tuple, torch.nn.Module)) or self.model is None)

        self.name = name
        self.models_root = models_root
        self.model_path = os.path.join(models_root, self.name)
        self.logs_path = os.path.join(self.model_path, 'logs')

        self.state = TrainingState()
        self.resume_training = False
        self.eval_cpu = True

        if os.path.exists(self.model_path):
            if rewrite:
                shutil.rmtree(self.model_path)
            else:
                self.resume_training = True

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
            os.mkdir(self.logs_path)

        if connect_tb:
            self.tb_writer = SummaryWriter(logdir=self.logs_path)

    def cuda(self):
        if self.model is not None:
            self.model.cuda()
        self.state.cuda = True

    def train(self, criterion, optimizer, optimizer_params, scheduler, scheduler_params, training_data_loader,
              evaluation_data_loader, pretrained_weights, train_metrics, val_metrics,
              track_metric, epoches, default_val, comparator, eval_cpu):

        self.eval_cpu = eval_cpu

        assert isinstance(criterion, (tuple, list, torch.nn.modules.loss._Loss))

        # load weights if any
        if self.resume_training:
            # load training and continue
            self.load_latest()
        elif pretrained_weights is not None:
            # load dictionary only
            self.model.load_state_dict(pretrained_weights)
        else:
            self.state.best_val = default_val

        if isinstance(optimizer, type):
            optimizer = optimizer(params=self.model.parameters(), **optimizer_params)

        if scheduler is not None:
            if isinstance(scheduler, type):
                scheduler = scheduler(optimizer=optimizer, **scheduler_params)

        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler) or scheduler is None

        if self.state.optimizer_state is not None:
            optimizer.load_state_dict(self.state.optimizer_state)
            print('Loaded optimizer state')

        # prepare dicts for metrics
        if not self.state.train_metric:
            for m in train_metrics:
                self.state.train_metric[m.name] = []

            for m in val_metrics:
                self.state.val_metric[m.name] = []

        gc.collect()

        # training loop
        start_epoch = self.state.epoch
        for i in range(start_epoch, epoches):
            tic = time.time()

            self.state.global_step = self._train_one_epoch(criterion, optimizer, training_data_loader,
                                                           train_metrics, self.state.train_metric, i,
                                                           self.state.global_step, scheduler)

            self._evaluate_and_save(evaluation_data_loader, val_metrics, track_metric, self.state.val_metric, i,
                                    comparator)

            tac = time.time()
            print('Epoch %d, time %s \n' % (i, tac - tic))

            self._save(suffix='_epoch_' + str(self.state.epoch))
            self._save(suffix='last_model')
            self.state.epoch = self.state.epoch + 1

        np.random.seed(np.random.get_state()[1][0] + 16)

    def predict(self, batch):
        self.model.eval()

        if self.state.cuda:
            self.model.cuda()

        with torch.no_grad():
            assert isinstance(batch[0], list)
            data = [Variable(b) for b in batch[0]]

            if self.state.cuda:
                data = [d.cuda() for d in data]

            output = self.model(data)
        return output

    def _train_one_epoch(self, criterion, optimizer, training_data_loader, train_metrics, train_metrics_results, epoch,
                         global_step, scheduler):

        aggregate_batches = 1
        for m in train_metrics:
            m.reset()

        if self.state.cuda:
            self.model.cuda()

        self.model.train()

        optimizer.zero_grad()
        for idx, batch in enumerate(training_data_loader):

            assert (isinstance(batch[0], list) and isinstance(batch[1], list))
            data = [Variable(b) for b in batch[0]]
            target = [Variable(b, requires_grad=False) for b in batch[1]]

            if self.state.cuda:
                data = [d.cuda() for d in data]
                target = [t.cuda() for t in target]

            output = self.model(data)

            if isinstance(criterion, (tuple, list)):
                loss_val = [c(output, target) for c in criterion]
                loss = sum(loss_val) / (len(loss_val))
            else:
                loss_val = criterion(output, target)
                loss = loss_val

            loss.backward()

            if (idx+1) % aggregate_batches == 0:

                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            for m in train_metrics:
                m.update(output, target)

            for idxx, l in enumerate(loss_val):
                self.tb_writer.add_scalar('loss/loss-{}'.format(idxx), l.item(), global_step)

            for idxx, param_group in enumerate(optimizer.param_groups):
                self.tb_writer.add_scalar('misc/lr-{}'.format(idxx), param_group['lr'], global_step)

            global_step = global_step + 1

        for m in train_metrics:
            train_metrics_results[m.name].append(m.get())
            metrics.print_metrics(self.tb_writer, m, 'train/', epoch)

        self.state.optimizer_state = optimizer.state_dict()
        return global_step

    def _evaluate_and_save(self, evaluation_data_loader, val_metrics, track_metric, val_metrics_results, epoch,
                           comparator):

        for m in val_metrics:
            m.reset()

        self.model.eval()

        for batch in evaluation_data_loader:
            gc.collect()

            assert (isinstance(batch[0], list) and isinstance(batch[1], list))

            data = batch[0]#[b.cuda() for b in batch[0]]
            tmp_model = self.model.module.cpu()
            tmp_model.eval()
            with torch.no_grad():
                output = tmp_model(data)

            target = [Variable(b, requires_grad=False) for b in batch[1]]
            for m in val_metrics:
                m.update(target, output)

        val = 0.0
        for m in val_metrics:
            if m.name == track_metric:
                val = m.get()

            metrics.print_metrics(self.tb_writer, m, 'val/', epoch)
            val_metrics_results[m.name].append(m.get())

        if comparator(val, self.state.best_val):
            self.state.best_val = val
            self._save(suffix='best_model')
            print('model saved')

    def _save(self, suffix):
        s = {'state': self.state,
             'model': self.model}

        torch.save(s, os.path.join(self.model_path, self.name + suffix + '.pth'))

    def _load(self, suffix):
        print('loading model')
        s = torch.load(os.path.join(self.model_path, self.name + suffix + '.pth'), map_location=torch.device('cpu'))
        self.state = s['state']
        if self.model is None:
            self.model = s['model']
        else:
            self.model.load_state_dict(s['model'].state_dict())

    def load_latest(self):
        self._load('last_model')

    def load_best(self):
        self._load('best_model')
