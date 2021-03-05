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
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter


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


class Trainer():
    def __init__(self, name, models_root, model=None, resume=False):

        self.model = model

        assert (isinstance(self.model, (list, tuple, torch.nn.Module)) or self.model is None)

        self.name = name
        self.models_root = models_root
        self.model_path = os.path.join(models_root, self.name)
        self.logs_path = os.path.join(self.model_path, 'logs')

        self.state = TrainingState()
        self.resume_training = False

        if os.path.exists(self.model_path):
            if resume:
                self.resume_training = True
            else:
                shutil.rmtree(self.model_path)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            os.makedirs(self.logs_path)

        self.tb_writer = SummaryWriter(logdir=self.logs_path)

    # pylint: disable=too-many-arguments
    def train(self, criterion, optimizer, optimizer_params, scheduler, scheduler_params, training_data_loader,
              evaluation_data_loader, pretrained_weights, train_metrics, val_metrics,
              track_metric, epoches, default_val=0, comparator=lambda x, y: x > y):

        # pylint: disable=protected-access
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

        # prepare dicts for metrics
        if not self.state.train_metric:
            for m in train_metrics:
                self.state.train_metric[m.name] = []

            for m in val_metrics:
                self.state.val_metric[m.name] = []

        # training loop
        start_epoch = self.state.epoch
        for i in range(start_epoch, epoches):
            tic = time.time()

            if scheduler is not None:
                scheduler.step()

            self.state.global_step = self._train_one_epoch(criterion, optimizer, training_data_loader, train_metrics,
                                                           self.state.train_metric, i, self.state.global_step)

            self._evaluate_and_save(evaluation_data_loader, val_metrics, track_metric, self.state.val_metric, i,
                                    comparator)

            tac = time.time()
            print('Epoch %d, time %s \n' % (i, tac - tic))

            self._save(suffix='epoch_' + str(self.state.epoch))
            self._save(suffix='last_model')
            self.state.epoch = self.state.epoch + 1

    def predict(self, batch):
        self.model.eval()
        assert isinstance(batch[0], list)
        data = [Variable(b) for b in batch[0]]

        if self.state.cuda:
            data = [d.cuda() for d in data]

        output = self.model(data)
        return output

    def _train_one_epoch(self, criterion, optimizer, training_data_loader, train_metrics, train_metrics_results, epoch,
                         global_step):

        for m in train_metrics:
            m.reset()

        self.model.train()

        for batch in training_data_loader:
            assert (isinstance(batch[0], list) and isinstance(batch[1], list))
            data = [Variable(b) for b in batch[0]]
            target = [Variable(b, requires_grad=False) for b in batch[1]]

            if self.state.cuda:
                data = [d.cuda() for d in data]
                target = [t.cuda() for t in target]

            output = self.model(data)

            if isinstance(criterion, (tuple, list)):
                loss_val = [c(output, target) for c in criterion]
                loss = sum(loss_val)
            else:
                loss = loss_val = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            for m in train_metrics:
                m.update(output, target)

            for idx, l in enumerate(loss_val):
                self.tb_writer.add_scalar('loss/loss-{}'.format(idx), l.item(), global_step)
            global_step = global_step + 1

        for m in train_metrics:
            train_metrics_results[m.name].append(m.get())
            self.tb_writer.add_scalar('train/' + m.name, m.get(), epoch)
            print('Epoch %d, Training %s %s' % (epoch, m.name, m.get()))

        self.state.optimizer_state = optimizer.state_dict()
        return global_step

    def _evaluate_and_save(self, evaluation_data_loader, val_metrics, track_metric, val_metrics_results, epoch,
                           comparator):

        for m in val_metrics:
            m.reset()

        self.model.eval()

        for batch in evaluation_data_loader:
            assert isinstance(batch[0], list) and isinstance(batch[1], list)
            data = [Variable(b) for b in batch[0]]
            target = [Variable(b, requires_grad=False) for b in batch[1]]

            if self.state.cuda:
                data = [d.cuda() for d in data]
                target = [t.cuda() for t in target]

            output = self.model(data)

            for m in val_metrics:
                m.update(target, output)

        val = 0.0
        for m in val_metrics:
            if m.name == track_metric:
                val = m.get()

            self.tb_writer.add_scalar('val/' + m.name, m.get(), epoch)
            val_metrics_results[m.name].append(m.get())
            print('Epoch %d, Validation %s %s' % (epoch, m.name, m.get()))

        if comparator(val, self.state.best_val):
            self.state.best_val = val
            self._save(suffix='best_model')
            print('model saved')

    def _save(self, suffix):
        s = {'state': self.state,
             'model': self.model}

        torch.save(s, os.path.join(self.model_path, self.name + '_' + suffix + '.pth'))

    def _load(self, suffix):
        print('loading model...')
        model_path = os.path.join(self.model_path, self.name + '_' + suffix + '.pth')
        s = torch.load(model_path)
        self.state = s['state']
        if self.model is None:
            self.model = s['model']
        else:
            self.model.load_state_dict(s['model'].state_dict())

    def load_latest(self):
        self._load('last_model')

    def load_best(self):
        self._load('best_model')
