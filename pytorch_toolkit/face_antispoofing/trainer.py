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

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from eval_protocol import evaluate
from utils import (AverageMeter, cutmix, load_checkpoint,
                   mixup_target, precision, save_checkpoint)


class Trainer:
    def __init__(self, model, criterion, optimizer, device,
                 config, train_loader, val_loader, test_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_step, self.val_step = 0, 0
        self.best_accuracy, self.current_accuracy, self.current_auc = 0, 0, 0
        self.current_eer, self.best_acer = float('inf'), float('inf')
        self.path_to_checkpoint = os.path.join(self.config.checkpoint.experiment_path,
                                                self.config.checkpoint.snapshot_name)
        self.data_parallel = self.config.data_parallel.use_parallel
        self.writer = SummaryWriter(self.config.checkpoint.experiment_path)

    def train(self, epoch: int):
        ''' method to train your model for epoch '''
        losses = AverageMeter()
        accuracy = AverageMeter()
        # switch to train mode and train one epoch
        self.model.train()
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
        for i, (input_, target) in loop:
            if i == self.config.test_steps:
                break
            input_ = input_.to(self.device)
            target = target.to(self.device)
            # compute output and loss
            if self.config.aug.type_aug:
                if self.config.aug.type_aug == 'mixup':
                    aug_output = mixup_target(input_, target, self.config, self.device)
                else:
                    assert self.config.aug.type_aug == 'cutmix'
                    aug_output = cutmix(input_, target, self.config, self.device)
                input_, target_a, target_b, lam = aug_output
                tuple_target = (target_a, target_b, lam)
                if self.config.multi_task_learning:
                    hot_target = lam*F.one_hot(target_a[:,0], 2) + (1-lam)*F.one_hot(target_b[:,0], 2)
                else:
                    hot_target = lam*F.one_hot(target_a, 2) + (1-lam)*F.one_hot(target_b, 2)
                output = self.make_output(input_, hot_target)
                if self.config.multi_task_learning:
                    loss = self.multi_task_criterion(output, tuple_target)
                else:
                    loss = self.mixup_criterion(self.criterion, output,
                                                target_a, target_b, lam, 2)
            else:
                new_target = (F.one_hot(target[:,0], num_classes=2)
                            if self.config.multi_task_learning
                            else F.one_hot(target, num_classes=2))
                output = self.make_output(input_, new_target)
                loss = (self.multi_task_criterion(output, target)
                        if self.config.multi_task_learning
                        else self.criterion(output, new_target))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure accuracy
            s = self.config.loss.amsoftmax.s
            acc = (precision(output[0], target[:,0].reshape(-1), s)
                  if self.config.multi_task_learning
                  else precision(output, target, s))
            # record loss
            losses.update(loss.item(), input_.size(0))
            accuracy.update(acc, input_.size(0))

            # write to writer for tensorboard
            self.writer.add_scalar('Train/loss', loss, global_step=self.train_step)
            self.writer.add_scalar('Train/accuracy', accuracy.avg, global_step=self.train_step)
            self.train_step += 1

            # update progress bar
            max_epochs = self.config.epochs.max_epoch
            loop.set_description(f'Epoch [{epoch}/{max_epochs}]')
            loop.set_postfix(loss=loss.item(), avr_loss = losses.avg,
                             acc=acc, avr_acc=accuracy.avg,
                             lr=self.optimizer.param_groups[0]['lr'])
        return losses.avg, accuracy.avg

    def validate(self):
        ''' method to validate model on current epoch '''
        # meters
        losses = AverageMeter()
        accuracy = AverageMeter()
        # switch to evaluation mode and inference the model
        self.model.eval()
        loop = tqdm(enumerate(self.val_loader), total=len(self.val_loader), leave=False)
        criterion = self.criterion[0] if self.config.multi_task_learning else self.criterion
        for i, (input_, target) in loop:
            if i == self.config.test_steps:
                break
            input_ = input_.to(self.device)
            target = target.to(self.device)
            if len(target.shape) > 1:
                target = target[:, 0].reshape(-1)
            # computing output and loss
            with torch.no_grad():
                features = self.model(input_)
                if self.data_parallel:
                    model1 = self.model.module
                else:
                    model1 = self.model

                output = model1.make_logits(features, all=True)
                if isinstance(output, tuple):
                    output = output[0]

                new_target = F.one_hot(target, num_classes=2)
                loss = criterion(output, new_target)

            # measure accuracy and record loss
            acc = precision(output, target, s=self.config.loss.amsoftmax.s)
            losses.update(loss.item(), input_.size(0))
            accuracy.update(acc, input_.size(0))

            # update progress bar
            loop.set_postfix(loss=loss.item(), avr_loss = losses.avg, acc=acc, avr_acc=accuracy.avg)

        print(f'val accuracy on epoch: {round(accuracy.avg, 3)}, loss on epoch:{round(losses.avg, 3)}')
        # write val in writer
        self.writer.add_scalar('Val/loss', losses.avg, global_step=self.val_step)
        self.writer.add_scalar('Val/accuracy',  accuracy.avg, global_step=self.val_step)
        self.val_step += 1

        return accuracy.avg

    def eval(self, epoch: int, epoch_accuracy: float, save_chkpt: bool=True):
        # evaluate on last 10 epoch and remember best accuracy, AUC, EER, ACER and then save checkpoint
        if (epoch%10 == 0 or epoch >= (self.config.epochs.max_epoch - 10)) and (epoch_accuracy > self.current_accuracy):
            print('__VAL__:')
            AUC, EER, apcer, bpcer, acer = evaluate(self.model, self.val_loader,
                                                    self.config, self.device, compute_accuracy=False)
            print(self.print_result(AUC, EER, epoch_accuracy, apcer, bpcer, acer))
            if acer < self.best_acer:
                self.best_acer = acer
                if save_chkpt:
                    checkpoint = {'state_dict': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
                    save_checkpoint(checkpoint, f'{self.path_to_checkpoint}')
                self.current_accuracy = epoch_accuracy
                self.current_eer = EER
                self.current_auc = AUC
                AUC, EER, accur, apcer, bpcer, acer, _, _ = evaluate(self.model, self.test_loader, self.config,
                                                                     self.device, compute_accuracy=True)
                print('__TEST__:')
                print(self.print_result(AUC, EER, accur, apcer, bpcer, acer))

    def make_output(self, input_: torch.tensor, target: torch.tensor):
        ''' target - one hot for main task
        return output
        If use rsc compute output applying rsc method'''
        assert target.shape[1] == 2
        if self.config.RSC.use_rsc:
            # making features after avg pooling
            features = self.model(input_)
            if self.data_parallel:
                model1 = self.model.module
            else:
                model1 = self.model
            # do everything after convolutions layers, strating after the avg pooling
            all_tasks_output = model1.make_logits(features, all=True)
            logits = (all_tasks_output[0]
                      if self.config.multi_task_learning
                      else all_tasks_output)
            if isinstance(logits, tuple):
                logits = logits[0]
            # take a derivative, make tensor, shape as features, but gradients insted features
            if self.config.aug.type_aug:
                fold_target = target.argmax(dim=1)
                target = F.one_hot(fold_target, num_classes=target.shape[1])
            target_logits = torch.sum(logits*target, dim=1)
            gradients = torch.autograd.grad(target_logits, features,
                                            grad_outputs=torch.ones_like(target_logits),
                                            create_graph=True)[0]
            # get value of 1-p quatile
            quantile = torch.tensor(np.quantile(a=gradients.data.cpu().numpy(),
                                                q=1-self.config.RSC.p, axis=(1,2,3)),
                                                device=input_.device)
            quantile = quantile.reshape(input_.size(0),1,1,1)
            # create mask
            mask = gradients < quantile
            # element wise product of features and mask, correction for expectition value
            new_features = (features*mask)/(1-self.config.RSC.p)
            # compute new logits
            new_logits = model1.make_logits(new_features, all=False)
            if isinstance(new_logits, tuple):
                new_logits = new_logits[0]
            # compute this operation batch wise
            random_uniform = torch.rand(size=(input_.size(0), 1), device=input_.device)
            random_mask = random_uniform <= self.config.RSC.b
            output = torch.where(random_mask, new_logits, logits)
            if self.config.loss.loss_type == 'soft_triple':
                output = ((output, all_tasks_output[0][1])
                          if self.config.multi_task_learning
                          else (output, all_tasks_output[1]))
            output = (output, *all_tasks_output[1:])
            return output
        else:
            assert not self.config.RSC.use_rsc
            features = self.model(input_)
            if self.data_parallel:
                model1 = self.model.module
            else:
                model1 = self.model
            output = model1.make_logits(features, all=True)
            return output

    def multi_task_criterion(self, output: tuple, target: torch.tensor,
                             C: float=1., Cs: float=0.1, Ci: float=0.1, Cf: float=1.):
        ''' output -> tuple of given losses
        target -> torch tensor of a shape [batch*num_tasks]
        return loss function '''
        softmax, cross_entropy, bce = self.criterion
        if self.config.aug.type_aug:
            target_a, target_b, lam = target
            spoof_loss = self.mixup_criterion(softmax, output[0], target_a[:,0], target_b[:,0], lam, 2)
            spoof_type_loss = self.mixup_criterion(cross_entropy, output[1], y_a=target_a[:,1],
                                                                y_b=target_b[:,1],
                                                                lam=lam, num_classes=11)
            lightning_loss = self.mixup_criterion(cross_entropy, output[2], y_a=target_a[:,2],
                                                                y_b=target_b[:,2],
                                                                lam=lam, num_classes=5)
            real_atr_loss = lam*bce(output[3], target_a[:,3:].type(torch.float32)) + (1-lam)*bce(output[3],
                                    target_b[:,3:].type(torch.float32))
        else:
            # spoof loss, take derivitive
            spoof_target = F.one_hot(target[:,0], num_classes=2)
            spoof_type_target = F.one_hot(target[:,1], num_classes=11)
            lightning_target = F.one_hot(target[:,2], num_classes=5)

            # compute losses
            spoof_loss = softmax(output[0], spoof_target)
            spoof_type_loss = cross_entropy(output[1], spoof_type_target)
            lightning_loss = cross_entropy(output[2], lightning_target)

            # filter output for real images and compute third loss
            mask = target[:,0] == 0
            filtered_output = output[3][mask]
            filtered_target = target[:,3:][mask].type(torch.float32)
            real_atr_loss = bce(filtered_output, filtered_target)
        # combine losses
        loss = C*spoof_loss + Cs*spoof_type_loss + Ci*lightning_loss + Cf*real_atr_loss
        return loss

    @staticmethod
    def mixup_criterion(criterion, pred, y_a, y_b, lam, num_classes):
        ''' y_a and y_b considered to be folded target labels.
        All losses waits to get one_hot target as an input except the BCELoss '''
        ya_hot = F.one_hot(y_a, num_classes=num_classes)
        yb_hot = F.one_hot(y_b, num_classes=num_classes)
        mixed_target = lam * ya_hot  + (1 - lam) * yb_hot
        return criterion(pred, mixed_target)

    def test(self, file_name):
        ''' get metrics and record it to the file '''
        print('_____________EVAULATION_____________')
        # load snapshot
        load_checkpoint(self.path_to_checkpoint, self.model,
                        map_location=self.device, optimizer=None,
                        strict=True)

        for loader in (self.val_loader, self.test_loader):
            # printing results
            AUC, EER, accur, apcer, bpcer, acer, _, _ = evaluate(self.model, loader, self.config,
                                                                self.device, compute_accuracy=True)
            results = self.print_result(AUC, EER, accur, apcer, bpcer, acer)
            with open(os.path.join(self.config.checkpoint.experiment_path, file_name), 'a') as f:
                f.write(results)

    @staticmethod
    def print_result(AUC, EER, accur, apcer, bpcer, acer):
        results = (f'accuracy on test data = {round(np.mean(accur)*100,3)}\n'
                   + f'AUC = {round(AUC,3)}\n'
                   + f'EER = {round(EER*100,2)}\n'
                   + f'apcer = {round(apcer*100,2)}\n'
                   + f'bpcer = {round(bpcer*100,2)}\n'
                   + f'acer = {round(acer*100,2)}\n')
        return results

    def get_exp_info(self):
        if not self.config.test_steps:
            exp_num = self.config.exp_num
            print(f'_______INIT EXPERIMENT {exp_num}______')
            train_dataset, test_dataset = self.config.dataset, self.config.test_dataset.type
            print(f'training on {train_dataset}, testing on {test_dataset}')
            print('\n\nSNAPSHOT')
            for key, item in self.config.checkpoint.items():
                print(f'{key} --> {item}')
            print('\n\nMODEL')
            for key, item in self.config.model.items():
                print(f'{key} --> {item}')
            loss_type = self.config.loss.loss_type
            print(f'\n\nLOSS TYPE : {loss_type.upper()}')
            for key, item in self.config.loss[f'{loss_type}'].items():
                print(f'{key} --> {item}')
            print('\n\nDROPOUT PARAMS')
            for key, item in self.config.dropout.items():
                print(f'{key} --> {item}')
            print('\n\nOPTIMAIZER')
            for key, item in self.config.optimizer.items():
                print(f'{key} --> {item}')
            print('\n\nADDITIONAL USING PARAMETRS')
            if self.config.aug.type_aug:
                type_aug = self.config.aug.type_aug
                print(f'\nAUG TYPE = {type_aug} USING')
                for key, item in self.config.aug.items():
                    print(f'{key} --> {item}')
            if self.config.RSC.use_rsc:
                print('RSC USING')
                for key, item in self.config.RSC.items():
                    print(f'{key} --> {item}')
            if self.data_parallel:
                ids = self.config.data_parallel.parallel_params.device_ids
                print(f'USING DATA PATALLEL ON {ids} GPU')
            if self.config.data.sampler:
                print('USING SAMPLER')
            if self.config.loss.amsoftmax.ratio != (1,1):
                print(self.config.loss.amsoftmax.ratio)
                print('USING ADAPTIVE LOSS')
            if self.config.multi_task_learning:
                print('multi_task_learning using'.upper())
            theta = self.config.conv_cd.theta
            if theta > 0:
                print(f'CDC method: {theta}')
