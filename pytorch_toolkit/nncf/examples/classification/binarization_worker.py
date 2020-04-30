"""
 Copyright (c) 2019-2020 Intel Corporation
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

import time


import copy
import os.path as osp
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from examples.classification.main import create_data_loaders, validate, AverageMeter, accuracy, get_lr, create_datasets
from examples.common.distributed import configure_distributed
from examples.common.example_logger import logger
from examples.common.execution import ExecutionMode, get_device, prepare_model_for_execution
from examples.common.model_loader import load_model
from examples.common.utils import configure_logging, print_args, make_additional_checkpoints, get_name, print_statistics
from nncf.binarization.algo import BinarizationController
from nncf.model_creation import create_compressed_model
from nncf.checkpoint_loading import load_state
from nncf.utils import manual_seed, is_main_process


class KDLossCalculator:
    def __init__(self, original_model, temperature=1.0):
        self.original_model = original_model
        self.original_model.eval()
        self.temperature = temperature

    def loss(self, inputs, binarized_network_outputs):
        T = self.temperature
        with torch.no_grad():
            ref_output = self.original_model(inputs).detach()
        kd_loss = -(nn.functional.log_softmax(binarized_network_outputs / T, dim=1) *
                    nn.functional.softmax(ref_output / T, dim=1)).mean() * (T * T * binarized_network_outputs.shape[1])
        return kd_loss


def get_binarization_optimizer(params_to_optimize, binarization_config):
    params = binarization_config.get("params", {})
    base_lr = params.get("base_lr", 1e-3)
    base_wd = params.get("base_wd", 1e-5)
    return torch.optim.Adam(params_to_optimize,
                            lr=base_lr,
                            weight_decay=base_wd)


class BinarizationOptimizerScheduler:
    def __init__(self, optimizer, binarization_config):
        params = binarization_config.get('params', {})
        self.base_lr = binarization_config.get("base_lr", 1e-3)
        self.lr_poly_drop_start_epoch = params.get('lr_poly_drop_start_epoch', None)
        self.lr_poly_drop_duration_epochs = params.get('lr_poly_drop_duration_epochs', 30)
        self.disable_wd_start_epoch = params.get('disable_wd_start_epoch', None)
        self.optimizer = optimizer
        self.last_epoch = 0

    def step(self, epoch_fraction):
        epoch_float = self.last_epoch + epoch_fraction
        if self.lr_poly_drop_start_epoch is not None:
            start = self.lr_poly_drop_start_epoch
            finish = self.lr_poly_drop_start_epoch + self.lr_poly_drop_duration_epochs
            if start <= epoch_float < finish:
                lr = self.base_lr * pow(float(finish - epoch_float) / float(self.lr_poly_drop_duration_epochs), 2.1)
                for group in self.optimizer.param_groups:
                    group['lr'] = lr

        if self.disable_wd_start_epoch is not None:
            if epoch_float > self.disable_wd_start_epoch:
                for group in self.optimizer.param_groups:
                    group['weight_decay'] = 0.0

    def epoch_step(self, epoch=None):
        if epoch is not None:
            self.last_epoch = epoch
        self.last_epoch += 1

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def state_dict(self):
        return self.__dict__


def main_worker_binarization(current_gpu, config):
    config.current_gpu = current_gpu
    config.distributed = config.execution_mode in (ExecutionMode.DISTRIBUTED, ExecutionMode.MULTIPROCESSING_DISTRIBUTED)
    if config.distributed:
        configure_distributed(config)

    config.device = get_device(config)

    if is_main_process():
        configure_logging(logger, config)
        print_args(config)

    if config.seed is not None:
        manual_seed(config.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    # create model
    model_name = config['model']
    weights = config.get('weights')
    model = load_model(model_name,
                       pretrained=config.get('pretrained', True) if weights is None else False,
                       num_classes=config.get('num_classes', 1000),
                       model_params=config.get('model_params'))

    original_model = copy.deepcopy(model)
    compression_ctrl, model = create_compressed_model(model, config)
    if not isinstance(compression_ctrl, BinarizationController):
        raise RuntimeError("The binarization sample worker may only be run with the binarization algorithm!")

    if weights:
        load_state(model, torch.load(weights, map_location='cpu'))

    model, _ = prepare_model_for_execution(model, config)
    original_model.to(config.device)

    if config.distributed:
        compression_ctrl.distributed()

    is_inception = 'inception' in model_name

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(config.device)

    params_to_optimize = model.parameters()

    compression_config = config['compression']
    binarization_config = compression_config if isinstance(compression_config, dict) else compression_config[0]
    optimizer = get_binarization_optimizer(params_to_optimize, binarization_config)
    optimizer_scheduler = BinarizationOptimizerScheduler(optimizer, binarization_config)
    kd_loss_calculator = KDLossCalculator(original_model)

    resuming_checkpoint = config.resuming_checkpoint
    best_acc1 = 0
    # optionally resume from a checkpoint
    if resuming_checkpoint is not None:
        model, config, optimizer, optimizer_scheduler, kd_loss_calculator, compression_ctrl, best_acc1 = \
            resume_from_checkpoint(resuming_checkpoint, model,
                                   config, optimizer, optimizer_scheduler, kd_loss_calculator, compression_ctrl)

    if config.to_onnx is not None:
        compression_ctrl.export_model(config.to_onnx)
        logger.info("Saved to {}".format(config.to_onnx))
        return

    if config.execution_mode != ExecutionMode.CPU_ONLY:
        cudnn.benchmark = True

    # Data loading code
    train_dataset, val_dataset = create_datasets(config)
    train_loader, train_sampler, val_loader = create_data_loaders(config, train_dataset, val_dataset)

    if config.mode.lower() == 'test':
        print_statistics(compression_ctrl.statistics())
        validate(val_loader, model, criterion, config)

    if config.mode.lower() == 'train':
        if not resuming_checkpoint:
            compression_ctrl.initialize(data_loader=train_loader, criterion=criterion)

        batch_multiplier = (binarization_config.get("params", {})).get("batch_multiplier", 1)
        train_bin(config, compression_ctrl, model, criterion, is_inception, optimizer_scheduler, model_name, optimizer,
                  train_loader, train_sampler, val_loader, kd_loss_calculator, batch_multiplier, best_acc1)


def train_bin(config, compression_ctrl, model, criterion, is_inception, optimizer_scheduler, model_name, optimizer,
              train_loader, train_sampler, val_loader, kd_loss_calculator, batch_multiplier, best_acc1=0):
    for epoch in range(config.start_epoch, config.epochs):
        config.cur_epoch = epoch
        if config.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_epoch_bin(train_loader, batch_multiplier, model, criterion, optimizer, optimizer_scheduler,
                        kd_loss_calculator, compression_ctrl, epoch, config, is_inception)

        # compute compression algo statistics
        stats = compression_ctrl.statistics()

        acc1 = best_acc1
        if epoch % config.test_every_n_epochs == 0:
            # evaluate on validation set
            acc1, _ = validate(val_loader, model, criterion, config)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # update compression scheduler state at the end of the epoch
        compression_ctrl.scheduler.epoch_step()
        optimizer_scheduler.epoch_step()

        if is_main_process():
            print_statistics(stats)

            checkpoint_path = osp.join(config.checkpoint_save_dir, get_name(config) + '_last.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'arch': model_name,
                'state_dict': model.state_dict(),
                'original_model_state_dict': kd_loss_calculator.original_model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'compression_scheduler': compression_ctrl.scheduler.state_dict(),
                'optimizer_scheduler': optimizer_scheduler.state_dict()
            }

            torch.save(checkpoint, checkpoint_path)
            make_additional_checkpoints(checkpoint_path, is_best, epoch + 1, config)

            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    config.tb.add_scalar("compression/statistics/{0}".format(key), value, len(train_loader) * epoch)


def resume_from_checkpoint(resuming_checkpoint, model, config, optimizer, optimizer_scheduler, kd_loss_calculator,
                           compression_ctrl):
    best_acc1 = 0
    if osp.isfile(resuming_checkpoint):
        logger.info("=> loading checkpoint '{}'".format(resuming_checkpoint))
        checkpoint = torch.load(resuming_checkpoint, map_location='cpu')
        load_state(model, checkpoint['state_dict'], is_resume=True)
        if config.mode.lower() == 'train' and config.to_onnx is None:
            config.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            kd_loss_calculator.original_model.load_state_dict(checkpoint['original_model_state_dict'])
            compression_ctrl.scheduler.load_state_dict(checkpoint['compression_scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer_scheduler.load_state_dict(checkpoint['optimizer_scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch: {}, best_acc1: {:.3f})"
                        .format(resuming_checkpoint, checkpoint['epoch'], best_acc1))
        else:
            logger.info("=> loaded checkpoint '{}'".format(resuming_checkpoint))
    else:
        raise FileNotFoundError("no checkpoint found at '{}'".format(resuming_checkpoint))
    return model, config, optimizer, optimizer_scheduler, kd_loss_calculator, compression_ctrl, best_acc1


def train_epoch_bin(train_loader, batch_multiplier, model, criterion, optimizer,
                    optimizer_scheduler: BinarizationOptimizerScheduler, kd_loss_calculator: KDLossCalculator,
                    compression_ctrl, epoch, config, is_inception=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses_meter = AverageMeter()
    criterion_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    compression_scheduler = compression_ctrl.scheduler

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_ = input_.to(config.device)
        target = target.to(config.device)

        # compute output
        if is_inception:
            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
            output, aux_outputs = model(input_)
            loss1 = criterion(output, target)
            loss2 = criterion(aux_outputs, target)
            criterion_loss = loss1 + 0.4 * loss2
        else:
            output = model(input_)
            criterion_loss = criterion(output, target)

        # compute KD loss
        kd_loss = kd_loss_calculator.loss(input_, output)
        loss = criterion_loss + kd_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input_.size(0))
        comp_loss_val = kd_loss.item()
        kd_losses_meter.update(comp_loss_val, input_.size(0))
        criterion_losses.update(criterion_loss.item(), input_.size(0))
        top1.update(acc1, input_.size(0))
        top1.update(acc1, input_.size(0))
        top5.update(acc5, input_.size(0))

        # compute gradient and do SGD step
        if i % batch_multiplier == 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss.backward()

        compression_scheduler.step()
        optimizer_scheduler.step(float(i) / len(train_loader))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            logger.info(
                '{rank}: '
                'Epoch: [{0}][{1}/{2}] '
                'Lr: {3:.3} '
                'Wd: {4:.3} '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                'CE_loss: {ce_loss.val:.4f} ({ce_loss.avg:.4f}) '
                'KD_loss: {kd_loss.val:.4f} ({kd_loss.avg:.4f}) '
                'Loss: {loss.val:.4f} ({loss.avg:.4f}) '
                'Acc@1: {top1.val:.3f} ({top1.avg:.3f}) '
                'Acc@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), get_lr(optimizer), get_wd(optimizer), batch_time=batch_time,
                    data_time=data_time, ce_loss=criterion_losses, kd_loss=kd_losses_meter,
                    loss=losses, top1=top1, top5=top5,
                    rank='{}:'.format(config.rank) if config.multiprocessing_distributed else ''
                ))

        if is_main_process():
            global_step = len(train_loader) * epoch
            config.tb.add_scalar("train/learning_rate", get_lr(optimizer), i + global_step)
            config.tb.add_scalar("train/criterion_loss", criterion_losses.avg, i + global_step)
            config.tb.add_scalar("train/kd_loss", kd_losses_meter.avg, i + global_step)
            config.tb.add_scalar("train/loss", losses.avg, i + global_step)
            config.tb.add_scalar("train/top1", top1.avg, i + global_step)
            config.tb.add_scalar("train/top5", top5.avg, i + global_step)

            for stat_name, stat_value in compression_ctrl.statistics().items():
                if isinstance(stat_value, (int, float)):
                    config.tb.add_scalar('train/statistics/{}'.format(stat_name), stat_value, i + global_step)


def get_wd(optimizer):
    return optimizer.param_groups[0]['weight_decay']
