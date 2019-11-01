"""
 Copyright (c) 2019 Intel Corporation
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
import os.path as osp
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import CIFAR10, CIFAR100

from examples.common.argparser import get_common_argument_parser
from examples.common.distributed import configure_distributed, is_main_process
from examples.common.execution import ExecutionMode, get_device, get_execution_mode, \
    prepare_model_for_execution, start_worker
from examples.common.model_loader import load_model
from examples.common.optimizer import get_parameter_groups, make_optimizer
from examples.common.utils import configure_logging, configure_paths, create_code_snapshot, \
    print_args, make_additional_checkpoints, get_name, is_binarization
from nncf.config import Config
from nncf.dynamic_graph import patch_torch_operators
from nncf.helpers import create_compressed_model, load_state, safe_thread_call
from nncf.utils import manual_seed, print_statistics

patch_torch_operators()

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def get_argument_parser():
    parser = get_common_argument_parser()
    parser.add_argument(
        "--dataset",
        help="Dataset to use.",
        choices=["imagenet", "cifar100", "cifar10"],
        default=None
    )
    parser.add_argument('--test-every-n-epochs', default=1, type=int,
                        help='Enables running validation every given number of epochs')
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(args=argv)
    config = Config.from_json(args.config)
    config.update_from_args(args, parser)
    if config.dist_url == "env://":
        config.update_from_env()

    configure_paths(config)
    source_root = Path(__file__).absolute().parents[2]  # nncf root
    create_code_snapshot(source_root, osp.join(config.log_dir, "snapshot.tar.gz"))

    if config.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    config.execution_mode = get_execution_mode(config)

    if not is_binarization(config):
        start_worker(main_worker, config)
    else:
        from examples.classification.binarization_worker import main_worker_binarization
        start_worker(main_worker_binarization, config)


def main_worker(current_gpu, config):
    config.current_gpu = current_gpu
    config.distributed = config.execution_mode in (ExecutionMode.DISTRIBUTED, ExecutionMode.MULTIPROCESSING_DISTRIBUTED)
    if config.distributed:
        configure_distributed(config)

    config.device = get_device(config)

    if is_main_process():
        configure_logging(config)
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
    compression_algo, model = create_compressed_model(model, config)
    if weights:
        load_state(model, torch.load(weights, map_location='cpu'))
    model, _ = prepare_model_for_execution(model, config)
    if config.distributed:
        compression_algo.distributed()

    is_inception = 'inception' in model_name

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(config.device)

    params_to_optimize = get_parameter_groups(model, config)
    optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)

    resuming_checkpoint = config.resuming_checkpoint
    best_acc1 = 0
    # optionally resume from a checkpoint
    if resuming_checkpoint is not None:
        model, config, optimizer, compression_algo, best_acc1 = \
            resume_from_checkpoint(resuming_checkpoint, model,
                                   config, optimizer, compression_algo)

    if config.to_onnx is not None:
        compression_algo.export_model(config.to_onnx)
        print("Saved to", config.to_onnx)
        return

    if config.execution_mode != ExecutionMode.CPU_ONLY:
        cudnn.benchmark = True

    # Data loading code
    train_loader, train_sampler, val_loader = create_dataloaders(config)

    if config.mode.lower() == 'test':
        print_statistics(compression_algo.statistics())
        validate(val_loader, model, criterion, config)

    if config.mode.lower() == 'train':
        if not resuming_checkpoint:
            compression_algo.initialize(train_loader)
        train(config, compression_algo, model, criterion, is_inception, lr_scheduler, model_name, optimizer,
              train_loader, train_sampler, val_loader, best_acc1)


def train(config, compression_algo, model, criterion, is_inception, lr_scheduler, model_name, optimizer,
          train_loader, train_sampler, val_loader, best_acc1=0):
    for epoch in range(config.start_epoch, config.epochs):
        config.cur_epoch = epoch
        if config.distributed:
            train_sampler.set_epoch(epoch)
        lr_scheduler.step(epoch if not isinstance(lr_scheduler, ReduceLROnPlateau) else best_acc1)

        # train for one epoch
        train_epoch(train_loader, model, criterion, optimizer, compression_algo, epoch, config,
                    is_inception)

        # update compression scheduler state at the end of the epoch
        compression_algo.scheduler.epoch_step()

        # compute compression algo statistics
        stats = compression_algo.statistics()

        acc1 = best_acc1
        if epoch % config.test_every_n_epochs == 0:
            # evaluate on validation set
            acc1, _ = validate(val_loader, model, criterion, config)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_main_process():
            print_statistics(stats)

            checkpoint_path = osp.join(config.checkpoint_save_dir, get_name(config) + '_last.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'arch': model_name,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': compression_algo.scheduler.state_dict()
            }

            torch.save(checkpoint, checkpoint_path)
            make_additional_checkpoints(checkpoint_path, is_best, epoch + 1, config)

            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    config.tb.add_scalar("compression/statistics/{0}".format(key), value, len(train_loader) * epoch)


def resume_from_checkpoint(resuming_checkpoint, model, config, optimizer, compression_algo):
    best_acc1 = 0
    if osp.isfile(resuming_checkpoint):
        print("=> loading checkpoint '{}'".format(resuming_checkpoint))
        checkpoint = torch.load(resuming_checkpoint, map_location='cpu')
        load_state(model, checkpoint['state_dict'], is_resume=True)
        if config.mode.lower() == 'train' and config.to_onnx is None:
            config.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            compression_algo.scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch: {}, best_acc1: {:.3f})"
                  .format(resuming_checkpoint, checkpoint['epoch'], best_acc1))
        else:
            print("=> loaded checkpoint '{}'".format(resuming_checkpoint))
    else:
        raise FileNotFoundError("no checkpoint found at '{}'".format(resuming_checkpoint))
    return model, config, optimizer, compression_algo, best_acc1


def get_dataset(dataset_config, config, transform, is_train):
    if dataset_config == 'imagenet':
        prefix = 'train' if is_train else 'val'
        return datasets.ImageFolder(osp.join(config.dataset_dir, prefix), transform)
    return create_cifar(config, dataset_config, is_train, transform)


def create_cifar(config, dataset_config, is_train, transform):
    create_cifar_fn = None
    if dataset_config == 'cifar100':
        create_cifar_fn = partial(CIFAR100, config.dataset_dir, train=is_train, transform=transform)
    if dataset_config == 'cifar10':
        create_cifar_fn = partial(CIFAR10, config.dataset_dir, train=is_train, transform=transform)
    if create_cifar_fn:
        return safe_thread_call(partial(create_cifar_fn, download=True), partial(create_cifar_fn, download=False))
    return None


def create_dataloaders(config):
    dataset_config = config.dataset if config.dataset is not None else 'imagenet'
    dataset_config = dataset_config.lower()
    assert dataset_config in ['imagenet', 'cifar100', 'cifar10'], "Unknown dataset option"

    if dataset_config == 'imagenet':
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
    elif dataset_config == 'cifar100':
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                         std=(0.2023, 0.1994, 0.2010))
    elif dataset_config == 'cifar10':
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                         std=(0.5, 0.5, 0.5))

    model_size = config["input_sample_size"][-1]
    size = int(model_size / 0.875)
    val_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(model_size),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = get_dataset(dataset_config, config, val_transform, is_train=False)

    pin_memory = config.execution_mode != ExecutionMode.CPU_ONLY

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    batch_size = int(config.batch_size)
    workers = int(config.workers)
    if config.execution_mode == ExecutionMode.MULTIPROCESSING_DISTRIBUTED:
        batch_size //= config.ngpus_per_node
        workers //= config.ngpus_per_node

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=pin_memory)

    if config.mode.lower() == "test":
        return None, None, val_loader

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(model_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = get_dataset(dataset_config, config, train_transforms, is_train=True)

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=pin_memory, sampler=train_sampler)
    return train_loader, train_sampler, val_loader


def train_epoch(train_loader, model, criterion, optimizer, compression_algo, epoch, config,
                is_inception=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    compression_losses = AverageMeter()
    criterion_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    compression_scheduler = compression_algo.scheduler

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

        # compute compression loss
        compression_loss = compression_algo.loss()
        loss = criterion_loss + compression_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input_.size(0))
        comp_loss_val = compression_loss.item() if isinstance(compression_loss, torch.Tensor) else compression_loss
        compression_losses.update(comp_loss_val, input_.size(0))
        criterion_losses.update(criterion_loss.item(), input_.size(0))
        top1.update(acc1, input_.size(0))
        top5.update(acc5, input_.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        compression_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print(
                '{rank}: '
                'Epoch: [{0}][{1}/{2}] '
                'Lr: {3:.3} '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                'CE_loss: {ce_loss.val:.4f} ({ce_loss.avg:.4f}) '
                'CR_loss: {cr_loss.val:.4f} ({cr_loss.avg:.4f}) '
                'Loss: {loss.val:.4f} ({loss.avg:.4f}) '
                'Acc@1: {top1.val:.3f} ({top1.avg:.3f}) '
                'Acc@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), get_lr(optimizer), batch_time=batch_time,
                    data_time=data_time, ce_loss=criterion_losses, cr_loss=compression_losses,
                    loss=losses, top1=top1, top5=top5,
                    rank='{}:'.format(config.rank) if config.multiprocessing_distributed else ''
                ))

        if is_main_process():
            global_step = len(train_loader) * epoch
            config.tb.add_scalar("train/learning_rate", get_lr(optimizer), i + global_step)
            config.tb.add_scalar("train/criterion_loss", criterion_losses.avg, i + global_step)
            config.tb.add_scalar("train/compression_loss", compression_losses.avg, i + global_step)
            config.tb.add_scalar("train/loss", losses.avg, i + global_step)
            config.tb.add_scalar("train/top1", top1.avg, i + global_step)
            config.tb.add_scalar("train/top5", top5.avg, i + global_step)

            for stat_name, stat_value in compression_algo.statistics().items():
                if isinstance(stat_value, (int, float)):
                    config.tb.add_scalar('train/statistics/{}'.format(stat_name), stat_value, i + global_step)


def validate(val_loader, model, criterion, config):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input_, target) in enumerate(val_loader):
            input_ = input_.to(config.device)
            target = target.to(config.device)

            # compute output
            output = model(input_)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss, input_.size(0))
            top1.update(acc1, input_.size(0))
            top5.update(acc5, input_.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                print(
                    '{rank}'
                    'Test: [{0}/{1}] '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:.4f} ({loss.avg:.4f}) '
                    'Acc@1: {top1.val:.3f} ({top1.avg:.3f}) '
                    'Acc@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5,
                        rank='{}:'.format(config.rank) if config.multiprocessing_distributed else ''
                    ))

        if is_main_process():
            config.tb.add_scalar("val/loss", losses.avg, len(val_loader) * config.get('cur_epoch', 0))
            config.tb.add_scalar("val/top1", top1.avg, len(val_loader) * config.get('cur_epoch', 0))
            config.tb.add_scalar("val/top5", top5.avg, len(val_loader) * config.get('cur_epoch', 0))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


if __name__ == '__main__':
    main(sys.argv[1:])
