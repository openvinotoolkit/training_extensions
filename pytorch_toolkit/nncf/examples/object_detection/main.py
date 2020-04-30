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


import os.path as osp
import sys
import time
from pathlib import Path

import torch
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau

from examples.common.argparser import get_common_argument_parser
from examples.common.distributed import DistributedSampler, configure_distributed
from examples.common.execution import ExecutionMode, get_device, get_execution_mode
from examples.common.execution import prepare_model_for_execution, start_worker
from examples.common.utils import get_name, make_additional_checkpoints, print_statistics
from examples.common.utils import write_metrics
from examples.common.optimizer import get_parameter_groups, make_optimizer
from examples.common.utils import configure_logging, configure_paths, create_code_snapshot, is_on_first_rank, print_args
from examples.object_detection.dataset import detection_collate, get_testing_dataset, get_training_dataset
from examples.object_detection.eval import test_net
from examples.object_detection.layers.modules import MultiBoxLoss
from examples.object_detection.model import build_ssd
from examples.common.example_logger import logger


from nncf import Config, create_compressed_model, load_state

from nncf.dynamic_graph.graph_builder import create_input_infos

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_option(args, config, key, default=None):
    """Gets key option from args if it is provided, otherwise tries to get it from config"""
    if hasattr(args, key) and getattr(args, key) is not None:
        return getattr(args, key)
    return config.get(key, default)


def get_argument_parser():
    parser = get_common_argument_parser()

    parser.add_argument('--basenet', default='', help='pretrained base model, should be located in save_folder')
    parser.add_argument('--test-interval', default=5000, type=int, help='test interval')
    parser.add_argument("--dataset", help="Dataset to use.", choices=["voc", "coco"], default=None)
    parser.add_argument('--train_imgs', help='path to training images or VOC root directory')
    parser.add_argument('--train_anno', help='path to training annotations or VOC root directory')
    parser.add_argument('--test_imgs', help='path to testing images or VOC root directory')
    parser.add_argument('--test_anno', help='path to testing annotations or VOC root directory')
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(args=argv)
    config = Config.from_json(args.config)
    config.update_from_args(args, parser)
    configure_paths(config)
    source_root = Path(__file__).absolute().parents[2]  # nncf root
    create_code_snapshot(source_root, osp.join(config.log_dir, "snapshot.tar.gz"))

    config.execution_mode = get_execution_mode(config)

    if config.dataset_dir is not None:
        config.train_imgs = config.train_anno = config.test_imgs = config.test_anno = config.dataset_dir
    start_worker(main_worker, config)


def main_worker(current_gpu, config):
    #################################
    # Setup experiment environment
    #################################
    config.current_gpu = current_gpu
    config.distributed = config.execution_mode in (ExecutionMode.DISTRIBUTED, ExecutionMode.MULTIPROCESSING_DISTRIBUTED)
    if config.distributed:
        configure_distributed(config)
    if is_on_first_rank(config):
        configure_logging(logger, config)
        print_args(config)

    config.device = get_device(config)
    config.start_iter = 0

    ##########################
    # Prepare metrics log file
    ##########################

    if config.metrics_dump is not None:
        write_metrics(0, config.metrics_dump)

    ##################
    # Prepare model
    ##################

    compression_ctrl, net = create_model(config)
    if config.distributed:
        config.batch_size //= config.ngpus_per_node
        config.workers //= config.ngpus_per_node
        compression_ctrl.distributed()

    ###########################
    # Criterion and optimizer
    ###########################

    params_to_optimize = get_parameter_groups(net, config)
    optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)

    criterion = MultiBoxLoss(
        config,
        config['num_classes'],
        overlap_thresh=0.5,
        prior_for_matching=True,
        bkg_label=0,
        neg_mining=True,
        neg_pos=3,
        neg_overlap=0.5,
        encode_target=False,
        device=config.device
    )

    ###########################
    # Load checkpoint
    ###########################

    resuming_checkpoint = config.resuming_checkpoint
    if resuming_checkpoint:
        logger.info('Resuming training, loading {}...'.format(resuming_checkpoint))
        checkpoint = torch.load(resuming_checkpoint, map_location='cpu')
        # use checkpoint itself in case of only state dict is saved
        # i.e. checkpoint is created with `torch.save(module.state_dict())`
        state_dict = checkpoint.get('state_dict', checkpoint)
        load_state(net, state_dict, is_resume=True)
        if config.mode.lower() == 'train' and config.to_onnx is None:
            compression_ctrl.scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint.get('optimizer', optimizer.state_dict()))
            config.start_iter = checkpoint.get('iter', 0) + 1

    if config.to_onnx:
        compression_ctrl.export_model(config.to_onnx)
        logger.info("Saved to {}".format(config.to_onnx))
        return

    ###########################
    # Prepare data
    ###########################

    test_data_loader, train_data_loader = create_dataloaders(config)

    if config.mode.lower() == 'test':
        with torch.no_grad():
            print_statistics(compression_ctrl.statistics())
            net.eval()
            mAp = test_net(net, config.device, test_data_loader, distributed=config.distributed)
            if config.metrics_dump is not None:
                write_metrics(mAp, config.metrics_dump)
            return

    if not resuming_checkpoint:
        compression_ctrl.initialize(train_data_loader)

    train(net, compression_ctrl, train_data_loader, test_data_loader, criterion, optimizer, config, lr_scheduler)


def create_dataloaders(config):
    logger.info('Loading Dataset...')
    train_dataset = get_training_dataset(config.dataset, config.train_anno, config.train_imgs, config)
    logger.info("Loaded {} training images".format(len(train_dataset)))
    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=config.ngpus_per_node,
                                                                        rank=config.rank)
    else:
        train_sampler = None
    train_data_loader = data.DataLoader(
        train_dataset, config.batch_size,
        num_workers=config.workers,
        shuffle=(train_sampler is None),
        collate_fn=detection_collate,
        pin_memory=True,
        sampler=train_sampler
    )
    test_dataset = get_testing_dataset(config.dataset, config.test_anno, config.test_imgs, config)
    logger.info("Loaded {} testing images".format(len(test_dataset)))
    if config.distributed:
        test_sampler = DistributedSampler(test_dataset, config.rank, config.world_size)
    else:
        test_sampler = None
    test_data_loader = data.DataLoader(
        test_dataset, config.batch_size,
        num_workers=config.workers,
        shuffle=False,
        collate_fn=detection_collate,
        pin_memory=True,
        drop_last=False,
        sampler=test_sampler
    )
    return test_data_loader, train_data_loader


def create_model(config):
    input_info_list = create_input_infos(config)
    image_size = input_info_list[0].shape[-1]
    ssd_net = build_ssd(config.model, config.ssd_params, image_size, config.num_classes, config)
    compression_ctrl, ssd_net = create_compressed_model(ssd_net, config)
    weights = config.get('weights')
    if weights:
        sd = torch.load(weights, map_location='cpu')
        load_state(ssd_net, sd)
    ssd_net.train()
    model, _ = prepare_model_for_execution(ssd_net, config)
    return compression_ctrl, model


def train_step(batch_iterator, compression_ctrl, config, criterion, net, train_data_loader):
    batch_loss_l = torch.tensor(0.).to(config.device)
    batch_loss_c = torch.tensor(0.).to(config.device)
    batch_loss = torch.tensor(0.).to(config.device)
    for _ in range(0, config.iter_size):
        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            logger.debug("StopIteration: can not load batch")
            batch_iterator = iter(train_data_loader)
            break

        images = images.to(config.device)
        targets = [anno.requires_grad_(False).to(config.device) for anno in targets]

        # forward
        out = net(images)
        # backprop
        loss_l, loss_c = criterion(out, targets)
        loss_comp = compression_ctrl.loss()
        loss = loss_l + loss_c + loss_comp
        batch_loss += loss
        loss.backward()
        batch_loss_l += loss_l
        batch_loss_c += loss_c
    return batch_iterator, batch_loss, batch_loss_c, batch_loss_l, loss_comp


def train(net, compression_ctrl, train_data_loader, test_data_loader, criterion, optimizer, config, lr_scheduler):
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0

    epoch_size = len(train_data_loader)
    logger.info('Training {} on {} dataset...'.format(config.model, train_data_loader.dataset.name))
    batch_iterator = None

    t_start = time.time()
    print_statistics(compression_ctrl.statistics())

    best_mAp = 0
    test_freq_in_epochs = max(config.test_interval // epoch_size, 1)

    for iteration in range(config.start_iter, config['max_iter']):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(train_data_loader)

        epoch = iteration // epoch_size

        if (iteration + 1) % epoch_size == 0:
            compression_ctrl.scheduler.epoch_step(epoch)

            is_best = False

            if (epoch + 1) % test_freq_in_epochs == 0:
                if is_on_first_rank(config):
                    print_statistics(compression_ctrl.statistics())
                with torch.no_grad():
                    net.eval()
                    mAP = test_net(net, config.device, test_data_loader, distributed=config.multiprocessing_distributed)
                    if mAP > best_mAp:
                        is_best = True
                        best_mAp = mAP
                    net.train()

            # Learning rate scheduling should be applied after optimizer’s update
            if not isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(epoch)
            else:
                lr_scheduler.step(mAP)

            if is_on_first_rank(config):
                logger.info('Saving state, iter: {}'.format(iteration))

                checkpoint_file_path = osp.join(config.checkpoint_save_dir, "{}_last.pth".format(get_name(config)))
                torch.save({
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter': config['max_iter'],
                    'scheduler': compression_ctrl.scheduler.state_dict()
                }, str(checkpoint_file_path))
                make_additional_checkpoints(checkpoint_file_path,
                                            is_best=is_best,
                                            epoch=epoch + 1,
                                            config=config)

        compression_ctrl.scheduler.step(iteration - config.start_iter)

        optimizer.zero_grad()
        batch_iterator, batch_loss, batch_loss_c, batch_loss_l, loss_comp = train_step(
            batch_iterator, compression_ctrl, config, criterion, net, train_data_loader
        )
        optimizer.step()


        batch_loss_l = batch_loss_l / config.iter_size
        batch_loss_c = batch_loss_c / config.iter_size
        model_loss = (batch_loss_l + batch_loss_c) / config.iter_size
        batch_loss = batch_loss / config.iter_size

        loc_loss += batch_loss_l.item()
        conf_loss += batch_loss_c.item()

        ###########################
        # Logging
        ###########################

        if is_on_first_rank(config):
            config.tb.add_scalar("train/loss_l", batch_loss_l.item(), iteration)
            config.tb.add_scalar("train/loss_c", batch_loss_c.item(), iteration)
            config.tb.add_scalar("train/loss", batch_loss.item(), iteration)

        if iteration % config.print_freq == 0:
            t_finish = time.time()
            t_elapsed = t_finish - t_start
            t_start = time.time()
            logger.info('{}: iter {} epoch {} || Loss: {:.4} || Time {:.4}s || lr: {} || CR loss: {}'.format(
                config.rank, iteration, epoch, model_loss.item(), t_elapsed, optimizer.param_groups[0]['lr'],
                loss_comp.item() if isinstance(loss_comp, torch.Tensor) else loss_comp
            ))

    if config.metrics_dump is not None:
        write_metrics(best_mAp, config.metrics_dump)


if __name__ == '__main__':
    main(sys.argv[1:])
