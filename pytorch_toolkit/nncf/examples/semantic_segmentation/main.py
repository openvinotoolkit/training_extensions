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

# Major parts of this sample reuse code from:
# https://github.com/davidtvs/PyTorch-ENet
# https://github.com/pytorch/vision/tree/master/references/segmentation

import functools
import os
import sys
from os import path as osp

import numpy as np
import torch
import torchvision.transforms as T
from nncf.utils import is_main_process
from torch.optim.lr_scheduler import ReduceLROnPlateau

import examples.semantic_segmentation.utils.data as data_utils
import examples.semantic_segmentation.utils.loss_funcs as loss_funcs
import examples.semantic_segmentation.utils.transforms as JT

from examples.common.argparser import get_common_argument_parser
from examples.common.distributed import configure_distributed
from examples.common.execution import ExecutionMode, get_device, get_execution_mode, \
    prepare_model_for_execution, start_worker
from examples.common.model_loader import load_model
from examples.common.optimizer import make_optimizer
from examples.common.utils import configure_logging, configure_paths, make_additional_checkpoints, print_args, \
    write_metrics, print_statistics
from examples.semantic_segmentation.metric import IoU
from examples.semantic_segmentation.test import Test
from examples.semantic_segmentation.train import Train
from examples.semantic_segmentation.utils.checkpoint import load_checkpoint, save_checkpoint
from examples.common.example_logger import logger
from nncf import Config, create_compressed_model, load_state

def get_arguments(args):
    parser = get_common_argument_parser()
    parser.add_argument(
        "--dataset",
        help="Dataset to use.",
        choices=["camvid", "cityscapes", "mapillary"],
        default=None
    )
    return parser.parse_args(args=args)


def get_preprocessing_transforms(config):
    transforms = []
    for k, v in config.preprocessing.items():
        if k == "resize":
            transforms.append(JT.Resize((v["height"], v["width"])))
    return transforms


def get_augmentations_transforms(config):
    transforms = []
    for k, v in config.augmentations.items():
        if k == "random_hflip":
            transforms.append(JT.RandomHorizontalFlip(v))
        elif k == "random_crop":
            transforms.append(JT.RandomCrop(v))
        elif k == "random_resize":
            transforms.append(JT.RandomResize(v["min_size"], v["max_size"]))
        elif k == "random_scale_aligned":
            transforms.append(JT.RandomScaleAligned(**v))
        elif k == "resize":
            transforms.append(JT.Resize((v["height"], v["width"])))
        elif k == "random_sized_crop":
            transforms.append(JT.RandomSizedCrop(v))
    return transforms


def get_joint_transforms(is_train, config):
    joint_transforms = []
    if is_train and "augmentations" in config:
        joint_transforms += get_augmentations_transforms(config)

    if "preprocessing" in config:
        joint_transforms += get_preprocessing_transforms(config)
        joint_transforms.append(JT.ToTensor())
        if "normalize" in config["preprocessing"]:
            v = config["preprocessing"]["normalize"]
            joint_transforms.append(JT.Normalize(v["mean"], v["std"]))
    else:
        joint_transforms.append(JT.ToTensor())
    return JT.Compose(joint_transforms)


def get_class_weights(train_set, num_classes, config):
    # Get class weights from the selected weighing technique
    logger.info("\nWeighing technique: {}".format(config.weighing))
    weighing = config.get('weighing', 'none')
    if isinstance(weighing, list):
        # Class weights were directly specified in config
        return np.asarray(weighing)

    train_loader_for_weight_count = torch.utils.data.DataLoader(
        train_set,
        batch_size=1, collate_fn=data_utils.collate_fn)
    logger.info("Computing class weights...")
    logger.info("(this can take a while depending on the dataset size)")
    if weighing.lower() == 'enet':
        class_weights = data_utils.enet_weighing(train_loader_for_weight_count, num_classes)
    elif weighing.lower() == 'mfb':
        class_weights = data_utils.median_freq_balancing(train_loader_for_weight_count, num_classes)
    else:
        class_weights = None
    return class_weights


def get_dataset(dataset_name: str) -> torch.utils.data.Dataset:
    # Import the requested dataset
    if dataset_name.lower() == 'camvid':
        from examples.semantic_segmentation.datasets import CamVid as dataset
        # Remove the road_marking class from the CamVid dataset as it's merged
        # with the road class
        if 'road_marking' in dataset.color_encoding:
            del dataset.color_encoding['road_marking']
    elif dataset_name.lower() == 'cityscapes':
        from examples.semantic_segmentation.datasets import Cityscapes as dataset
    elif dataset_name.lower() == 'mapillary':
        from examples.semantic_segmentation.datasets import Mapillary as dataset
    else:
        # Should never happen...but just in case it does
        raise RuntimeError("\"{0}\" is not a supported dataset.".format(
            dataset_name))
    return dataset


def load_dataset(dataset, config):
    logger.info("\nLoading dataset...\n")

    logger.info("Selected dataset: {}".format(config.dataset))
    logger.info("Dataset directory: {}".format(config.dataset_dir))

    transforms_train = get_joint_transforms(is_train=True, config=config)
    transforms_val = get_joint_transforms(is_train=False, config=config)

    # Get selected dataset
    train_set = dataset(
        root=config.dataset_dir,
        image_set='train',
        transforms=transforms_train)

    val_set = dataset(
        config.dataset_dir,
        image_set='val',
        transforms=transforms_val)

    # Samplers
    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_set)

    batch_size = config.batch_size
    num_workers = config.workers

    if config.multiprocessing_distributed:
        batch_size //= config.ngpus_per_node
        num_workers //= config.ngpus_per_node

    # Loaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers,
        collate_fn=data_utils.collate_fn, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1, num_workers=num_workers,
        collate_fn=data_utils.collate_fn)

    # Get encoding between pixel values in label images and RGB colors
    class_encoding = train_set.color_encoding

    # Get number of classes to predict
    num_classes = len(class_encoding)

    # Print information for debugging
    logger.info("Number of classes to predict: {}".format(num_classes))
    logger.info("Train dataset size: {}".format(len(train_set)))
    logger.info("Validation dataset size: {}".format(len(val_set)))

    # Get a batch of samples to display
    if config.mode.lower() == 'test':
        images, labels = iter(val_loader).next()
    else:
        images, labels = iter(train_loader).next()
    logger.info("Image size: {}".format(images.size()))
    logger.info("Label size: {}".format(labels.size()))
    logger.info("Class-color encoding: {}".format(class_encoding))

    # Show a batch of samples and labels
    if config.imshow_batch and config.mode.lower() != 'test':
        logger.info("Close the figure window to continue...")
        label_to_rgb = T.Compose([
            data_utils.LongTensorToRGBPIL(class_encoding),
            T.ToTensor()
        ])
        color_labels = data_utils.batch_transform(labels, label_to_rgb)
        data_utils.imshow_batch(images, color_labels)

    class_weights = get_class_weights(train_set, num_classes, config)

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(config.device)
        # Set the weight of the unlabeled class to 0
        ignore_unlabeled = config.get("ignore_unlabeled", True)
        if ignore_unlabeled and ('unlabeled' in class_encoding):
            ignore_index = list(class_encoding).index('unlabeled')
            class_weights[ignore_index] = 0

    logger.info("Class weights: {}".format(class_weights))

    return (train_loader, val_loader), class_weights


def get_aux_loss_dependent_params(model_without_dp, class_weights, aux_lr, config):
    if config.model == "icnet":
        params_to_optimize = model_without_dp.parameters()
        criterion = functools.partial(loss_funcs.cross_entropy_icnet, weight=class_weights)
        return params_to_optimize, criterion

    model_params_config = config.get('model_params', {})
    is_aux_loss = model_params_config.get('aux_loss', False)
    if is_aux_loss:
        params_to_optimize = [
            {"params": [p for p in model_without_dp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_dp.classifier.parameters() if p.requires_grad]},
        ]
        params = [p for p in model_without_dp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": aux_lr})
        criterion = functools.partial(loss_funcs.cross_entropy_aux, weight=class_weights)
    else:
        params_to_optimize = model_without_dp.parameters()
        criterion = functools.partial(loss_funcs.cross_entropy, weight=class_weights)
    return params_to_optimize, criterion


# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
def train(model, model_without_dp, compression_ctrl, train_loader, val_loader, class_weights, class_encoding, config):
    logger.info("\nTraining...\n")

    # Check if the network architecture is correct
    logger.info(model)

    optim_config = config.get('optimizer', {})
    optim_params = optim_config.get('optimizer_params', {})
    lr = optim_params.get("lr", 1e-4)

    params_to_optimize, criterion = get_aux_loss_dependent_params(model_without_dp,
                                                                  class_weights,
                                                                  lr * 10,
                                                                  config)

    optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)

    # Evaluation metric

    ignore_index = None
    ignore_unlabeled = config.get("ignore_unlabeled", True)
    if ignore_unlabeled and ('unlabeled' in class_encoding):
        ignore_index = list(class_encoding).index('unlabeled')

    metric = IoU(len(class_encoding), ignore_index=ignore_index)

    best_miou = -1
    resuming_checkpoint = config.resuming_checkpoint
    # Optionally resume from a checkpoint
    if resuming_checkpoint is not None:
        model, optimizer, start_epoch, best_miou, _ = \
            load_checkpoint(
                model, resuming_checkpoint, config.device,
                optimizer, compression_ctrl.scheduler)
        logger.info("Resuming from model: Start epoch = {0} "
                    "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
        config.start_epoch = start_epoch

    # Start Training
    train_obj = Train(model, train_loader, optimizer, criterion, compression_ctrl, metric, config.device,
                      config.model)
    val_obj = Test(model, val_loader, criterion, metric, config.device,
                   config.model)

    for epoch in range(config.start_epoch, config.epochs):
        logger.info(">>>> [Epoch: {0:d}] Training".format(epoch))

        if config.distributed:
            train_loader.sampler.set_epoch(epoch)

        epoch_loss, (iou, miou) = train_obj.run_epoch(config.print_step)
        if not isinstance(lr_scheduler, ReduceLROnPlateau):
            # Learning rate scheduling should be applied after optimizer’s update
            lr_scheduler.step(epoch)
        compression_ctrl.scheduler.epoch_step()

        logger.info(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                    format(epoch, epoch_loss, miou))

        if is_main_process():
            config.tb.add_scalar("train/loss", epoch_loss, epoch)
            config.tb.add_scalar("train/mIoU", miou, epoch)
            config.tb.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], epoch)
            config.tb.add_scalar("train/compression_loss", compression_ctrl.loss(), epoch)

            for key, value in compression_ctrl.statistics().items():
                if isinstance(value, (int, float)):
                    config.tb.add_scalar("compression/statistics/{0}".format(key), value, epoch)

        if (epoch + 1) % config.save_freq == 0 or epoch + 1 == config.epochs:
            logger.info(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss, (iou, miou) = val_obj.run_epoch(config.print_step)

            logger.info(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                        format(epoch, loss, miou))

            if is_main_process():
                config.tb.add_scalar("val/mIoU", miou, epoch)
                config.tb.add_scalar("val/loss", loss, epoch)
                for i, (key, class_iou) in enumerate(zip(class_encoding.keys(), iou)):
                    config.tb.add_scalar("{}/mIoU_Cls{}_{}".format(config.dataset, i, key), class_iou, epoch)

            is_best = miou > best_miou
            best_miou = max(miou, best_miou)

            if config.metrics_dump is not None:
                write_metrics(best_miou, config.metrics_dump)

            if isinstance(lr_scheduler, ReduceLROnPlateau):
                # Learning rate scheduling should be applied after optimizer’s update
                lr_scheduler.step(best_miou)

            # Print per class IoU on last epoch or if best iou
            if epoch + 1 == config.epochs or is_best:
                for key, class_iou in zip(class_encoding.keys(), iou):
                    logger.info("{0}: {1:.4f}".format(key, class_iou))

            # Save the model if it's the best thus far
            if is_main_process():
                checkpoint_path = save_checkpoint(model,
                                                  optimizer, epoch + 1, best_miou,
                                                  compression_ctrl.scheduler, config)

                make_additional_checkpoints(checkpoint_path, is_best, epoch + 1, config)
                print_statistics(compression_ctrl.statistics())

    return model


def test(model, test_loader, class_weights, class_encoding, config):
    logger.info("\nTesting...\n")

    _, criterion = get_aux_loss_dependent_params(model,
                                                 class_weights,
                                                 0,
                                                 config)

    # Evaluation metric

    ignore_index = None

    ignore_unlabeled = config.get("ignore_unlabeled", True)
    if ignore_unlabeled and ('unlabeled' in class_encoding):
        ignore_index = list(class_encoding).index('unlabeled')

    metric = IoU(len(class_encoding), ignore_index=ignore_index)

    # Test the trained model on the test set
    test_obj = Test(model, test_loader, criterion, metric, config.device, config.model)

    logger.info(">>>> Running test dataset")

    loss, (iou, miou) = test_obj.run_epoch(config.print_step)
    class_iou = dict(zip(class_encoding.keys(), iou))

    logger.info(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))
    if config.metrics_dump is not None:
        write_metrics(miou, config.metrics_dump)

    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        logger.info("{0}: {1:.4f}".format(key, class_iou))

    # Show a batch of samples and labels
    if config.imshow_batch:
        logger.info("A batch of predictions from the test set...")
        images, gt_labels = iter(test_loader).next()
        color_predictions = predict(model, images, class_encoding, config)

        from examples.common.models.segmentation.unet import UNet, center_crop
        if isinstance(model, UNet):
            # UNet predicts center image crops
            outputs_size_hw = (color_predictions.size()[2], color_predictions.size()[3])
            gt_labels = center_crop(gt_labels, outputs_size_hw).contiguous()
        data_utils.show_ground_truth_vs_prediction(images, gt_labels, color_predictions, class_encoding)


def predict(model, images, class_encoding, config):
    images = images.to(config.device)

    model.eval()
    with torch.no_grad():
        predictions = model(images)

    if isinstance(predictions, dict):
        predictions = predictions['out']

    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)

    color_predictions = data_utils.label_to_color(predictions, class_encoding)
    return color_predictions


def main_worker(current_gpu, config):
    config.current_gpu = current_gpu
    config.distributed = config.execution_mode in (ExecutionMode.DISTRIBUTED, ExecutionMode.MULTIPROCESSING_DISTRIBUTED)
    if config.distributed:
        configure_distributed(config)

    if is_main_process():
        configure_logging(logger, config)
        print_args(config)

    logger.info(config)

    config.device = get_device(config)
    dataset = get_dataset(config.dataset)
    color_encoding = dataset.color_encoding
    num_classes = len(color_encoding)

    if config.metrics_dump is not None:
        write_metrics(0, config.metrics_dump)

    weights = config.get('weights')
    model = load_model(config.model,
                       pretrained=config.get('pretrained', True) if weights is None else False,
                       num_classes=num_classes,
                       model_params=config.get('model_params', {}))
    compression_ctrl, model = create_compressed_model(model, config)
    if weights:
        sd = torch.load(weights, map_location='cpu')
        load_state(model, sd)

    model, model_without_dp = prepare_model_for_execution(model, config)

    if config.distributed:
        compression_ctrl.distributed()

    resuming_checkpoint = config.resuming_checkpoint

    if resuming_checkpoint is not None:
        if not config.pretrained:
            # Load the previously saved model state
            model, _, _, _, _ = \
                load_checkpoint(model, resuming_checkpoint, config.device,
                                compression_scheduler=compression_ctrl.scheduler)

    if config.to_onnx is not None:
        compression_ctrl.export_model(config.to_onnx)
        logger.info("Saved to {}".format(config.to_onnx))
        return

    if config.mode.lower() == 'test':
        logger.info(model)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info("Trainable argument count:{params}".format(params=params))

        model = model.to(config.device)
        loaders, w_class = load_dataset(dataset, config)
        _, val_loader = loaders
        test(model, val_loader, w_class, color_encoding, config)
        print_statistics(compression_ctrl.statistics())
    elif config.mode.lower() == 'train':
        loaders, w_class = load_dataset(dataset, config)
        train_loader, val_loader = loaders
        if not resuming_checkpoint:
            compression_ctrl.initialize(train_loader)
        train(model, model_without_dp, compression_ctrl, train_loader, val_loader, w_class, color_encoding, config)
    else:
        # Should never happen...but just in case it does
        raise RuntimeError(
            "\"{0}\" is not a valid choice for execution mode.".format(
                config.mode))


def main(argv):
    parser = get_common_argument_parser()
    arguments = parser.parse_args(args=argv)
    config = Config.from_json(arguments.config)
    config.update_from_args(arguments, parser)
    if config.dist_url == "env://":
        config.update_from_env()

    if config.mode.lower() != 'test':
        if not osp.exists(config.log_dir):
            os.makedirs(config.log_dir)

        config.log_dir = str(config.log_dir)
        configure_paths(config)
        logger.info("Save directory: {}".format(config.log_dir))
    else:
        config.log_dir = "/tmp/"

    config.execution_mode = get_execution_mode(config)
    start_worker(main_worker, config)


if __name__ == '__main__':
    main(sys.argv[1:])
