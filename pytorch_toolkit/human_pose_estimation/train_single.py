import argparse
import cv2
import os

import torch
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.lip import LipTrainDataset, LipValDataset
from datasets.transformations import SinglePersonRotate, SinglePersonCropPad, SinglePersonFlip, SinglePersonBodyMasking,\
    ChannelPermutation
from modules.calc_pckh import calc_pckh
from modules.get_parameters import get_parameters_conv, get_parameters_bn, get_parameters_conv_depthwise
from models.single_person_pose_with_mobilenet import SinglePersonPoseEstimationWithMobileNet
from modules.loss import l2_loss
from modules.load_state import load_state, load_from_mobilenet
from val_single import evaluate

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader


def train(images_folder, num_refinement_stages, base_lr, batch_size, batches_per_iter,
          num_workers, checkpoint_path, weights_only, from_mobilenet, checkpoints_folder,
          log_after, checkpoint_after):
    net = SinglePersonPoseEstimationWithMobileNet(num_refinement_stages).cuda()
    stride = 8
    sigma = 7
    dataset = LipTrainDataset(images_folder, stride, sigma,
                              transform=transforms.Compose([
                                   SinglePersonBodyMasking(),
                                   ChannelPermutation(),
                                   SinglePersonRotate(pad=(128, 128, 128), max_rotate_degree=40),
                                   SinglePersonCropPad(pad=(128, 128, 128), crop_x=256, crop_y=256),
                                   SinglePersonFlip(left_keypoints_indice=LipTrainDataset.left_keypoints_indice,
                                                    right_keypoints_indice=LipTrainDataset.right_keypoints_indice)
                              ]))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    optimizer = optim.Adam([
        {'params': get_parameters_conv(net.model, 'weight')},
        {'params': get_parameters_conv_depthwise(net.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.model, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(net.cpm, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(net.cpm, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv_depthwise(net.cpm, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_conv(net.initial_stage, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(net.initial_stage, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_bn(net.initial_stage, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.initial_stage, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(net.refinement_stages, 'weight'), 'lr': base_lr * 4},
        {'params': get_parameters_conv(net.refinement_stages, 'bias'), 'lr': base_lr * 8, 'weight_decay': 0},
        {'params': get_parameters_bn(net.refinement_stages, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.refinement_stages, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
    ], lr=base_lr, weight_decay=5e-4)

    num_iter = 0
    current_epoch = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, threshold=1e-2, verbose=True)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)

        if from_mobilenet:
            load_from_mobilenet(net, checkpoint)
        else:
            load_state(net, checkpoint)
            if not weights_only:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                num_iter = checkpoint['iter']
                current_epoch = checkpoint['current_epoch']+1

    net = DataParallel(net)
    net.train()
    for epochId in range(current_epoch, 100):
        print('Epoch: {}'.format(epochId))
        net.train()
        total_losses = [0] * (num_refinement_stages + 1)  # heatmaps loss per stage
        batch_per_iter_idx = 0
        for batch_data in train_loader:
            if batch_per_iter_idx == 0:
                optimizer.zero_grad()

            images = batch_data['image'].cuda()
            keypoint_maps = batch_data['keypoint_maps'].cuda()

            stages_output = net(images)

            losses = []
            for loss_idx in range(len(total_losses)):
                losses.append(l2_loss(stages_output[loss_idx], keypoint_maps, images.shape[0]))
                total_losses[loss_idx] += losses[-1].item() / batches_per_iter

            loss = losses[0]
            for loss_idx in range(1, len(losses)):
                loss += losses[loss_idx]
            loss /= batches_per_iter
            loss.backward()
            batch_per_iter_idx += 1
            if batch_per_iter_idx == batches_per_iter:
                optimizer.step()
                batch_per_iter_idx = 0
                num_iter += 1
            else:
                continue

            if num_iter % log_after == 0:
                print('Iter: {}'.format(num_iter))
                for loss_idx in range(len(total_losses)):
                    print('\n'.join(['stage{}_heatmaps_loss: {}']).format(
                        loss_idx + 1, total_losses[loss_idx] / log_after))
                for loss_idx in range(len(total_losses)):
                    total_losses[loss_idx] = 0

        snapshot_name = '{}/checkpoint_last_epoch.pth'.format(checkpoints_folder)
        torch.save({'state_dict': net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iter': num_iter,
                    'current_epoch': epochId},
                   snapshot_name)
        if (epochId + 1) % checkpoint_after == 0:
            snapshot_name = '{}/checkpoint_epoch_{}.pth'.format(checkpoints_folder, epochId)
            torch.save({'state_dict': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iter': num_iter,
                        'current_epoch': epochId},
                       snapshot_name)
        print('Validation...')
        net.eval()
        eval_num = 1000
        val_dataset = LipValDataset(images_folder, eval_num)
        predictions_name = '{}/val_results.csv'.format(checkpoints_folder)
        evaluate(val_dataset, predictions_name, net)
        pck = calc_pckh(val_dataset.labels_file_path, predictions_name, eval_num=eval_num)

        val_loss = 100 - pck[-1][-1]  # 100 - avg_pckh
        print('Val loss: {}'.format(val_loss))
        scheduler.step(val_loss, epochId)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-folder', type=str, required=True, help='path to dataset folder')
    parser.add_argument('--num-refinement-stages', type=int, default=5, help='number of refinement stages')
    parser.add_argument('--base-lr', type=float, default=4e-5, help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--batches-per-iter', type=int, default=1, help='number of batches to accumulate gradient from')
    parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint to continue training from')
    parser.add_argument('--from-mobilenet', action='store_true',
                        help='load weights from mobilenet feature extractor')
    parser.add_argument('--weights-only', action='store_true',
                        help='just initialize layers with pretrained weights and start training from the beginning')
    parser.add_argument('--experiment-name', type=str, default='default',
                        help='experiment name to create folder for checkpoints')
    parser.add_argument('--log-after', type=int, default=100, help='number of iterations to print train loss')
    parser.add_argument('--checkpoint-after', type=int, default=10,
                        help='number of epochs to save checkpoint')
    args = parser.parse_args()

    checkpoints_folder = '{}_checkpoints'.format(args.experiment_name)
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    train(args.dataset_folder, args.num_refinement_stages, args.base_lr, args.batch_size,
          args.batches_per_iter, args.num_workers, args.checkpoint_path, args.weights_only, args.from_mobilenet,
          checkpoints_folder, args.log_after, args.checkpoint_after)
