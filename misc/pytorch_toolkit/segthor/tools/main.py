#!/usr/bin/env python3
#
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
import pickle
import random
import argparse
import gc
from sklearn.model_selection import KFold
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data.dataloader import DataLoader

from segthor import dataloader
from segthor import loss
from segthor import metrics
from segthor import train
from segthor import weight_init
from segthor.model import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch SegTHOR training")

    parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
    parser.add_argument("--nEpochs", type=int, default=500, help="Number of epochs to train for")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use")
    parser.add_argument("--train_path", default="./data", type=str, help="Path to train data", required=True)
    parser.add_argument("--name", default="test", type=str, help="Experiment name")
    parser.add_argument("--models_path", default="./models", type=str, help="Path to models folder")
    parser.add_argument("--splits", default=5, type=int, help="Number of splits in CV")
    parser.add_argument("--gpus", default=4, type=int, help="Number of gpus to use")
    parser.add_argument("--seed", default=1337, type=int, help="Seed for random generators")

    return parser.parse_args()


def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    print('worker id {} seed {}'.format(worker_id, seed))

# pylint: disable=R0914,R0915
def main():
    opt = parse_args()

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    kf = KFold(n_splits=opt.splits)
    splits = []
    evaluation_metrics = []

    print("CV {} splits".format(kf.get_n_splits()))

    series = [f for f in os.listdir(opt.train_path) if os.path.isdir(os.path.join(opt.train_path, f))]
    series.sort()

    for idx, (train_index, test_index) in enumerate(kf.split(series)):

        print(train_index, test_index)

        print("===> Building model")
        layers = [1, 2, 2, 4, 6]
        number_of_channels = [int(4*8*2**i) for i in range(1, 6)]
        model = UNet(depth=len(layers), encoder_layers=layers,
                     number_of_channels=number_of_channels, number_of_outputs=5)
        model.apply(weight_init.weight_init)
        model = torch.nn.DataParallel(module=model, device_ids=range(opt.gpus))

        trainer = train.Trainer(model=model, name=opt.name+str(idx), models_root=opt.models_path, rewrite=False)
        trainer.cuda()

        gc.collect()

        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)

        cudnn.benchmark = True

        print("===> Loading datasets")
        print('Train data:', opt.train_path)


        series_train = [series[i] for i in train_index]
        series_val = [series[i] for i in test_index]
        print('Train {}'.format(series_train))
        print('Val {}'.format(series_val))

        train_set = dataloader.SimpleReader(path=opt.train_path,
                                            patch_size=(16*13, 16*8, 16*5),
                                            series=series_train,
                                            multiplier=500,
                                            patches_from_single_image=1)

        val_set = dataloader.FullReader(path=opt.train_path, series=series_val)

        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
                                          batch_size=opt.batchSize, shuffle=True,
                                          drop_last=True, worker_init_fn=worker_init_fn)

        batch_sampler = Data.BatchSampler(
            sampler=Data.SequentialSampler(val_set),
            batch_size=1,
            drop_last=True
        )

        evaluation_data_loader = DataLoader(dataset=val_set, num_workers=0,
                                            batch_sampler=batch_sampler)

        criterion = [loss.Dice_loss_joint(index=0, priority=1).cuda()]

        print("===> Training")

        trainer.train(criterion=criterion,
                      optimizer=optim.SGD,
                      optimizer_params={"lr":1e-0,
                                        "weight_decay":1e-6,
                                        "momentum":0.9
                                        },
                      scheduler=torch.optim.lr_scheduler.MultiStepLR,
                      scheduler_params={"milestones":[150000, 200000, 240000, 280000],
                                        "gamma":0.2},
                      training_data_loader=training_data_loader,
                      evaluation_data_loader=evaluation_data_loader,
                      pretrained_weights=None,
                      train_metrics=[
                          metrics.Dice(name='Dice', input_index=0, target_index=0),
                      ],
                      val_metrics=[
                          metrics.Dice(name='Dice', input_index=0, target_index=0),
                          metrics.Hausdorff_ITK(name='Hausdorff_ITK', input_index=0, target_index=0)
                      ],
                      track_metric='Dice',
                      epoches=opt.nEpochs,
                      default_val=np.array([0, 0, 0, 0, 0]),
                      comparator=lambda x, y: np.min(x)+np.mean(x) > np.min(y)+np.mean(y),
                      eval_cpu=True
                      )

        evaluation_metrics.append(trainer.state.best_val)
        splits.append((series_train, series_val))


    avg_val = 0

    for i in evaluation_metrics:
        avg_val = avg_val + i

    print('Average val {}'.format(avg_val/len(evaluation_metrics)))

    pickle.dump(evaluation_metrics, open(opt.name+'_eval.p', 'wb'))
    pickle.dump(splits, open(opt.name+'_splits.p', 'wb'))

if __name__ == "__main__":
    main()
