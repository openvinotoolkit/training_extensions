"""
 MIT License

 Copyright (c) 2018 Kaiyang Zhou

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

import torch

from torchreid.data.datamanager import DataManager
from torchreid.data.datasets import __image_datasets

from .datasets.globalme import GlobalMe
from .transforms import build_transforms
from .sampler import build_train_sampler


__image_datasets['globalme'] = GlobalMe


def init_image_dataset(name, **kwargs):
    """Initializes an image dataset."""
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError('Invalid dataset name. Received "{}", '
                         'but expected to be one of {}'.format(name, avai_datasets))
    return __image_datasets[name](**kwargs)


class ImageDataManagerWithTransforms(DataManager):
    data_type = 'image'

    def __init__(self, root='', sources=None, targets=None, height=256, width=128, transforms='random_flip',
                 norm_mean=None, norm_std=None, use_gpu=True, split_id=0, combineall=False,
                 batch_size_train=32, batch_size_test=32, workers=4, num_instances=4, train_sampler='',
                 cuhk03_labeled=False, cuhk03_classic_split=False, market1501_500k=False, apply_masks_to_test=False):

        super(ImageDataManagerWithTransforms, self).__init__(
            sources=sources, targets=targets, height=height, width=width,
            transforms=None, norm_mean=norm_mean, norm_std=norm_std, use_gpu=use_gpu
        )

        self.transform_tr, self.transform_te = build_transforms(
            self.height, self.width, transforms=transforms,
            norm_mean=norm_mean, norm_std=norm_std,
            apply_masks_to_test=apply_masks_to_test
        )

        print('=> Loading train (source) dataset')
        trainset = []
        for name in self.sources:
            trainset_ = init_image_dataset(
                name,
                transform=self.transform_tr,
                mode='train',
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            trainset.append(trainset_)
        trainset = sum(trainset)

        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams

        train_sampler = build_train_sampler(
            trainset.train, train_sampler,
            batch_size=batch_size_train,
            num_instances=num_instances
        )

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )

        print('=> Loading test (target) dataset')
        self.testloader = {name: {'query': None, 'gallery': None} for name in self.targets}
        self.testdataset = {name: {'query': None, 'gallery': None} for name in self.targets}

        for name in self.targets:
            # build query loader
            queryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode='query',
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            self.testloader[name]['query'] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            # build gallery loader
            galleryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode='gallery',
                combineall=combineall,
                verbose=False,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            self.testloader[name]['gallery'] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.testdataset[name]['query'] = queryset.query
            self.testdataset[name]['gallery'] = galleryset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  train            : {}'.format(self.sources))
        print('  # train datasets : {}'.format(len(self.sources)))
        print('  # train ids      : {}'.format(self.num_train_pids))
        print('  # train images   : {}'.format(len(trainset)))
        print('  # train cameras  : {}'.format(self.num_train_cams))
        print('  test             : {}'.format(self.targets))
        print('  *****************************************')
        print('\n')
