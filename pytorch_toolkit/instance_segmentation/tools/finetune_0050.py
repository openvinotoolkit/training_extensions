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
import resource

from segmentoly.data.dataparallel import collate
from segmentoly.data.transforms import *
from segmentoly.datasets.factory import get_dataset
from segmentoly.rcnn.model_zoo.instance_segmentation_security_0050 import InstanceSegmentationSecurity0050 as Model
from segmentoly.utils.logging import setup_logging, TextLogger, TensorboardLogger
from segmentoly.utils.lr_scheduler import MultiStepLRWithWarmUp
from segmentoly.utils.training_engine import DefaultMaskRCNNTrainingEngine
from segmentoly.utils.weights import load_checkpoint

logger = logging.getLogger(__name__)


class Trainer(DefaultMaskRCNNTrainingEngine):
    def __init__(self):
        super().__init__()
        self.identifier = 'instance-segmentation-security-0050'
        self.description = 'Fine-tuning of instance-segmentation-security-0050'
        self.root_directory = osp.join(osp.dirname(osp.abspath(__file__)), '..')
        self.run_directory = self.create_run_directory(osp.join(self.root_directory, 'outputs'))

        setup_logging(file_path=osp.join(self.run_directory, 'log.txt'))

        logger.info('Running {}'.format(self.identifier))
        logger.info(self.description)
        logger.info('Working directory "{}"'.format(self.run_directory))

        self.batch_size = 32
        self.virtual_iter_size = 1

        # Training dataset.
        training_transforms = Compose(
            [
                RandomResize(mode='size', heights=(416, 448, 480, 512, 544), widths=(416, 448, 480, 512, 544)),
                RandomHorizontalFlip(prob=0.5),
                ToTensor(),
                Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.], rgb=False),
            ],
        )
        training_dataset_name = 'coco_2017_train'
        logger.info('Training dataset {}'.format(training_dataset_name))
        training_dataset = get_dataset(training_dataset_name, True, True, training_transforms)
        logger.info(training_dataset)
        self.training_data_loader = torch.utils.data.DataLoader(
            training_dataset, batch_size=self.batch_size, num_workers=0,
            shuffle=True, drop_last=True, collate_fn=collate
        )

        # Validation datasets.
        validation_transforms = Compose(
            [
                Resize(size=[480, 480]),
                ToTensor(),
                Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.], rgb=False),
            ]
        )
        validation_datasets = []
        validation_dataset_name = 'coco_2017_val'
        logger.info('Validation dataset #{}: {}'.format(len(validation_datasets) + 1, validation_dataset_name))
        validation_datasets.append(get_dataset(validation_dataset_name, False, False, validation_transforms))
        logger.info(validation_datasets[-1])

        self.validation_data_loaders = []
        for validation_dataset in validation_datasets:
            self.validation_data_loaders.append(torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=1, num_workers=8,
                shuffle=False, drop_last=False, collate_fn=collate)
            )
        self.validate_every = 500

        for validation_dataset in validation_datasets:
            assert training_dataset.classes_num == validation_dataset.classes_num

        # Model and optimizer.
        logger.info('Model:')
        self.model = Model(training_dataset.classes_num)
        logger.info(self.model)

        self.training_iterations_num = 500
        lr_scheduler_milestones = [500]
        base_lr = 0.001
        weight_decay = 0.0001
        logger.info('Optimizer:')
        self.optimizer = torch.optim.SGD(self.setup_optimizer(self.model, base_lr, weight_decay),
                                         lr=base_lr, weight_decay=weight_decay, momentum=0.9)
        logger.info(self.optimizer)
        logger.info('Learning Rate scheduler:')
        self.lr_scheduler = MultiStepLRWithWarmUp(
            self.optimizer,
            milestones=lr_scheduler_milestones,
            warmup_iters=100,
            warmup_method='linear',
            warmup_factor_base=0.333,
            gamma=0.1,
            last_epoch=0
        )
        logger.info(self.lr_scheduler)

        self.start_step = 0
        checkpoint_file_path = osp.join(self.root_directory, 'data', 'pretrained_models',
                                        'converted', 'coco', 'ote', 'instance_segmentation_security_0050.pth')
        if not osp.exists(checkpoint_file_path):
            raise IOError('Initial checkpoint file "{}" does not exist. '
                          'Please fetch pretrained networks using '
                          'tools/download_pretrained_weights.py script first.'.format(checkpoint_file_path))
        logger.info('Loading weights from "{}"'.format(checkpoint_file_path))
        load_checkpoint(self.model, checkpoint_file_path)

        # Loggers and misc. stuff.
        self.loggers = [TextLogger(logger),
                        TensorboardLogger(self.run_directory)]
        self.log_every = 50

        self.checkpoint_every = 500


if __name__ == '__main__':
    # RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    experiment = Trainer()
    torch.backends.cudnn.benchmark = False
    experiment.run(experiment.start_step)
    logger.info('Done.')
