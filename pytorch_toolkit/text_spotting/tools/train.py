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

import warnings

warnings.simplefilter("ignore", UserWarning)
import argparse

import logging
import os.path as osp
import resource
from collections import OrderedDict
import json
import sys
from tqdm import tqdm

from segmentoly.data.dataparallel import collate
from segmentoly.data.transforms import *

from segmentoly.utils.logging import setup_logging, TextLogger, TensorboardLogger
from segmentoly.utils.lr_scheduler import MultiStepLRWithWarmUp
from segmentoly.utils.training_engine import DefaultMaskRCNNTrainingEngine
from segmentoly.utils.weights import load_checkpoint

from text_spotting.models.text_detectors import make_text_detector
from text_spotting.data.transforms import *
from text_spotting.datasets.factory import get_dataset
from text_spotting.data.alphabet import AlphabetDecoder
from text_spotting.utils.postprocess import postprocess

logger = logging.getLogger(__name__)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('config')
    args.add_argument('--work_dir')
    return args.parse_args()


class Trainer(DefaultMaskRCNNTrainingEngine):

    def __init__(self, work_dir, config):
        super().__init__()
        self.identifier = config['identifier']
        self.description = config['description']
        self.root_directory = work_dir if work_dir else osp.join(osp.dirname(osp.abspath(__file__)),
                                                                 '..')
        self.run_directory = self.create_run_directory(osp.join(self.root_directory, 'models'))

        setup_logging(file_path=osp.join(self.run_directory, 'log.txt'))

        logger.info('Running {}'.format(self.identifier))
        logger.info(self.description)
        logger.info('Working directory "{}"'.format(self.run_directory))

        self.batch_size = config['training_details']['batch_size']
        self.virtual_iter_size = config['training_details']['virtual_iter_size']

        model_class = make_text_detector(**config['model'])

        alphabet_decoder = AlphabetDecoder()

        # Training dataset.
        training_transforms = Compose([
                                          getattr(sys.modules[__name__], k)(**v) for k, v in
                                          config['training_transforms'].items()
                                      ] + [AlphabetDecodeTransform(alphabet_decoder)])

        training_dataset_name = config['training_dataset_name']
        logger.info('Training dataset {}'.format(training_dataset_name))
        training_dataset = get_dataset(training_dataset_name, True, True, training_transforms,
                                       alphabet_decoder=alphabet_decoder,
                                       remove_images_without_text=True)
        logger.info(training_dataset)
        self.training_data_loader = torch.utils.data.DataLoader(
            training_dataset, batch_size=self.batch_size, num_workers=0,
            shuffle=True, drop_last=True, collate_fn=collate
        )

        # Validation datasets.
        validation_transforms = Compose([
            getattr(sys.modules[__name__], k)(**v) for k, v in
            config['validation_transforms'].items()
        ])
        self.confidence_threshold = config['validation_confidence_threshold']
        validation_datasets = []
        validation_dataset_name = config['validation_dataset_name']
        logger.info('Validation dataset #{}: {}'.format(len(validation_datasets) + 1,
                                                        validation_dataset_name))
        validation_datasets.append(
            get_dataset(validation_dataset_name, False, False, validation_transforms,
                        alphabet_decoder=alphabet_decoder))
        logger.info(validation_datasets[-1])

        self.validation_data_loaders = []
        for validation_dataset in validation_datasets:
            self.validation_data_loaders.append(torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=1, num_workers=8,
                shuffle=False, drop_last=False, collate_fn=collate)
            )
        self.validate_every = config['training_details']['validate_every']

        for validation_dataset in validation_datasets:
            assert training_dataset.classes_num == validation_dataset.classes_num

        # Model and optimizer.
        logger.info('Model:')

        self.model = model_class(cls_num=training_dataset.classes_num, shape=config['shape'],
                                 num_chars=len(alphabet_decoder.alphabet))

        logger.info(self.model)

        self.training_iterations_num = config['training_details']['training_iterations_num']
        lr_scheduler_milestones = config['training_details']['lr_scheduler_milestones']
        base_lr = config['training_details']['base_lr']
        weight_decay = config['training_details']['weight_decay']
        logger.info('Optimizer:')
        self.optimizer = torch.optim.SGD(self.setup_optimizer(self.model, base_lr, weight_decay),
                                         lr=base_lr, weight_decay=weight_decay, momentum=0.9)
        logger.info(self.optimizer)
        logger.info('Learning Rate scheduler:')
        self.lr_scheduler = MultiStepLRWithWarmUp(
            self.optimizer,
            milestones=lr_scheduler_milestones,
            warmup_iters=1000,
            warmup_method='linear',
            warmup_factor_base=0.333,
            gamma=0.1,
            last_epoch=0
        )
        logger.info(self.lr_scheduler)

        self.start_step = 0
        if 'backbone_checkpoint' in config and config['backbone_checkpoint']:
            checkpoint_file_path = osp.join(self.root_directory, config['backbone_checkpoint'])
            if not osp.exists(checkpoint_file_path):
                raise IOError('Initial checkpoint file "{}" does not exist. '
                              'Please fetch pre-trained backbone networks using '
                              'tools/download_pretrained_weights.py script first.'.format(
                    checkpoint_file_path))
            logger.info('Loading weights from "{}"'.format(checkpoint_file_path))
            load_checkpoint(self.model.backbone, checkpoint_file_path, verbose=True,
                            skip_prefix='text_recogn')

        if 'checkpoint' in config and config['checkpoint']:
            checkpoint_file_path = osp.join(self.root_directory, config['checkpoint'])
            if not osp.exists(checkpoint_file_path):
                raise IOError('Checkpoint file "{}" does not exist. '.format(checkpoint_file_path))
            logger.info('Loading weights from "{}"'.format(checkpoint_file_path))
            load_checkpoint(self.model, checkpoint_file_path, verbose=True)

        # Loggers and misc. stuff.
        self.loggers = [TextLogger(logger), TensorboardLogger(self.run_directory)]
        self.log_every = 50

        self.checkpoint_every = config['training_details']['checkpoint_every']

    def validate(self, net, data_loader, idx=0):
        net.eval()
        logging.info('Processing the dataset...')
        boxes_all = []
        masks_all = []
        classes_all = []
        scores_all = []
        text_all = []
        for data_batch in tqdm(iter(data_loader)):
            im_data = data_batch['im_data']
            im_info = data_batch['im_info']
            with torch.no_grad():
                boxes, classes, scores, _, masks, text_probs = net(im_data, im_info)
            meta = data_batch['meta'][0]
            scores, classes, boxes, masks, text_probs = postprocess(
                scores, classes, boxes, masks, text_probs,
                im_h=meta['original_size'][0],
                im_w=meta['original_size'][1],
                im_scale_y=meta['processed_size'][0] / meta['original_size'][0],
                im_scale_x=meta['processed_size'][1] / meta['original_size'][1],
                full_image_masks=True,
                encode_masks=True,
                confidence_threshold=self.confidence_threshold)

            boxes_all.append(boxes)
            masks_all.append(masks)
            classes_all.append(classes)
            scores_all.append(scores)
            text_all.append(text_probs)

        logging.info('Evaluating results...')
        evaluation_results = data_loader.dataset.evaluate(scores_all, classes_all, boxes_all,
                                                          masks_all, text_all)
        evaluation_results = {'val{}/{}'.format(idx, k): v for k, v in evaluation_results.items()}
        return evaluation_results


if __name__ == '__main__':
    assert sys.version_info[0] == 3 and sys.version_info[1] > 6

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    args = parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f, object_pairs_hook=OrderedDict)

    experiment = Trainer(args.work_dir, config)
    torch.backends.cudnn.benchmark = False
    experiment.run(experiment.start_step)
    logger.info('Done.')
