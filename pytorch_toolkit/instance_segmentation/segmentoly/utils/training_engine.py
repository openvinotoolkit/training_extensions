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
import math
import os
import traceback
from collections import defaultdict
from datetime import datetime
from os import path as osp

import torch.nn as nn
from tqdm import tqdm

from segmentoly.data.dataparallel import ShallowDataParallel
from segmentoly.data.transforms import *
from segmentoly.utils.env import SuccessExit
from segmentoly.utils.postprocess import postprocess
from segmentoly.utils.profile import Timer
import segmentoly.utils.weights as weight_utils


class TrainingEngine(object):
    def __init__(self):
        self.identifier = None
        self.description = ''

    @staticmethod
    def save_ckpt(output_dir, step, model, optimizer, save_extra_dict=True):
        """Save checkpoint"""
        ckpt_dir = os.path.join(output_dir, 'ckpt')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        save_name = os.path.join(ckpt_dir, 'model_step_{}.pth'.format(step))
        if isinstance(model, nn.DataParallel):
            model = model.module

        # Create and save a dictionary with correctly matched parameter and its momentum buffer
        corrected_matcher = {}
        if save_extra_dict:
            for group in optimizer.param_groups:
                for p in group['params']:
                    param_state = optimizer.state[p]
                    if 'momentum_buffer' in param_state:
                        buffer = param_state['momentum_buffer'].cpu()
                        corrected_matcher[p.cpu()] = buffer

        torch.save(
            {'step': step,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'optimizer_corrected': corrected_matcher
             },
            save_name
        )
        logging.info('save model to "{}"'.format(save_name))

    @staticmethod
    def load_checkpoint(net, optimizer, load_ckpt=None, load_backbone=None, resume=False):
        start_step = 0
        if load_ckpt:
            logging.info('loading checkpoint "{}"'.format(load_ckpt))
            checkpoint = torch.load(load_ckpt, map_location=lambda storage, loc: storage)
            weight_utils.load_rcnn_ckpt(net, checkpoint['model'])
            if resume:
                start_step = checkpoint['step']
                optimizer.load_state_dict(checkpoint['optimizer'])

                corrected_matcher = checkpoint.get('optimizer_corrected', {})
                if len(corrected_matcher) > 0:
                    for group in optimizer.param_groups:
                        for param_original in group['params']:
                            if 'momentum_buffer' in optimizer.state[param_original]:
                                for param_loaded, buffer in corrected_matcher.items():
                                    shapes_are_equal = np.array_equal(list(param_original.shape),
                                                                      list(param_loaded.shape))
                                    if shapes_are_equal and torch.all(
                                        torch.eq(param_original.data, param_loaded.data.cuda())):
                                        optimizer.state[param_original]['momentum_buffer'] = buffer.cuda()
                                        break
                else:
                    # If a checkpoint does not have additional dictionary with matched
                    # parameters and its buffers, just match them by shapes
                    if len(optimizer.state) > 0:  # It means that a checkpoint has momentum_buffer
                        used_buffers = {}
                        copy_buf = None
                        for p in optimizer.state.keys():
                            used_buffers[p] = False
                        for group in optimizer.param_groups:
                            for param in group['params']:
                                for p, buffer in optimizer.state.items():
                                    if 'momentum_buffer' not in buffer:
                                        continue
                                    if np.array_equal(list(param.shape), list(buffer['momentum_buffer'].shape)) and not \
                                       used_buffers[p]:
                                        copy_buf = optimizer.state[param]['momentum_buffer'].cuda()
                                        optimizer.state[param]['momentum_buffer'] = buffer['momentum_buffer'].cuda()
                                        optimizer.state[p]['momentum_buffer'] = copy_buf.cuda()
                                        used_buffers[param] = True
                        del used_buffers
                        del copy_buf
                logging.info('Resume training from {} step'.format(start_step))

            del checkpoint
            torch.cuda.empty_cache()

        if load_backbone:
            logging.info('loading backbone weights from "{}"'.format(load_backbone))
            assert hasattr(net, 'backbone')
            weight_utils.load_checkpoint(net.backbone, load_backbone)

        return start_step

    @staticmethod
    def adjust_virtual_iteration_size(available_gpus_num, total_batch_size, per_gpu_batch_size):
        """Adjust training parameters taking into account available hardware resources."""

        virtual_iter_size = int(math.ceil(total_batch_size / per_gpu_batch_size / available_gpus_num))
        effective_batch_size = available_gpus_num * per_gpu_batch_size * virtual_iter_size
        logging.info('Virtual iter size set to {}'.format(virtual_iter_size))
        logging.info('Effective batch size {}'.format(effective_batch_size))
        return virtual_iter_size

    @staticmethod
    def create_run_directory(root_directory):
        # Determine output directory.
        run_name = datetime.now().strftime('%b%d-%H-%M-%S')
        run_directory = osp.join(root_directory, run_name)
        # Create output directory if necessary.
        if not os.path.exists(run_directory):
            os.makedirs(run_directory)
        return run_directory

    @staticmethod
    def set_random_seeds():
        # torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0xACE)
        torch.cuda.manual_seed_all(0xACE)
        np.random.seed(0xACE)

    def run(self, *args, **kwargs):
        raise NotImplementedError


class DefaultMaskRCNNTrainingEngine(TrainingEngine):
    def __init__(self):
        super().__init__()
        self.set_random_seeds()

        self.identifier = 'Default MaskRCNN training experiment template'
        self.description = 'This is just a template for actual training of MaskRCNN like networks.'
        self.root_directory = './'
        self.run_directory = './'

        self.batch_size = 1
        self.virtual_iter_size = 16

        self.training_data_loader = None
        self.validation_data_loaders = []
        self.validate_every = 5000

        self.model = None

        self.training_iterations_num, lr_scheduler_milestones = self.training_schedule(1)
        self.optimizer = None
        self.lr_scheduler = None

        self.loggers = []
        self.log_every = 20

        self.checkpoint_every = 5000

    @staticmethod
    def training_schedule(multiplier):
        training_iterations_num = int(90000 * multiplier)
        milestones = [int(i * multiplier) for i in (60000, 80000)]
        return training_iterations_num, milestones

    def setup_optimizer(self, net, base_lr, base_weight_decay):
        gn_param_nameset = set()
        for name, module in net.named_modules():
            if isinstance(module, nn.GroupNorm):
                gn_param_nameset.add(name + '.weight')
                gn_param_nameset.add(name + '.bias')
        gn_params = []
        gn_param_names = []
        bias_params = []
        bias_param_names = []
        nonbias_params = []
        nonbias_param_names = []
        nograd_param_names = []
        for key, value in dict(net.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    bias_params.append(value)
                    bias_param_names.append(key)
                elif key in gn_param_nameset:
                    gn_params.append(value)
                    gn_param_names.append(key)
                else:
                    nonbias_params.append(value)
                    nonbias_param_names.append(key)
            else:
                nograd_param_names.append(key)
        assert (gn_param_nameset - set(nograd_param_names) - set(bias_param_names)) == set(gn_param_names)

        # Learning rate of 0 is a dummy value to be set properly at the start of training
        params = [
            {'params': nonbias_params,
             'lr': 0,
             'initial_lr': base_lr,
             'weight_decay': base_weight_decay},
            {'params': bias_params,
             'lr': 0,
             'initial_lr': base_lr,
             'weight_decay': 0},
            {'params': gn_params,
             'lr': 0,
             'initial_lr': base_lr,
             'weight_decay': 0}
        ]
        return params

    def validate(self, net, data_loader, idx=0):
        net.eval()
        logging.info('Processing the dataset...')
        boxes_all = []
        masks_all = []
        classes_all = []
        scores_all = []
        for data_batch in tqdm(iter(data_loader)):
            im_data = data_batch['im_data']
            im_info = data_batch['im_info']
            with torch.no_grad():
                boxes, classes, scores, _, masks = net(im_data, im_info)
            meta = data_batch['meta'][0]
            scores, classes, boxes, masks = postprocess(scores, classes, boxes, masks,
                                                        im_h=meta['original_size'][0],
                                                        im_w=meta['original_size'][1],
                                                        im_scale=meta['processed_size'][0] /
                                                                 meta['original_size'][0],
                                                        full_image_masks=True, encode_masks=True)
            boxes_all.append(boxes)
            masks_all.append(masks)
            classes_all.append(classes)
            scores_all.append(scores)

        logging.info('Evaluating results...')
        evaluation_results = data_loader.dataset.evaluate(scores_all, classes_all, boxes_all, masks_all)
        evaluation_results = {'val{}/{}'.format(idx, k): v for k, v in evaluation_results.items()}
        return evaluation_results

    @staticmethod
    def update_metrics(aggregated_outputs, flattened_outputs):
        for k, v in flattened_outputs.items():
            aggregated_outputs[k].append(v)
        return aggregated_outputs

    def check_experimental_setup(self):
        assert self.model is not None, 'Please specify the model to train.'
        assert self.optimizer is not None, 'Please specify optimizer.'
        assert self.lr_scheduler is not None, 'Please specify learning rate scheduler.'
        assert self.training_data_loader is not None, 'Please specify data loader for training data.'

    def run(self, start_step=0, *args, **kwargs):
        self.check_experimental_setup()

        available_gpus_num = torch.cuda.device_count() if torch.cuda.is_available() else 1
        logging.info('Using {} GPUs for training'.format(available_gpus_num))
        # virtual_iter_size = self.adjust_virtual_iteration_size(available_gpus_num, self.batch_size_total,
        #                                                        self.batch_size_per_gpu)
        virtual_iter_size = self.virtual_iter_size

        training_data_iterator = iter(self.training_data_loader)

        if torch.cuda.is_available():
            self.model.cuda()

        model = ShallowDataParallel(self.model)

        self.save_ckpt(self.run_directory, start_step, model, self.optimizer, False)

        torch.cuda.empty_cache()

        timers = defaultdict(Timer)
        logging.info('Start training...')
        step = start_step
        try:
            for step in range(start_step, self.training_iterations_num):
                with timers['total']:
                    if not self.model.training:
                        model.train()
                    self.lr_scheduler.step()
                    lr = self.optimizer.param_groups[0]['lr']
                    self.optimizer.zero_grad()

                    metrics = defaultdict(list)
                    minibatch_losses = []

                    for virtual_iter in range(virtual_iter_size):
                        try:
                            data_batch = next(training_data_iterator)
                        except StopIteration:
                            training_data_iterator = iter(self.training_data_loader)
                            data_batch = next(training_data_iterator)

                        with timers['forward']:
                            minibatch_metrics, loss = model(**data_batch)

                        # Average losses from different GPUs.
                        total_loss = torch.mean(loss)
                        with timers['backward']:
                            (total_loss / virtual_iter_size).backward()
                        minibatch_losses.append(total_loss.detach_())

                        self.update_metrics(metrics, minibatch_metrics)

                    if step % self.log_every == 0:
                        for k, v in metrics.items():
                            # Average over all effective batch elements.
                            metrics[k] = torch.mean(torch.cat(v))
                        for training_logger in self.loggers:
                            training_logger(step, self.training_iterations_num,
                                            lr, sum(minibatch_losses) / len(minibatch_losses), metrics, timers)

                    with timers['optimization_step']:
                        self.optimizer.step()

                    if (step + 1) % self.checkpoint_every == 0:
                        logging.info('Saving snapshot...')
                        self.save_ckpt(self.run_directory, step + 1, model, self.optimizer, False)

                    if (step + 1) % self.validate_every == 0:
                        logging.info('Running validation...')
                        with timers['validation']:
                            for validation_data_loader in self.validation_data_loaders:
                                validation_results = self.validate(model, validation_data_loader)
                                for training_logger in self.loggers:
                                    training_logger(step + 1, self.training_iterations_num,
                                                    None, None,
                                                    validation_results, None)

            # Save last checkpoint.
            self.save_ckpt(self.run_directory, step + 1, model, self.optimizer, False)
            raise SuccessExit('Program has finished successfully')

        except (RuntimeError, KeyboardInterrupt, SuccessExit) as err:
            if not isinstance(err, SuccessExit):
                logging.info('Save checkpoint on exception ...')
                self.save_ckpt(self.run_directory, step + 1, model, self.optimizer, False)
                stack_trace_msg = '\n' + str(traceback.format_exc())
                logging.warning(stack_trace_msg)

        finally:
            try:
                del self.training_data_loader
            except ConnectionResetError:
                pass

            try:
                for i, _ in enumerate(self.validation_data_loaders):
                    del self.validation_data_loaders[i]
            except ConnectionResetError:
                pass

            for training_logger in self.loggers:
                training_logger.close()
