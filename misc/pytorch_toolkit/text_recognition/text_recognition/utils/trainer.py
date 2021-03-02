"""
MIT License

Copyright (c) 2019 luopeixiang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

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

import os
from functools import partial
from pprint import pformat, pprint
from warnings import warn

import numpy as np
import torch
import torch.multiprocessing
import torch.nn
import torch.optim as optim
from text_recognition.data.utils import (collate_fn, create_list_of_transforms,
                                 ctc_greedy_search, get_timestamp)
from text_recognition.data.vocab import END_TOKEN, PAD_TOKEN, read_vocab
from text_recognition.datasets.dataset import BatchRandomSampler, str_to_class
from text_recognition.models.model import TextRecognitionModel
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def calculate_loss(logits, targets, target_lengths, should_cut_by_min=False, ctc_loss=None):
    """args:
        logits: probability distribution return by model
                [B, MAX_LEN, voc_size]
        targets: target formulas
                [B, MAX_LEN]
    """

    if ctc_loss is None:
        if should_cut_by_min:
            required_len = min(logits.size(1), targets.size(1))
            logits = logits.narrow(1, 0, required_len)
            targets = targets.narrow(1, 0, required_len)
            if required_len < targets.size(1):
                warn("Cutting tensor leads to tensor sized less than target")
        else:
            # narrows on 1st dim from 'start_pos' 'length' symbols
            logits = logits.narrow(1, 0, targets.size(1))

        padding = torch.ones_like(targets) * PAD_TOKEN
        mask_for_tgt = (targets != padding)
        b_size, max_len, vocab_size = logits.shape  # batch size, length of the formula, vocab size
        targets = targets.masked_select(mask_for_tgt)
        mask_for_logits = mask_for_tgt.unsqueeze(2).expand(-1, -1, vocab_size)
        logits = logits.masked_select(mask_for_logits).contiguous().view(-1, vocab_size)
        logits = torch.log(logits)
        assert logits.size(0) == targets.size(0)
        pred = torch.max(logits.data, 1)[1]

        accuracy = (pred == targets)
        accuracy = accuracy.cpu().numpy().astype(np.uint32)
        accuracy = np.sum(accuracy) / len(accuracy)
        accuracy = accuracy.item()
        loss = torch.nn.functional.nll_loss(logits, targets)
    else:
        logits = torch.nn.functional.log_softmax(logits, dim=2)
        max_len, b_size, vocab_size = logits.shape  # batch size, length of the formula, vocab size
        input_lengths = torch.full(size=(b_size,), fill_value=max_len, dtype=torch.long)
        loss = ctc_loss(logits, targets, input_lengths=input_lengths, target_lengths=target_lengths)

        predictions = ctc_greedy_search(logits.detach(), ctc_loss.blank)
        accuracy = 0
        for i in range(b_size):
            gt = torch.IntTensor([c for c in targets[i] if c not in (PAD_TOKEN, END_TOKEN)])
            if len(predictions[i]) == len(gt) and torch.all(predictions[i].eq(gt)):
                accuracy += 1
        accuracy /= b_size
    return loss, accuracy


class Trainer:
    def __init__(self, work_dir, config):
        self.config = config
        self.model_path = config.get('model_path')
        self.train_paths = config.get('train_paths')
        self.val_path = config.get('val_path')
        self.vocab = read_vocab(config.get('vocab_path'))
        self.train_transforms_list = config.get('train_transforms_list')
        self.val_transforms_list = config.get('val_transforms_list')
        self.loss_type = config.get("loss_type", 'NLL')
        self.total_epochs = config.get('epochs', 30)
        self.learing_rate = config.get('learning_rate', 1e-3)
        self.clip = config.get('clip_grad', 5.0)
        self.work_dir = os.path.abspath(work_dir)
        self.save_dir = os.path.join(self.work_dir, config.get("save_dir", "model_checkpoints"))
        self.val_results_path = os.path.join(self.work_dir, "val_results")
        self.step = 0
        self.global_step = 0
        self._test_steps = config.get('_test_steps', 1e18)
        self.epoch = 1
        self.best_val_loss = 1e18
        self.best_val_accuracy = 0.0
        self.best_val_loss_test = 1e18
        self.best_val_accuracy_test = 0.0
        self.print_freq = config.get('print_freq', 16)
        self.save_freq = config.get('save_freq', 2000)
        self.val_freq = config.get('val_freq', 5000)
        self.logs_path = os.path.join(self.work_dir, config.get("log_path", "logs"))
        self.writer = SummaryWriter(self.logs_path)
        self.device = config.get('device', 'cpu')
        self.writer.add_text("General info", pformat(config))
        self.create_dirs()
        self.load_dataset()
        self.loss = torch.nn.CTCLoss(blank=0, zero_infinity=self.config.get(
            "CTCLossZeroInf", False)) if self.loss_type == "CTC" else None
        self.out_size = len(self.vocab) + 1 if self.loss_type == 'CTC' else len(self.vocab)
        self.model = TextRecognitionModel(config.get('backbone_config'), self.out_size, config.get('head', {}))
        if self.model_path is not None:
            self.model.load_weights(self.model_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.optimizer = getattr(optim, config.get('optimizer', "Adam"))(self.model.parameters(), self.learing_rate)
        self.lr_scheduler = getattr(optim.lr_scheduler, self.config.get("scheduler", "ReduceLROnPlateau"))(
            self.optimizer, **self.config.get("scheduler_params", {}))
        self.time = get_timestamp()

    def create_dirs(self):
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        print("Created logs folder: {}".format(self.logs_path))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print("Created save folder: {}".format(self.save_dir))
        if not os.path.exists(self.val_results_path):
            os.makedirs(self.val_results_path)
        print("Created validation results folder: {}".format(self.val_results_path))

    def load_dataset(self):
        train_datasets = []
        val_datasets = []
        for param in self.config.get("datasets"):
            dataset_type = param.pop("type")
            subset = param.pop('subset')
            dataset = str_to_class[dataset_type](**param)
            if subset == 'train':
                train_datasets.append(dataset)
            elif subset == 'validate':
                val_datasets.append(dataset)

        train_dataset = ConcatDataset(train_datasets)
        train_sampler = BatchRandomSampler(
            dataset=train_dataset, batch_size=self.config.get('batch_size', 4))
        pprint("Creating training transforms list: {}".format(self.train_transforms_list), indent=4, width=120)
        batch_transform_train = create_list_of_transforms(self.train_transforms_list)
        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=partial(collate_fn, self.vocab.sign2id,
                               batch_transform=batch_transform_train,
                               use_ctc=(self.loss_type == "CTC")),
            num_workers=self.config.get("num_workers", 4))
        val_dataset = ConcatDataset(val_datasets)
        val_sampler = BatchRandomSampler(dataset=val_dataset, batch_size=1)
        pprint("Creating val transforms list: {}".format(self.val_transforms_list), indent=4, width=120)
        batch_transform_val = create_list_of_transforms(self.val_transforms_list)
        self.val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=partial(collate_fn, self.vocab.sign2id,
                               batch_transform=batch_transform_val, use_ctc=(self.loss_type == "CTC")),
            num_workers=self.config.get("num_workers", 4))
        print("num workers: ", self.config.get("num_workers"))

    def train(self):
        losses = 0.0
        accuracies = 0.0
        while self.epoch <= self.total_epochs:
            for _, target_lengths, imgs, training_gt, loss_computation_gt in self.train_loader:
                step_loss, step_accuracy = self.train_step(imgs, target_lengths, training_gt, loss_computation_gt)
                losses += step_loss
                accuracies += step_accuracy
                self.writer.add_scalar('Train loss', step_loss, self.global_step)
                self.writer.add_scalar('Train accuracy', step_accuracy, self.global_step)

                # log message
                if self.global_step % self.print_freq == 0:
                    total_step = len(self.train_loader)
                    print("Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}, accuracy: {:.4f}".format(
                        self.epoch, self.step, total_step,
                        100 * self.step / total_step,
                        losses / self.print_freq, accuracies / self.print_freq
                    ))
                    losses = 0.0
                    accuracies = 0.0
                if self.global_step % self.save_freq == 0:
                    self.save_model("model_epoch_{}_step_{}_{}.pth".format(
                        self.epoch,
                        self.step,
                        self.time,
                    ))
                    self.writer.add_scalar('Learning rate', self.learing_rate, self.global_step)
                if self.global_step % self.val_freq == 0:

                    step_loss, step_accuracy = self.validate()
                    self.writer.add_scalar('Loss/validation', step_loss, self.global_step)
                    self.writer.add_scalar('Accuracy/validation', step_accuracy, self.global_step)
                    if step_loss < self.best_val_loss:
                        self.best_val_loss = step_loss
                        self.save_model("loss_best_model_{}.pth".format(self.time))
                    if step_accuracy > self.best_val_accuracy:
                        self.best_val_accuracy = step_accuracy
                        self.save_model("accuracy_best_model_{}.pth".format(self.time))

                    step_loss, step_accuracy = self.validate(use_gt_token=False)
                    self.writer.add_scalar('Loss/test_mode_validation', step_loss, self.global_step)
                    self.writer.add_scalar('Accuracy/test_mode_validation', step_accuracy, self.global_step)
                    if step_loss < self.best_val_loss_test:
                        self.best_val_loss_test = step_loss
                        self.save_model("loss_test_best_model_{}.pth".format(self.time))
                    if step_accuracy > self.best_val_accuracy_test:
                        self.best_val_accuracy_test = step_accuracy
                        self.save_model("accuracy_test_best_model_{}.pth".format(self.time))

                    self.lr_scheduler.step(step_loss)
                self._current_loss = losses
                if self.global_step >= self._test_steps:
                    return

            self.epoch += 1
            self.step = 0

    def train_step(self, imgs, target_lengths, training_gt, loss_computation_gt):
        self.optimizer.zero_grad()
        imgs = imgs.to(self.device)
        training_gt = training_gt.to(self.device)
        loss_computation_gt = loss_computation_gt.to(self.device)
        logits, _ = self.model(imgs, training_gt)
        cut = False if self.loss_type == 'CTC' else True
        loss, accuracy = calculate_loss(logits, loss_computation_gt, target_lengths, should_cut_by_min=cut,
                                        ctc_loss=self.loss)
        self.step += 1
        self.global_step += 1
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss.item(), accuracy

    def validate(self, use_gt_token=True):
        self.model.eval()
        val_total_loss = 0.0
        val_total_accuracy = 0.0
        print("Validation started")
        with torch.no_grad():
            filename = "{}/results_epoch_{}_step_{}_{}.txt".format(self.val_results_path,
                                                                   self.epoch, self.step, self.time)
            with open(filename, 'w') as output_file:

                for img_name, target_lengths, imgs, training_gt, loss_computation_gt in tqdm(self.val_loader):

                    imgs = imgs.to(self.device)
                    training_gt = training_gt.to(self.device)
                    loss_computation_gt = loss_computation_gt.to(self.device)
                    logits, pred = self.model(imgs, training_gt if use_gt_token else None)
                    if self.loss_type == 'CTC':
                        pred = torch.nn.functional.log_softmax(logits.detach(), dim=2)
                        pred = ctc_greedy_search(pred, blank_token=self.loss.blank)
                    for j, phrase in enumerate(pred):
                        gold_phrase_str = self.vocab.construct_phrase(
                            loss_computation_gt[j], ignore_end_token=self.config.get("use_ctc"))
                        pred_phrase_str = self.vocab.construct_phrase(phrase,
                                                                      max_len=1 + len(gold_phrase_str.split()),
                                                                      ignore_end_token=self.config.get("use_ctc")
                                                                      )
                        output_file.write(img_name[j] + '\t' +
                                          pred_phrase_str + '\t' +
                                          gold_phrase_str + '\n')
                        val_total_accuracy += int(pred_phrase_str == gold_phrase_str)
                    cut = False if self.loss_type == 'CTC' else True
                    loss, _ = calculate_loss(logits, loss_computation_gt, target_lengths,
                                             should_cut_by_min=cut, ctc_loss=self.loss)
                    loss = loss.detach()
                    val_total_loss += loss
            avg_loss = val_total_loss / len(self.val_loader)
            avg_accuracy = val_total_accuracy / len(self.val_loader)
            print("Epoch {}, validation average loss: {:.4f}".format(
                self.epoch, avg_loss
            ))
            print("Epoch {}, validation average accuracy: {:.4f}".format(
                self.epoch, avg_accuracy
            ))
        self.save_model("validation_epoch_{}_step_{}_{}.pth".format(self.epoch, self.step, self.time))
        self.model.train()
        return avg_loss, avg_accuracy

    def save_model(self, name):
        print("Saving model as name ", name)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_dir, name)
        )
