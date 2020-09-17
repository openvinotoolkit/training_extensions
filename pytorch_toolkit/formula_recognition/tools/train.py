import argparse
import yaml
import os.path
import sys
from functools import partial
from pprint import pformat

from tqdm import tqdm

import torch
import torch.optim as optim
from im2latex.data.utils import (cal_loss, collate_fn,
                                 create_list_of_transforms, get_timestamp)
from im2latex.data.vocab import read_vocab
from im2latex.datasets.im2latex_dataset import (BatchRandomSampler,
                                                Im2LatexDataset)
from im2latex.models.im2latex_model import Im2latexModel
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(self, work_dir, config):
        self.config = config
        self.model_path = config.get('model_path')
        self.train_paths = config.get('train_paths')
        self.val_path = config.get('val_path')
        self.vocab = read_vocab(config.get('vocab_path'))
        self.train_transforms_list = config.get('train_transforms_list')
        self.val_transforms_list = config.get('val_transforms_list')
        self.total_epochs = config.get('epochs', 30)
        self.learing_rate = config.get('learning_rate', 1e-3)
        self.clip = config.get('clip_grad', 5.0)
        self.work_dir = work_dir
        self.save_dir = os.path.join(self.work_dir, 'save_dir')
        self.val_results_path = os.path.join(self.save_dir, "val_results")
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
        self.logs_path = os.path.join(self.work_dir, 'logs')
        self.writer = SummaryWriter(self.logs_path)
        self.writer.add_text("General info", pformat(config))
        self.create_dirs()
        self.load_dataset()
        self.model = Im2latexModel(config.get('backbone_type'), config.get(
            'backbone_config'), len(self.vocab), config.get('head'))
        if self.model_path is not None:
            self.model.load_weights(self.model_path, old_model=config.get("old_model"))

        self.optimizer = getattr(optim, config.get('optimizer', "Adam"))(self.model.parameters(), self.learing_rate)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer)
        self.device = config.get('device', 'cpu')
        self.model = self.model.to(self.device)
        self.time = get_timestamp()

    def create_dirs(self):
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.val_results_path):
            os.makedirs(self.val_results_path)

    def load_dataset(self):
        train_dataset = ConcatDataset(
            [Im2LatexDataset(train_s, 'train') for train_s in self.train_paths]
        )
        train_sampler = BatchRandomSampler(
            dataset=train_dataset, batch_size=self.config.get('batch_size', 4))
        batch_transform_train = create_list_of_transforms(self.val_transforms_list)
        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=partial(collate_fn, self.vocab.sign2id,
                               batch_transform=batch_transform_train),
            num_workers=os.cpu_count())

        val_dataset = Im2LatexDataset(self.val_path, 'validate')
        val_sampler = BatchRandomSampler(dataset=val_dataset, batch_size=1)
        batch_transform_val = create_list_of_transforms(self.val_transforms_list)
        self.val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=partial(collate_fn, self.vocab.sign2id,
                               batch_transform=batch_transform_val),
            num_workers=os.cpu_count())

    def train(self):

        print("Created logs folder: {}".format(self.logs_path))
        while self.epoch <= self.total_epochs:
            losses = 0.0
            for _, imgs, tgt4training, tgt4cal_loss in self.train_loader:
                step_loss, step_accuracy = self.train_step(imgs, tgt4training, tgt4cal_loss)
                losses += step_loss
                self.writer.add_scalar('Train loss', step_loss, self.global_step)
                self.writer.add_scalar('Train accuracy', step_accuracy, self.global_step)

                # log message
                if self.global_step % self.print_freq == 0:
                    total_step = len(self.train_loader)
                    print("Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}".format(
                        self.epoch, self.step, total_step,
                        100 * self.step / total_step,
                        losses / self.print_freq
                    ))
                    losses = 0.0
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

    def train_step(self, imgs, tgt4training, tgt4cal_loss):
        self.optimizer.zero_grad()
        imgs = imgs.to(self.device)
        tgt4training = tgt4training.to(self.device)
        tgt4cal_loss = tgt4cal_loss.to(self.device)
        logits, _ = self.model(imgs, tgt4training)
        loss, accuracy = cal_loss(logits, tgt4cal_loss, should_cut_by_min=True)
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
            filename = "{}/results_epoch_{}_step{}_{}.txt".format(self.val_results_path,
                                                                  self.epoch, self.step, self.time)
            with open(filename, 'w') as output_file:

                for img_name, imgs, tgt4training, tgt4cal_loss in tqdm(self.val_loader):

                    imgs = imgs.to(self.device)
                    tgt4training = tgt4training.to(self.device)
                    tgt4cal_loss = tgt4cal_loss.to(self.device)
                    logits, pred = self.model(imgs, tgt4training if use_gt_token else None)

                    for j, phrase in enumerate(pred):
                        gold_phrase_str = self.vocab.construct_phrase(
                            tgt4cal_loss[j])
                        pred_phrase_str = self.vocab.construct_phrase(phrase,
                                                                      max_len=1 +
                                                                      len(gold_phrase_str.split(
                                                                      )))
                        output_file.write(img_name[j] + '\t' +
                                          pred_phrase_str + '\t' +
                                          gold_phrase_str + '\n')
                    loss, accuracy = cal_loss(logits, tgt4cal_loss,
                                              should_cut_by_min=True)
                    loss = loss.detach()
                    val_total_loss += loss
                    val_total_accuracy += accuracy
            avg_loss = val_total_loss / len(self.val_loader)
            avg_accuracy = val_total_accuracy / len(self.val_loader)
            print("Epoch {}, validation average loss: {:.4f}".format(
                self.epoch, avg_loss
            ))
            print("Epoch {}, validation average accuracy: {:.4f}".format(
                self.epoch, avg_accuracy
            ))

        self.model.train()
        return avg_loss, avg_accuracy

    def save_model(self, name):
        print("Saving model as name ", name)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_dir, name)
        )


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--config')
    args.add_argument('--work_dir')
    return args.parse_args()


if __name__ == "__main__":
    assert sys.version_info[0] == 3
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    experiment = Trainer(work_dir=args.work_dir, config=config)
    experiment.train()
