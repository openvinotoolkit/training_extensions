# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch
import pytorch_lightning as pl
# dataset
from torchtts.datasets import get_tts_datasets, collate_tts, TrainTTSDatasetSampler
from torch.utils.data import DataLoader
# models
from torchtts.models import GANTacotron, MultiScaleDiscriminator
# loss
from torchtts.loss import duration_loss


class PipelineTTS(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        try:
            self.trainset, self.valset = get_tts_datasets(cfg.data, max_mel_len=1700)
            self.train_sampler = TrainTTSDatasetSampler(self.trainset, self.cfg.trainer.batch_size,
                                                        3 * self.cfg.trainer.batch_size, self.cfg.trainer.distributed)
        except:
            print("Dataset was not loaded in initialization, please initialize it")

        self.generator = GANTacotron(cfg.model)
        self.discriminator = MultiScaleDiscriminator()

    def init_datasets(self, trainset, valset):
        self.trainset = trainset
        self.valset = valset
        self.train_sampler = TrainTTSDatasetSampler(self.trainset, self.cfg.trainer.batch_size,
                                                    3 * self.cfg.trainer.batch_size, self.cfg.trainer.distributed)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, x_len, m, mel_len = batch

        if optimizer_idx == 0:
            m_, mel_mask, (att, log_dur, log_dur_, loss_mel_proj, _) = self.generator(x, x_len, m, mel_len)

            disc_fake = self.discriminator((m_ * mel_mask).unsqueeze(1))
            disc_real = self.discriminator((m * mel_mask).unsqueeze(1))

            loss_g = 0.0
            loss_feat_match = 0.0

            mel_proj_loss_match = max(self.cfg.trainer.mel_projection_loss_cooling_epochs - self.current_epoch, 10)
            feat_match = max(self.cfg.trainer.features_l1_loss_cooling_epochs - self.current_epoch, 10)

            for (feats_fake, score_fake), (feats_real, _) in zip(disc_fake, disc_real):
                loss_g += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
                if feats_fake is None or feats_real is None:
                    continue
                for feat_f, feat_r in zip(feats_fake, feats_real):
                    loss_feat_match += torch.mean(torch.abs(feat_f - feat_r))

            loss_length = duration_loss(log_dur, log_dur_, x_len)

            loss = loss_g + loss_length + mel_proj_loss_match * loss_mel_proj + feat_match * loss_feat_match

            self.log("loss_g", loss_g, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("loss_feat_match", loss_feat_match, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("loss_length", loss_length, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("mel_proj_loss", loss_mel_proj, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            return {"loss": loss.unsqueeze(0)}
        else:
            self.generator.eval()
            with torch.no_grad():
                m_, mel_mask, (_, _, _, _, _) = self.generator(x, x_len, m, mel_len)
            fake_m = m_.detach()

            loss_d = 0.0

            if self.current_epoch < self.cfg.trainer.discriminator_smoothing_epochs:
                alpha = min(1.0, (self.current_epoch + 1) / self.cfg.trainer.discriminator_smoothing_epochs)
                fake_m = alpha * fake_m + (1.0 - alpha) * m
                m = alpha * m + (1 - alpha) * fake_m

            disc_fake = self.discriminator((fake_m * mel_mask).unsqueeze(1))
            disc_real = self.discriminator((m * mel_mask).unsqueeze(1))
            for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                loss_d += torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
                loss_d += torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))

            self.log("loss_d", loss_d, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            loss = loss_d

            return {"loss": loss.unsqueeze(0)}

    def validation_step(self, batch, batch_idx):
        x, x_len, m, mel_len = batch

        m_, mel_mask, (_, log_dur, log_dur_, _, _) = self.generator(x, x_len, m, mel_len)

        diff = 0.0
        for i in range(int(m.shape[0])):
            diff += (m[i, :, :mel_len[i]] - m_[i, :, :mel_len[i]]).abs().mean()

        diff = diff / int(m.shape[0])

        self.log("loss_val", diff, prog_bar=True, logger=True)

        return {"loss_val": diff}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss_val'] for x in outputs]).mean()
        self.log("loss_val", loss, prog_bar=True)

        lr = 0.0

        opt = self.optimizers()[0]
        for param_group in opt.param_groups:
            lr += param_group['lr']

        lr = lr / max(1, len(opt.param_groups))
        self.log("LR", lr, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def predict(self, dataloader: DataLoader):
        res = []
        self.generator.eval()

        for batch in dataloader:
            x, x_len, m, mel_len = batch
            with torch.no_grad():
                m_, mel_mask, (att, log_dur, log_dur_, loss_mel_proj, _) = self.generator(x, x_len, m, mel_len)
                res.append({"gt": m.squeeze().cpu().numpy(), "predict": m_.squeeze().detach().cpu().numpy()})

        self.generator.train()

        return res

    def compute_metrics(self, data):
        return data

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.cfg.trainer.batch_size,
            num_workers=self.cfg.trainer.num_workers,
            collate_fn=collate_tts,
            sampler=self.train_sampler,
            shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=8,
            num_workers=1,
            collate_fn=collate_tts,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=8,
            num_workers=1,
            collate_fn=collate_tts,
            shuffle=False
        )

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.cfg.trainer.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg.trainer.lr)

        sch_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, milestones=[150, 300], gamma=0.5)
        sch_d = torch.optim.lr_scheduler.MultiStepLR(opt_d, milestones=[150, 300], gamma=0.5)

        return [opt_g, opt_d], [sch_g, sch_d]

    def forward(self, *args, **kwargs):
        return self.generator(*args, **kwargs)

    def to_onnx(self, onnx_path, batch_size=1):
        return self.generator.to_onnx(onnx_path, self.cfg)
