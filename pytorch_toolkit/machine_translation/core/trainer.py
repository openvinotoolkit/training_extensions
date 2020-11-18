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
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .tokenizer import build_tokenizer
from .dataset import build_dataset
from .models import build_model
from .bleu import BLEU


class NMTTrainer(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg.trainer
        self.tokenizer = build_tokenizer(cfg.tokenizer)
        # dataset
        self.trainset = build_dataset(cfg.trainset)
        self.valset = build_dataset(cfg.valset)
        # model
        cfg.model.params.src_vocab_size = self.tokenizer.vocab_size("src")
        cfg.model.params.tgt_vocab_size = self.tokenizer.vocab_size("tgt")
        cfg.model.params.src_pad_idx = self.tokenizer.pad_idx("src")
        cfg.model.params.tgt_pad_idx = self.tokenizer.pad_idx("tgt")
        self.model = build_model(cfg.model)
        # metric
        self.metric = BLEU(
            num_refs=1,
            distributed=cfg.distributed_backend == 'ddp'
        )
        # device info
        self.register_buffer("device_info", torch.zeros(1))

    def log_dict(self, log, exclude_list=["loss"], on_step=True, prog_bar=True, logger=True, on_epoch=False):
        for k, v in log.items():
            if not k in exclude_list:
                v = torch.tensor([v.item()]).to(self.device_info.device)
                self.log(k, v, on_step=on_step, prog_bar=prog_bar, logger=logger, on_epoch=on_epoch)

    def training_step(self, batch, batch_idx):
        loss = self.model.loss(**self.model(**batch), **batch)
        loss["loss_train"] = torch.tensor([loss["loss"].item()]).to(self.device_info.device)
        self.log_dict(loss)
        return {"loss": loss["loss"].unsqueeze(0)}

    def validation_step(self, batch, batch_idx):
        loss = self.model.loss(**self.model(**batch), **batch)
        # evaluate BLEU
        out = self.model(src=batch["src"])
        preds = out["token_predictor_logits"].softmax(-1).argmax(-1)
        preds = self.tokenizer.decode_batch(preds, 'tgt', postprocess=True)
        gt = [self.tokenizer.decode_batch(batch["tgt"], 'tgt', postprocess=True)]
        self.metric.add(preds=preds, gt=gt)
        loss["loss_val"] = torch.tensor([loss["loss"].item()]).to(self.device_info.device)
        self.log_dict(loss)
        return {}

    def validation_epoch_end(self, outputs):
        bleu = torch.tensor(self.metric.value()["bleu"])
        self.metric.reset()
        self.log('bleu', bleu, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        # optimizer
        groups =  self.model.get_params()
        for i in range(len(groups)):
            groups[i]['lr'] *= self.cfg.lr
        opt = torch.optim.Adam(groups, betas=(0.9, 0.98), eps=1e-4)
        # lr scheduler
        milestones = [int(x) for x in self.cfg.milestones.split(',')]
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            collate_fn=self.tokenizer
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.tokenizer
        )

    def test_dataloader(self):
        return self.val_dataloader()


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def cli_mode(self):
        self.model.eval()
        while True:
            text = input("> ")
            if not text:
                break
            src = self.tokenizer.encode_batch([text], mode="src")
            preds = self.model(src=src)["token_predictor_logits"].argmax(-1)
            preds = self.tokenizer.decode_batch(preds, 'tgt', remove_extra=True, postprocess=False)
            print(preds)

    def to_onnx(self, onnx_path, denominator):
        class ONNXWrapper(nn.Module):
            def __init__(self, m, max_len):
                super().__init__()
                self.m = m
                self.max_len = max_len

            def forward(self, src):
                out = self.m(src=src, max_len=self.max_len)
                return out["token_predictor_logits"].argmax(-1)

        length_in = self.tokenizer.src.max_length
        src = self.tokenizer.encode_batch([" " * (length_in - 2)], mode="src").to(self.device_info.device)
        length_out = length_in + self.model.max_delta
        length_out = int(((length_out + denominator - 1) // denominator) * denominator)
        torch.onnx.export(
            ONNXWrapper(self.model, length_out),
            (src,),
            onnx_path,
            input_names=["tokens"],
            output_names=["preds"],
            verbose=True
        )
