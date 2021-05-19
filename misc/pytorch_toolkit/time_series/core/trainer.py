import os
# pytorch
import torch
import pytorch_lightning as pl
# dataset
from core.dataset import get_dataset
from torch.utils.data import DataLoader
# model
from core.models.temporal_fusion_transformer import TemporalFusionTransformer
# loss
from core.loss import QuantileLoss, NormalizedQuantileLoss


class TimeSeriesTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.pipeline
        self.trainset, self.valset, self.testset = get_dataset(cfg.dataset.name).get_split(**cfg.dataset.params)
        self.model = TemporalFusionTransformer(cfg.model)
        self.train_criterion = QuantileLoss(cfg.model.quantiles)
        self.test_criterion = NormalizedQuantileLoss()

    def training_step(self, batch, batch_idx):
        inputs, outputs, mean, scale = batch
        inputs, outputs = inputs.to(float), outputs.to(float)
        preds = self.model(inputs)
        loss = self.train_criterion(preds, torch.stack([outputs, outputs, outputs], dim=-1))
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss.unsqueeze(0)}

    def validation_step(self, batch, batch_idx):
        inputs, outputs, mean, scale = batch
        inputs, outputs = inputs.to(float), outputs.to(float)
        preds = self.model(inputs)
        loss = self.train_criterion(preds, torch.stack([outputs, outputs, outputs], dim=-1))
        self.log("loss_val", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss_val": loss.unsqueeze(0)}

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, "test")

    def eval_step(self, batch, batch_idx):
        inputs, outputs, mean, scale = batch
        inputs, outputs = inputs.to(float), outputs.to(float)
        preds = self.model(inputs)
        preds = preds * scale.view(-1, 1, 1) + mean.view(-1, 1, 1)
        outputs = outputs * scale + mean
        return {"outputs": outputs, "preds": preds}

    def eval_epoch_end(self, outputs, mode):
        preds, gt = [], []
        for out in outputs:
            preds.append(out["preds"])
            gt.append(out["outputs"])
        preds = torch.cat(preds, dim=0)
        gt = torch.cat(gt, dim=0)
        for i, q in enumerate(self.model.cfg.quantiles):
            loss = self.test_criterion(preds[:, :, i], gt, q).unsqueeze(0)
            self.log(f"loss_{mode}_{q}", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False
        )

    def configure_optimizers(self):
        # optimizer
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        # lr scheduler
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, self.cfg.milestones)
        return [opt], [sch]

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to_onnx(self, onnx_path, batch_size=1):
        inputs, _, _, _ = next(iter(self.train_dataloader()))
        inputs = inputs[0].unsqueeze(0).repeat(batch_size, 1, 1)
        torch.onnx.export(
            self.model,
            (inputs.float(),),
            onnx_path,
            input_names=["timestamps"],
            output_names=["quantiles"],
            verbose=True
        )
