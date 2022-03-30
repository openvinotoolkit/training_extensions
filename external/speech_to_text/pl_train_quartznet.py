#!/usr/bin/env python3
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import os
import sys
import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from addict import Dict
from speech_to_text.datasets import AudioDataset
from speech_to_text.trainers import LightningQuartzNetTrainer
import speech_to_text.utils as utils


def main(args):
    cfg = utils.load_cfg(args.cfg)
    # prepare tokenizer & dataset
    tokenizer = utils.build_tokenizer(
        data_path = cfg.trainset.data_path,
        model_path = cfg.tokenizer.model_path,
        vocab_size = cfg.tokenizer.vocab_size
    )
    trainset = utils.build_dataset(**cfg.trainset)
    valset = utils.build_dataset(**cfg.valset)
    cfg.optimizer.epoch_size = len(trainset) * cfg.trainset.batch_size
    # build pipeline
    pipeline = LightningQuartzNetTrainer(cfg, tokenizer=tokenizer)
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        utils.load_weights(pipeline, ckpt["state_dict"])
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.getcwd() if args.checkpoint_dir is None else args.checkpoint_dir,
        filename="{epoch}",
        save_top_k=True,
        save_last=True,
        verbose=True,
        monitor=cfg.pipeline.monitor,
        mode=cfg.pipeline.monitor_mode,
    )
    if args.gpus > 0:
        engine = {"devices": args.gpus, "accelerator": "gpu", "strategy": "ddp"}
    else:
        engine = {"accelerator": "cpu"}
    trainer = pl.Trainer(
        max_epochs=cfg.optimizer.epochs,
        accumulate_grad_batches=args.grad_batches,
        val_check_interval=args.val_check_interval if args.val_check_interval > 0 else 1.0,
        gradient_clip_val=15,
        gradient_clip_algorithm="value",
        callbacks=[checkpoint_callback],
        **engine
    )
    if args.val:
        pipeline.metrics.reset()
        output = trainer.test(
            pipeline.eval(),
            utils.build_dataloader(
                valset,
                tokenizer = tokenizer,
                audio_transforms_cfg = cfg.audio_transforms.val,
                batch_size = cfg.valset.batch_size,
                num_workers = cfg.valset.num_workers,
                shuffle = False
            )
        )
        print(output)
    elif args.export:
        pipeline.export(args.export_dir, to_openvino=True)
    else:
        trainer.fit(
            pipeline,
            utils.build_dataloader(
                trainset,
                tokenizer = tokenizer,
                audio_transforms_cfg = cfg.audio_transforms.train,
                batch_size = cfg.trainset.batch_size,
                num_workers = cfg.trainset.num_workers,
                shuffle = True
            ),
            utils.build_dataloader(
                valset,
                tokenizer = tokenizer,
                audio_transforms_cfg = cfg.audio_transforms.val,
                batch_size = cfg.valset.batch_size,
                num_workers = cfg.valset.num_workers,
                shuffle = False
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--gpus", type=int, default=0, help="Number of available GPUs")
    parser.add_argument('--distributed-backend', type=str, default="ddp", choices=('dp', 'ddp', 'ddp2'),
                        help='Supports three options dp, ddp, ddp2')
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to checkpoint_dir")
    parser.add_argument("--val-check-interval", type=int, default=0, help="Validation check interval")
    parser.add_argument("--grad-batches", type=int, default=1, help="Number of batches to accumulate")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--cfg", type=str, help="Path to config file")
    parser.add_argument("--val", action='store_true', help="Run validation mode only")
    parser.add_argument("--export", action='store_true', help="Export pre-trained model to onnx")
    parser.add_argument("--export-dir", type=str, default="./", help="Export dir")
    args = parser.parse_args()
    main(args)
