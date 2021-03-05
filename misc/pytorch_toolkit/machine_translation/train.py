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
import argparse
import os
import json
import torch
import pytorch_lightning as pl
from core.utils import load_cfg, JSONLogger
from core.trainer import NMTTrainer


def main(args):
    cfg = load_cfg(args.cfg)
    cfg.distributed_backend = args.distributed_backend
    nmt_trainer = NMTTrainer(cfg)
    if args.ckpt is not None:
        nmt_trainer.load_state_dict(torch.load(args.ckpt, map_location="cpu")["state_dict"])

    if args.cli_mode:
        nmt_trainer.cli_mode()
    elif args.to_onnx:
        nmt_trainer.to_onnx(args.onnx_path, args.onnx_denominator)
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=args.output_dir,
            save_top_k=True,
            verbose=True,
            monitor='bleu',
            mode='max'
        )
        if args.log_path is not None:
            logger = JSONLogger(args.log_path)
        else:
            logger = None
        trainer = pl.Trainer(
            gpus=args.gpus,
            logger=logger,
            max_epochs=cfg.trainer.max_epochs,
            accumulate_grad_batches=args.grad_batches,
            distributed_backend=args.distributed_backend,
            checkpoint_callback=checkpoint_callback,
            val_check_interval=args.val_check_interval,
        )
        if not args.eval:
            trainer.fit(nmt_trainer)
        else:
            trainer.test(nmt_trainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--gpus", type=int, default=-1, help="number of available GPUs")
    parser.add_argument('--distributed-backend', type=str, default="ddp", choices=('dp', 'ddp', 'ddp2'),
                        help='supports three options dp, ddp, ddp2')
    # configure
    parser.add_argument("--cfg", type=str, help="path to config file")
    parser.add_argument("--val-check-interval", type=int, default=2000, help="validation check interval")
    parser.add_argument("--grad-batches", type=int, default=1, help="number of batches to accumulate")
    # checkpoints
    parser.add_argument("--output-dir", type=str, default="checkpoints/", help="path to store checkpoints")
    # cli_mode
    parser.add_argument('--cli-mode', action='store_true', help="interactive translation mode")
    # load ckpt
    parser.add_argument('--ckpt', type=str, default=None, help="path to checkpoint")
    # log
    parser.add_argument('--log-path', type=str, default=None, help="path to output log file")
    # convert to onnx
    parser.add_argument('--to-onnx', action='store_true', help="convert model to onnx")
    parser.add_argument('--onnx-path', type=str, default="model.onnx", help="path to onnx model")
    parser.add_argument('--onnx-denominator', type=int, default=8, help="max input length of onnx model")
    # evaluate
    parser.add_argument('--eval', action='store_true', help="run validation only")
    args = parser.parse_args()
    main(args)
