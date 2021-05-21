import argparse
import torch
import json
import pytorch_lightning as pl
from core.utils import load_config
from core.trainer import TimeSeriesTrainer


def main(args):
    cfg = load_config(args.cfg)
    pipeline = TimeSeriesTrainer(cfg)
    if args.ckpt is not None:
        pipeline.load_state_dict(
            torch.load(args.ckpt, map_location="cpu")["state_dict"]
        )

    # checkpoint_callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir,
        save_top_k=True,
        save_last=True,
        verbose=True,
        monitor='loss_val',
        mode='min'
    )

    # create trainer
    trainer = pl.Trainer(
        gradient_clip_val=0.01,
        gpus=args.gpus,
        logger=None,
        max_epochs=cfg.pipeline.epochs,
        accumulate_grad_batches=args.grad_batches,
        distributed_backend=args.distributed_backend,
        precision=32,
        checkpoint_callback=checkpoint_callback,
        val_check_interval=args.val_check_interval,
    )
    if args.test:
        trainer.test(pipeline)
    elif args.to_onnx is not None:
        pipeline.to_onnx(args.to_onnx, args.onnx_batch_size)
    else:
        trainer.fit(pipeline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--gpus", type=int, default=0, help="number of available GPUs")
    parser.add_argument('--distributed-backend', type=str, default=None, choices=('dp', 'ddp', 'ddp2'),
                        help='supports three options dp, ddp, ddp2')
    # configure
    parser.add_argument("--cfg", type=str, help="path to config file")
    parser.add_argument("--val-check-interval", type=int, default=2000, help="validation check interval")
    parser.add_argument("--grad-batches", type=int, default=1, help="number of batches to accumulate")
    # checkpoints
    parser.add_argument("--output-dir", type=str, default="checkpoints/", help="path to store checkpoints")
    # test mode
    parser.add_argument("--test", action="store_true")
    # ckpt
    parser.add_argument("--ckpt", type=str, default=None)
    # onnx export
    parser.add_argument("--to-onnx", type=str, default=None)
    parser.add_argument("--onnx-batch-size", type=int, default=1)

    args = parser.parse_args()
    main(args)
