import argparse

import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from torchtts.pipelines.pipeline_tts import PipelineTTS
from torchtts.datasets.text.symbols import symbols
from torchtts.utils import load_cfg



def build_logger(cfg):
    return getattr(pl_loggers, cfg.type)(**cfg.params)

def main(args):
    cfg = load_cfg(args.cfg)
    cfg.model.encoder.num_chars = len(symbols)

    pl.seed_everything(10)

    pipeline = PipelineTTS(cfg)

    if torch.cuda.is_available():
        args.gpus = torch.cuda.device_count()

    if args.ckpt is not None:
        pipeline.load_state_dict(
            torch.load(args.ckpt, map_location="cpu")["state_dict"]
        )

    logger = build_logger(cfg.logger)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.trainer.chkpt_dir,
        save_top_k=True,
        save_last=True,
        verbose=True,
        monitor='loss_val',
        mode='min',
        every_n_val_epochs=1,
        filename='{epoch}-{loss_val:.2f}'
    )

    # create trainer
    trainer = pl.Trainer(
        gradient_clip_val=1.0,
        gpus=args.gpus,
        logger=logger,
        max_epochs=cfg.trainer.max_epochs,
        accumulate_grad_batches=args.grad_batches,
        distributed_backend=args.distributed_backend,
        precision=32,
        callbacks=[checkpoint_callback],
        val_check_interval=0.95,
        replace_sampler_ddp=False
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
    parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                        help='supports three options dp, ddp, ddp2')
    # configure
    parser.add_argument("--cfg", type=str, help="path to config file")
    parser.add_argument("--val-check-interval", type=int, default=100, help="validation check interval")
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
