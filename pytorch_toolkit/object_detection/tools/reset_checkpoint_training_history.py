import argparse

from mmcv import Config
from mmdet.models import build_detector
from mmdet.utils import get_root_logger, ExtendedDictAction
from mmcv.runner.checkpoint import load_checkpoint, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Resets optimizer, epoch and iter numbers in a checkpoint')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint')
    parser.add_argument('output', help='path to output snapshot')
    parser.add_argument('--update_config', nargs='+', action=ExtendedDictAction, help='arguments in dict')

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.update_config is not None:
        cfg.merge_from_dict(args.update_config)

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint)
    save_checkpoint(model, args.output, meta={'epoch': 0, 'iter': 0})

    return 0


if __name__ == '__main__':
    main()
