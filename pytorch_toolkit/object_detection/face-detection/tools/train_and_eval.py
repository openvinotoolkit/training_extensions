import argparse
import subprocess
import os

from mmcv.utils import Config

from eval import eval


def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.run('python -m pip install gdown'.split(' '))
    finally:
        globals()[package] = importlib.import_module(package)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('config')
    args.add_argument('gpu_num')
    args.add_argument('--out')
    args.add_argument('--compute_wider_metrics', action='store_true')
    args.add_argument('--wider_dir', default='wider')

    return args.parse_args()


def main():
    install_and_import('gdown')
    args = parse_args()

    mmdetection_tools = '../../external/mmdetection/tools'

    subprocess.run(f'{mmdetection_tools}/dist_train.sh'
                   f' {args.config}'
                   f' {args.gpu_num}'.split(' '))

    cfg = Config.fromfile(args.config)

    eval(args.config, snapshot=os.path.join(cfg.work_dir, "latest.pth"))


if __name__ == '__main__':
    main()
