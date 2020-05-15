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
    args.add_argument('config',
                      help='A path to model training configuration file (.py).')
    args.add_argument('gpu_num',
                      help='A number of GPU to use in training.')
    args.add_argument('out',
                      help='A path to output file where models metrics will be saved (.yml).')
    args.add_argument('--wider_dir',
                      help='Specify this  path if you would like to test your model on WiderFace dataset.')

    return args.parse_args()


def main():
    install_and_import('gdown')
    args = parse_args()

    mmdetection_tools = '../../external/mmdetection/tools'

    subprocess.run(f'{mmdetection_tools}/dist_train.sh'
                   f' {args.config}'
                   f' {args.gpu_num}'.split(' '))

    cfg = Config.fromfile(args.config)

    eval(args.config, os.path.join(cfg.work_dir, "latest.pth"), args.wider_dir, args.out)


if __name__ == '__main__':
    main()
