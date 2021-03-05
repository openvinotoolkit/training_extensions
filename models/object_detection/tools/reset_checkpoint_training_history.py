import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Resets optimizer, epoch and iter numbers in a checkpoint')
    parser.add_argument('input', help='train config file path')
    parser.add_argument('output', help='path to output snapshot')

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = torch.load(args.input)
    checkpoint['meta'] = {'epoch': 0, 'iter': 0}
    del checkpoint['optimizer']
    torch.save(checkpoint, args.input)

    return 0


if __name__ == '__main__':
    main()
