import os
import torch
import numpy as np
from generate import *
from math import sqrt
import argparse
from torchvision import models


def main(args):

    checkpoint = args.checkpoint

    nnClassCount = 3

    device = args.device

    model = models.densenet121(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(1024, nnClassCount), nn.Sigmoid())
    # print(f"{total_macs} is the number of macs.")
    model = nn.Sequential(model, nn.Sigmoid())
    model.to(device)

    onnx_model = "densenet_model.onnx"

    dummy_input = torch.randn(1, 3, 1024, 1024)

    torch.onnx.export(model, dummy_input, onnx_model, verbose=False)
    del model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=False,
        help="File location of model weights",
        default=None,
        type=str)
    parser.add_argument(
        "--device",
        required=True,
        default='cpu',
        help="CPU/GPU",
        type=str)
    args = parser.parse_args()

    main(args)
