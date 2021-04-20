import os
import torch
import numpy as np
from math import sqrt
import argparse


def main(args):

    lr = args.lr
    checkpoint = args.checkpoint
    nnClassCount = 3
    b_sz = args.bs

    device= args.device
    model = nn.Sequential(model, nn.Sigmoid())
    model.to(device)
    onnx_model="efficientnet_model.onnx"

    dummy_input=torch.randn(b_sz, 3, 1024, 1024)

    torch.onnx.export(model, dummy_input, onnx_model, verbose=False)
    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=False, help="File location of model weights",default= None ,type = str)
    parser.add_argument("--bs",required=False, help="Batchsize")
    parser.add_argument("--device",required=True, default='cpu', help="CPU/GPU",type = str)
    args = parser.parse_args()
    main(args)