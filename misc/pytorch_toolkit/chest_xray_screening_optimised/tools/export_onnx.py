import torch
import torch.nn as nn
from math import sqrt
import argparse
from generate import *


def main(args):

    # Specifying the variables for EfficientNet based optimisation

    alpha = args.alpha
    phi = args.phi
    beta = args.beta
    alpha = alpha ** phi
    beta = beta ** phi

    if beta is None:
        beta = round(sqrt(2 / alpha), 3)

    batch_size = args.bs
    class_count = 3             # Since objective is 3 class classification
    device = args.device

    model, _ = give_model(alpha, beta, class_count)
    model = nn.Sequential(model, nn.Sigmoid())
    model.to(device)

    onnx_model = args.modelname + ".onnx"  # Model name
    dummy_input = torch.randn(batch_size, 3, 1024, 1024)
    torch.onnx.export(model, dummy_input, onnx_model, verbose=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha",
        required=False,
        help="alpha for the model",
        default=(
            11 / 6),
        type=float)
    parser.add_argument(
        "--phi",
        required=False,
        help="Phi for the model.",
        default=1.0,
        type=float)
    parser.add_argument(
        "--beta",
        required=False,
        help="Beta for the model.",
        default=None,
        type=float)
    parser.add_argument("--bs", required=False, help="Batchsize")
    parser.add_argument(
        "--modelname",
        required=True,
        help="ONNX model will be saved with this filename",
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
