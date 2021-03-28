import os
import torch
import numpy as np
from generate import *
from math import sqrt
import argparse


def main(args):
	alpha = args.alpha
    lr = args.lr
    phi = args.phi
    beta = args.beta
    checkpoint = args.checkpoint

    if beta == None:
        beta = round(sqrt(2/alpha),3)
	

	nnClassCount = 3
	b_sz = args.bs

	# checkpoint='m-epoch-5-batchId-12500-aurocMean-0.5371114078483586.pth.tar'

	alpha = alpha ** phi
	beta = beta ** phi

	device= args.device

	# model, total_macs = give_model(alpha,beta,nnClassCount)
	# print(f"{total_macs} is the number of macs.")
	model = nn.Sequential(model, nn.Sigmoid())
	model.to(device)

	onnx_model="efficientnet_model.onnx"

	dummy_input=torch.randn(b_sz, 3, 1024, 1024)

	torch.onnx.export(model, dummy_input, onnx_model, verbose=False)
	del model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",required=False, help="alpha for the model",default= (11/6) ,type = float)
    parser.add_argument("--phi",required=False, help="Phi for the model.",default= 1.0 ,type = float)
    parser.add_argument("--beta",required=False, help="Beta for the model.",default= None ,type = float)
    parser.add_argument("--checkpoint",required=False, help="File location of model weights",default= None ,type = str)
    parser.add_argument("--bs",required=False, help="Batchsize")
    parser.add_argument("--device",required=True, default='cpu' help="CPU/GPU",type = str)
    args = parser.parse_args()
    
    
    main(args)




