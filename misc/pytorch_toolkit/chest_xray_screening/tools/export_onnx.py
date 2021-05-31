import torch
import argparse
from torchvision import models
import torch.nn as nn

def main(args):

    checkpoint = args.checkpoint
    model_name = args.modelname
    class_count = 3 # Dataset has 3 classes
    device= args.device

    model=models.densenet121(pretrained=True)
    model.classifier=nn.Sequential(nn.Linear(1024, class_count), nn.Sigmoid())
    model = nn.Sequential(model, nn.Sigmoid())
    model.to(device)

    onnx_model= model_name+".onnx"   # Name of the onnx model
    dummy_input=torch.randn(1, 3, 1024, 1024)
    torch.onnx.export(model, dummy_input, onnx_model, verbose=False)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=False, help="File location of model weights",default= None ,type = str)
    parser.add_argument("--modelname",required=True, help="ONNX model will be saved with this filename",default= None ,type = str)
    parser.add_argument("--device",required=True, default='cpu', help="CPU/GPU",type = str)
    args = parser.parse_args()

    main(args)

