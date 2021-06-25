"""
Demo code for IVUS2D and BUS
"""

import os
import argparse
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from src.model import GeneratorModel, GeneratorInter

parser = argparse.ArgumentParser()
parser.add_argument('--dilation_factor', type=float, default=0)
args = parser.parse_args()

#Load model

model_wts = "./checkpoints/model.pt"

gen_model = GeneratorModel(in_ch=1)
state_dict = torch.load(model_wts, map_location=torch.device("cpu"))
gen_model.load_state_dict(state_dict["generator1_weights"])

gen_model_freq = GeneratorInter(
                1, gen_model.cpu(), a=args.dilation_factor)
gen_model_freq.load_state_dict(
    state_dict["generator1_weights"], strict=False)


gen_model_freq.eval()

#Prepare Image
stage0 = Image.open("./media/stage0/ivus2d.png") #for bus change ivus2d.png to bus.png
ten_trans = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
stage0 = ten_trans(stage0)
stage0 = stage0.unsqueeze(0)


#Infer
output = gen_model_freq(stage0)

#Save
torchvision.utils.save_image(output.detach().cpu(), "output{}.png".format(str(args.dilation_factor)), normalize=True, scale_each=True)




