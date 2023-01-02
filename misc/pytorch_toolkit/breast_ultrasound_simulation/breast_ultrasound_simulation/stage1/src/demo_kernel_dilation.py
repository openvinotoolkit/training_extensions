"""
Demo code for BUS
"""

import os
import argparse
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from model import GeneratorModel, GeneratorInter

parser = argparse.ArgumentParser()
parser.add_argument('--dilation_factor', type=float, default=0)
args = parser.parse_args()

if not os.path.exists("../model_inference_results"):
    os.makedirs("../model_inference_results")

for kernel_dilation_factor in [(0.05 * i) for i in range(11)]:
    model_wts = "../checkpoints/results/latest.pt"

    gen_model = GeneratorModel(in_ch=1)
    state_dict = torch.load(model_wts, map_location=torch.device("cpu"))
    gen_model.load_state_dict(state_dict["generator1_weights"])

    gen_model_freq = GeneratorInter(1, gen_model.cpu(), a=kernel_dilation_factor)
    gen_model_freq.load_state_dict(state_dict["generator1_weights"], strict=False)

    gen_model_freq.eval()

    for image_name in os.listdir("../DATASET_BUS_INFERENCE/stage0"):

        #Prepare Image
        stage0 = Image.open("../DATASET_BUS_INFERENCE/stage0/" + image_name)
        ten_trans = transforms.Compose([transforms.ToTensor()])
        stage0 = ten_trans(stage0)
        stage0 = stage0.unsqueeze(0)

        #Infer
        output = gen_model_freq(stage0)

        #Save
        torchvision.utils.save_image(output.detach().cpu(),
                                     "../model_inference_results/" +
                                     image_name.replace(".", "_") + "_" +
                                     "latest_pt" + ("_dil_{0:.2f}".format(kernel_dilation_factor)).replace(".", "_") +
                                     ".png",
                                     normalize=True, scale_each=True)
