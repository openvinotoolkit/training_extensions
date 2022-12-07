#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm as tq
import time
from torch.utils import data
import os
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from .sumnet_bn_vgg import SUMNet
from .r2unet import R2U_Net
from .r2unet import U_Net
from torchvision import transforms
import json
from PIL import Image
from .dataloader import LungDataLoader
from .utils import dice_coefficient


def infer_lungseg(fold_no,savepath,network,jsonpath):
	""" Inference script for lung segmentation

	Parameters
	----------
	fold_no: int
		Fold number to which action is to be performed
	savepath: str
		Folder location to save the results
	network: str
		Network name
	jsonpath:
		Folder location where file is to be stored

	Returns
	-------
	None

	"""

	fold = 'fold'+str(fold_no)

	savePath = savepath+network+'/'+fold+'/'
	if not os.path.isdir(savePath):
		os.makedirs(savePath)

	with open(jsonpath+fold+'_pos_neg_eq.json') as f:
	    json_file = json.load(f)
	    test_set = json_file['test_set']

	testDset = LungDataLoader(is_transform=True,json_file=json_file,split="test_set",img_size=512)
	testDataLoader = data.DataLoader(testDset,batch_size=1,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)

	testBatches = 0
	testDice_lungs = 0

	if network == 'sumnet':
		net = SUMNet(in_ch=1,out_ch=2)
	elif network == 'unet':
		net = U_Net(img_ch=1,output_ch=2)
	else:
		net = R2U_Net(img_ch=1,output_ch=2)


	dice_list = []
	use_gpu = torch.cuda.is_available()

	if use_gpu:
		net = net.cuda()

	net.load_state_dict(torch.load(savePath+network+'_best_lungs.pt'))

	for data1 in tq(testDataLoader):

	    imgs, mask = data1
	    labels = mask
	    if use_gpu:
	        inputs = imgs.cuda()
	        labels = labels.cuda()

	    net_out = net(Variable(inputs))
	    net_out_sf = F.softmax(net_out.data,dim=1)

	    test_dice = dice_coefficient(net_out_sf,torch.argmax(labels,dim=1))
	    
	    pred_max = torch.argmax(net_out_sf, dim=1)    
	    preds = torch.zeros(pred_max.shape)
	    preds[pred_max == 1] = 1 

        if not os.path.isdir(savePath+'seg_results/GT/'):
            os.makedirs(savePath+'seg_results/GT/')
            np.save(savePath+'seg_results/GT/image'+str(testBatches),labels[:,1].cpu())
        else:
            np.save(savePath+'seg_results/GT/image'+str(testBatches),labels[:,1].cpu())

        if not os.path.isdir(savePath+'seg_results/pred/'):
            os.makedirs(savePath+'seg_results/pred/')
            np.save(savePath+'seg_results/pred/image'+str(testBatches),preds.cpu())
        else:
            np.save(savePath+'seg_results/pred/image'+str(testBatches),preds.cpu())

        if not os.path.isdir(savePath+'seg_results/image/'):
            os.makedirs(savePath+'seg_results/image/')
            np.save(savePath+'seg_results/image/image'+str(testBatches),inputs.cpu())
        else:
            np.save(savePath+'seg_results/image/image'+str(testBatches),inputs.cpu())

	    testDice_lungs += test_dice[0]
	    dice_list.append(test_dice[0].cpu())
	    testBatches += 1
	#     if testBatches>1:
	#         break
	    
	dice = np.mean(dice_list)
	print("Result:",fold,dice)    


	#Plots distribution of min values per volume
	plt.figure()
	plt.title('Distribution of Dice values')
	plt.hist(dice_list)
	plt.xlabel('Dice score')
	plt.ylabel('No. of Slices')
	plt.savefig(savePath+'dice_dist.jpg')
	# plt.show()
	plt.close()



def visualise_seg(loadpath):
	"""
	To visualise the segmentation performance(Qualitative results)

	Parameters
	----------

	loadpath: str
		Folder location from where the files are to be loaded

	Returns
	-------
	None

	"""

	image_list = os.listdir(loadpath+'GT/')
	count = 0
	for i in tq(image_list):
	    img = np.load(loadpath+'image/'+i)
	    GT = np.load(loadpath+'GT/'+i)
	    pred = np.load(loadpath+'pred/'+i)

	    plt.figure(figsize = [15,5])
	    plt.subplot(141)
	    plt.axis('off')
	    plt.title('Input Image')
	    plt.imshow(img[0][0],cmap = 'gray')
	    plt.subplot(142)
	    plt.axis('off')
	    plt.title('GT')
	    plt.imshow(GT[0],cmap = 'gray')     
	    plt.subplot(143)
	    plt.axis('off')
	    plt.title('Pred')
	    plt.imshow(pred[0],cmap = 'gray')
	    plt.subplot(144)
	    plt.title('GT - Pred')
	    plt.axis('off')
	    test = GT[0]-pred[0]
	    test[test>0] = 1
	    test[test<=0] = 0
	    plt.imshow(test,cmap = 'gray')
	    count += 1

        if not os.path.isdir(loadpath+'seg_results/op_images/'):
            os.makedirs(loadpath+'seg_results/op_images/')
    	    plt.savefig(loadpath+'seg_results/op_images/img'+str(count)+'.jpg')
        else:
            plt.savefig(loadpath+'seg_results/op_images/img'+str(count)+'.jpg')

	#     if count>10:
	#         break





