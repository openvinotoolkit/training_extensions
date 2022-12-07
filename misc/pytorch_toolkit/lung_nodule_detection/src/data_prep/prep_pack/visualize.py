#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt

def visualize_data(series_uid,slice_num,datapath,savepath):
	""" To visualize the image and nodule masks of the dataset

	Parameters
	----------

	series_uid: str
		Series_instance_uid or filename of the image to visualize
	slice_num: int
		Slice number to visulaize
	datapath: str
		Folder location where image and mask numpy is stored.
	savepath: str
		Folder location to save images


	"""
	    
	img_name = series_uid+'_slice'+str(slice_num)+'.npy'
	mask = np.load(datapath+'mask/'+img_name)
	img = np.load(datapath+'img/'+img_name)
	lungseg = np.load(datapath+'lungseg/'+img_name)

	plt.figure()
	plt.subplot(131)
	plt.imshow(img,cmap='gray')
	plt.title('Original Image')
	plt.subplot(132)
	plt.imshow(mask,cmap='gray')
	plt.title('Ground truth (Lung)')	
	plt.subplot(133)
	plt.imshow(mask,cmap='gray')
	plt.title('Ground truth (Nodule)')
	plt.savefig(savepath+'visualization.png')
	plt.show()
	plt.close()

