from torch.utils import data
from PIL import Image
import os
import random
import torch
import pandas as pd


class CustomDatasetPhase1(data.Dataset):
	
    def __init__(self, path_to_dataset, files256=None,files128=None,split=None,
				transform_images = None, transform_masks = None,
				images_path_rel = '.', masks_path_rel = '.',
				preserve_names = False):
        self.path_to_dataset = os.path.abspath(path_to_dataset)
        self.files256 = files256
        self.files128 = files128
        self.images_path_rel = images_path_rel # relative path to images
        self.masks_path_rel = masks_path_rel # relative path to masks (same as images)
        self.transform_images = transform_images # transforms
        self.transform_masks = transform_masks # transforms
        self.preserve_names = preserve_names # not important, debugging stuff
        self.split = split
        self.test_files = os.listdir(self.path_to_dataset)

        # This is the list of all samples
        self.cropimages = os.listdir(os.path.join(self.path_to_dataset, self.images_path_rel))

        # choose random samples for one epoch
        self.choose_random_subset()

    def choose_random_subset(self, how_many = 0):
        # chooses 'how_many' number of samples and discard the rest
        if how_many == 0:
            self.cropsubset = self.cropimages
        else:
            self.cropsubset = random.sample(self.cropimages, how_many)

    def __len__(self):
        if self.split == 'train':
            return min(len(self.files256),len(self.files128))
        else:
            return len(os.listdir(self.path_to_dataset))

    def __getitem__(self, i):
        # indexing function

        if not hasattr(self, 'cropsubset'):
            # if not chosen a subset, randomly chose the default 'how_many'
            self.choose_random_subset()

        # Read the images and masks (same as images)

        if self.split == 'train':
            fname256 = self.files256[i]
            fname128 = []
            image256 = Image.open(os.path.join(self.path_to_dataset, fname256))
            image128 = Image.open(os.path.join(self.path_to_dataset, fname128))
            # mask = Image.open(os.path.join(self.path_to_dataset, self.masks_path_rel, self.cropsubset[i]))

            # usual transformation apply
            if self.transform_images is not None:
                image256 = self.transform_images(image256)
                image128 = self.transform_images(image128)

            return image256, image128
        else:
            image = Image.open(os.path.join(self.path_to_dataset,self.test_files[i]))
            if self.transform_images is not None:
                image = self.transform_images(image)
            return image, self.test_files[i]
    
    
class CustomDatasetPhase2(data.Dataset):

    def __init__(self, path_to_latent, path_to_gdtruth,
                 transform_images=None, transform_masks=None,
                 mod=0, preserve_name=False):

        self.path_to_latent = path_to_latent  # root folder of the dataset
        self.path_to_gdtruth = path_to_gdtruth
        self.transform_images = transform_images  # transforms
        self.transform_masks = transform_masks  # transforms
        self.mod = mod
        self.preserve_name = preserve_name
        self.list_latent = os.listdir(self.path_to_latent)
        self.list_gdtruth = os.listdir(self.path_to_gdtruth)
        self.dataset = ["cbis", "luna"]

        # This is the list of all samples
        self.cropimages = self.list_latent

        # choose random samples for one epoch
        self.choose_random_subset()

    def choose_random_subset(self, how_many=0):
        # chooses 'how_many' number of samples and discard the rest
        if how_many == 0:
            self.cropsubset = self.cropimages
        else:
            self.cropsubset = random.sample(self.cropimages, how_many)

    def __len__(self):
        return len(self.list_latent)

    def __getitem__(self, index):

        path_latent = os.path.join(
            self.path_to_latent, self.list_latent[index])
        object = pd.read_pickle(path_latent)
        image = object["latent_int"]

        file_name = self.list_latent[index].rsplit('.latent')[0].rsplit('_')

        file_join = '_'.join(file_name)
        
        mask = Image.open(os.path.join(
                    self.path_to_gdtruth, self.list_gdtruth[0]))
                    
        for i in range(len(self.list_gdtruth)):
            if(file_join == self.list_gdtruth[i]):
                mask = Image.open(os.path.join(
                    self.path_to_gdtruth, self.list_gdtruth[i]))

        # usual transformation apply
        if self.transform_images is not None:
            image = torch.Tensor(image)

        if self.transform_masks is not None:
            mask = self.transform_masks(mask)

        if self.preserve_name == True:
            return image, mask, file_join
        else:
            return image, mask
