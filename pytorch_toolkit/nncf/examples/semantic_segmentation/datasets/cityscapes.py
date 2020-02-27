"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
from collections import OrderedDict
import torch.utils.data as data
import examples.semantic_segmentation.utils.data as data_utils


class Cityscapes(data.Dataset):
    """Cityscapes dataset https://www.cityscapes-dataset.com/.

    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.

    """
    # Training dataset root folders
    train_folder = "leftImg8bit/train"
    train_lbl_folder = "gtFine/train"

    # Validation dataset root folders
    val_folder = "leftImg8bit/val"
    val_lbl_folder = "gtFine/val"

    # Test dataset root folders
    test_folder = "leftImg8bit/val"
    test_lbl_folder = "gtFine/val"

    # Filters to find the images
    img_extension = '.png'
    lbl_name_filter = 'labelIds'

    # The values associated with the 35 classes
    full_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                    32, 33, -1)
    # The values above are remapped to the following
    new_classes = (0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 5, 0, 0, 0, 6, 0, 7,
                   8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 17, 18, 19, 0)

    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
        ('unlabeled', (0, 0, 0)),
        ('road', (128, 64, 128)),
        ('sidewalk', (244, 35, 232)),
        ('building', (70, 70, 70)),
        ('wall', (102, 102, 156)),
        ('fence', (190, 153, 153)),
        ('pole', (153, 153, 153)),
        ('traffic_light', (250, 170, 30)),
        ('traffic_sign', (220, 220, 0)),
        ('vegetation', (107, 142, 35)),
        ('terrain', (152, 251, 152)),
        ('sky', (70, 130, 180)),
        ('person', (220, 20, 60)),
        ('rider', (255, 0, 0)),
        ('car', (0, 0, 142)),
        ('truck', (0, 0, 70)),
        ('bus', (0, 60, 100)),
        ('train', (0, 80, 100)),
        ('motorcycle', (0, 0, 230)),
        ('bicycle', (119, 11, 32))
    ])

    def __init__(self,
                 root,
                 image_set='train',
                 transforms=None,
                 loader=data_utils.pil_loader):
        self.root_dir = root
        self.mode = image_set
        self.transforms = transforms
        self.loader = loader

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = data_utils.get_files(
                os.path.join(self.root_dir, self.train_folder),
                extension_filter=self.img_extension)

            self.train_labels = data_utils.get_files(
                os.path.join(self.root_dir, self.train_lbl_folder),
                name_filter=self.lbl_name_filter,
                extension_filter=self.img_extension)
        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = data_utils.get_files(
                os.path.join(self.root_dir, self.val_folder),
                extension_filter=self.img_extension)

            self.val_labels = data_utils.get_files(
                os.path.join(self.root_dir, self.val_lbl_folder),
                name_filter=self.lbl_name_filter,
                extension_filter=self.img_extension)
        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_data = data_utils.get_files(
                os.path.join(self.root_dir, self.test_folder),
                extension_filter=self.img_extension)

            self.test_labels = data_utils.get_files(
                os.path.join(self.root_dir, self.test_lbl_folder),
                name_filter=self.lbl_name_filter,
                extension_filter=self.img_extension)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.

        """
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)

        # Remap class labels
        label = data_utils.remap(label, self.full_classes, self.new_classes)

        if self.transforms is not None:
            img, label = self.transforms(img, label)

        return img, label

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        if self.mode.lower() == 'val':
            return len(self.val_data)
        if self.mode.lower() == 'test':
            return len(self.test_data)

        raise RuntimeError("Unexpected dataset mode. "
                           "Supported modes are: train, val and test")
