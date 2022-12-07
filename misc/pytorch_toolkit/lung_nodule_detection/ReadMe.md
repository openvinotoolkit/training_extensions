# Lung Segmentation and Nodule Detection in Computed Tomography Scan using a Convolutional Neural Network Trained Adversarially using Turing Test Loss

Lung cancer is the most common form of cancer found worldwide with a high mortality rate. Early detection of pulmonary nodules by screening with a low-dose computed tomography (CT) scan is crucial for its effective clinical management. Nodules which are symptomatic of malignancy occupy about 0.0125 - 0.025% of volume in a CT scan of a patient. Manual screening of all slices is a tedious task and presents a high risk of human errors. To tackle this problem we propose a computationally efficient two-stage framework. In the first stage, a convolutional neural network (CNN) trained adversarially using Turing test loss segments the lung region.
In the second stage, patches sampled from the segmented region are then classified to detect the presence of nodules. The proposed method is experimentally validated on the LUNA16 challenge dataset with a dice coefficient of **0.984 ± 0.0007** for 10-fold cross-validation.
**Paper** : [arXiv](https://arxiv.org/abs/2006.09308v1) 
BibTeX reference to cite, if you use it:

```bibtex
@inproceedings{Sathish2020LungSA, 
title={Lung Segmentation and Nodule Detection in
 Computed Tomography Scan using a Convolutional Neural Network
 Trained Adversarially using Turing Test Loss},
 author={Rakshith Sathish and Rachana Sathish
 and Ramanathan Sethuraman and Debdoot Sheet}, 
 year={2020} } 
```
## Dataset used

 The proposed method is experimentally validated by performing 10-fold cross-validation on the LUNA16 challenge dataset. 
 >Dataset download page: [https://luna16.grand-challenge.org/](https://luna16.grand-challenge.org/) 

The dataset consists of CT volumes from 880 subjects, provided as ten subsets for 10-fold cross-validation. In each fold of the experiment, eight subsets from the dataset were used for training and one each for validation and testing. The annotations provided includes binary masks for lung segmentation and, coordinates and spherical diameter of nodules present in each slice. LIDC-IDRI dataset from which LUNA16 is derived has nodule annotations in the form of contours which preserves its actual shape. Therefore, we use annotations from LUNA dataset only in Stage 1. The annotations for the nodules from the LIDC dataset is used in Stage 2 (nodule detection) to determine the presence of nodules in image patches.
> Dataset download page: [https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)

The ground truth annotations were marked in a two-phase image annotation process performed by four experienced thoracic radiologists. Systematic sampling of slices from the CT volumes was performed to ensure equal distribution of slices with and without the presence of nodules.

>**Note**: Systematically sampled slice numbers/images to be used are given in the repository inside the data preparation folder.

>**License**: Both the datasets are published by the creators under [Creative Commons Attribution 3.0 Unported License](https://creativecommons.org/licenses/by/3.0/)

# Using the code

## Code Organization
Code directory is organised into 3 subfolders; Data preparation, Training and Evaluation. Each of these subfolders has a .py file and a package folder containing function definitions. 
## Requirements

Create a conda virtual environment with
```
conda create --name <env> --file requirements.txt
```
This would create a virtual environment with all the necessary packages of the same version used during development.
 
## Data preparation
Follow the below steps to prepare and organise the data for training.
> Details about the arguments being passed and its purpose is explained within the code. To see the details run `python prepare_data.py -h`

> Make sure the dataset has been properly downloaded and extracted before proceeding.

1. ` python prepare_data.py  --genslices --masktype <type> --datasetpath <path> --save_path <path>  `
	This step extracts induvidual CT slices from the CT volumes provided in the dataset. Each of these slices are saved seperately as npy files with filename in the format `[series_uid]_slice[sliceno].npy`.
	Perform the above step for masktype nodule and lung seperately before proceeding to the next step.

2. `python prepare_data.py --createfolds --datapath <path> --savepath <path>
--datasetpath <path> `
	The above step first classifies the slices into two categories, positive and negative based on the presence of nodules in them. On completion, the dataset which consists of CT volumes from 880 subjects, provided as ten subsets are divided into 10-folds for cross-validation. In each fold of the experiment, eight subsets from the dataset are separated for training and one each for validation and testing. A balanced dataset consisting of an equal number (approx.) of positive and negative slices is identified for each fold. Filenames of these slices of each fold are stored in separate JSON files.

3. `python prepare_data.py --genpatch --jsonpath <path> --foldno <int> --category <type> --data_path <path> --lungsegpath <path> --savepath <path> --patchtype <type> `
	The above step generates patches which are used to train the classifier network.

4. `python prepare_data.py --visualize --seriesuid <str> --sliceno <int> --datapath <path> --savepath <path>`
	To visualize a particular slice use the above line. 

## Training

> Details about the arguments being passed and its purpose is explained within the code. To see the details run `python train_network.py -h`

To train the lung segmentation network without the discriminator execute the following line of code.
`python train_network.py --lungseg --foldno <int> --savepath <path> --jsonpath <path> --datapath <path> --lungsegpath <path> --network <str> --epochs <int:optional> `

To train the lung segmentation network with discrimator and turing test loss, execute 
`python train_network.py --lungsegadv --foldno <int> --savepath <path> --jsonpath <path> --datapath <path> --lungsegpath <path> --network <str> --epochs <int:optional> `

To train the patch classifier network execute
`python patch_classifier --savepath <path> --imgpath <path> --epochs <int:optional> `


## Evaluation

> Details about the arguments being passed and its purpose is explained within the code. To see the details run `python inference.py -h`

To evaluate the segmentation models execute
 `python inference.py --lunseg --foldno <int> --savepath <path> --jsonpath <path> --network <str>`

To evaluate the classifier network execute
`python inference.py --patchclass --savepath <path> --imgpath <path>`

## Pre-trained Models
## Results
## Support

If you face any issues while executing the codes, raise an issue with rakshith.sathish@gmail.com

# Authors and Acknowledgment
- Rakshith Sathish
- Rachana Sathish
- Ramanathan Sethuraman
- Debdoot Sheet
> This work was supported through a research grant from Intel India Grand
Challenge 2016 for Project MIRIAD

 # License
Copyright [2020] [IITKLIV]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
