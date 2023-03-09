# A Two-Stage Multiple Instance Learning Framework for the Detection of Breast Cancer in Mammograms

<div id="abs">

Mammograms are commonly employed in the large scale screening of breast cancer which is primarily characterized by the presence of malignant masses. However, automated image-level detection of malignancy is a challenging task given the small size of the mass regions and difficulty in discriminating between malignant, benign mass and healthy dense fibro-glandular tissue. To address these issues, we explore
a two-stage Multiple Instance Learning (MIL) framework. A Convolutional Neural Network (CNN) is trained in the first stage to extract local candidate patches in the mammograms that may contain either a benign or malignant mass. The second stage employs a MIL strategy for an image level benign vs. malignant classification. A global image-level feature is computed as a weighted average of patch-level features
learned using a CNN. Restricting the MIL only to the candidate
patches extracted in Stage 1 led to a significant improvement in classification performance in comparison to a dense extraction of patches from the entire mammogram. Our method performed well on the task of localization of masses with an average Precision/Recall of 0.76/0.80 and acheived an average AUC of 0.91 on the imagelevel classification task using a five-fold cross-validation on the INbreast dataset.


>**Paper** : C. K. Sarath, A. Chakravarty, N. Ghosh, T. Sarkar, R. Sethuraman and D. Sheet, **"A Two-Stage Multiple Instance Learning Framework for the Detection of Breast Cancer in Mammograms,"** 2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), 2020. </br> _Access the paper via_ 
[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9176427).

BibTeX reference to cite, if you use it:

```bibtex
@INPROCEEDINGS{9176427,
  author={Sarath, Chandra K. and Chakravarty, Arunava and Ghosh, Nirmalya and Sarkar, Tandra and Sethuraman, Ramanathan and Sheet, Debdoot},
  booktitle={2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)}, 
  title={A Two-Stage Multiple Instance Learning Framework for the Detection of Breast Cancer in Mammograms}, 
  year={2020},
  volume={},
  number={},
  pages={1128-1131},
  doi={10.1109/EMBC44109.2020.9176427}}
```

<img src="./media/mil_pipeline.png" width="900" height="200">


## Dataset used

The models made available in this repo are models trained using [RBIS-DDSM dataset](https://ieee-dataport.org/documents/re-curated-breast-imaging-subset-ddsm-dataset-rbis-ddsm).

> **Dataset:** </br>
Arunava Chakravarty, Tandra Sarkar , Rakshith Sathish, Ramanathan Sethuraman, Debdoot Sheet, February 2, 2022, "Re-curated Breast Imaging Subset DDSM Dataset (RBIS-DDSM)", IEEE Dataport, doi: https://dx.doi.org/10.21227/nqp1-sp19. </br>
_Access the dataset via_ [**IEEE Dataport**](https://ieee-dataport.org/documents/re-curated-breast-imaging-subset-ddsm-dataset-rbis-ddsm)


BibTeX reference to cite, if you use it:

```
@data{nqp1-sp19-22,
doi = {10.21227/nqp1-sp19},
url = {https://dx.doi.org/10.21227/nqp1-sp19},
author = {Chakravarty, Arunava and Sarkar , Tandra and Sathish, Rakshith and Sethuraman, Ramanathan and Sheet, Debdoot},
publisher = {IEEE Dataport},
title = {Re-curated Breast Imaging Subset DDSM Dataset (RBIS-DDSM)},
year = {2022} }
```

## Network Architecture

The block diagram and CNN architecture used in Stage 1 to detect the bounding boxes of the mass regions in the mammogram.
<img src="./media/mil_stage1_arch.png" width="700" height="400">

The architecture of the patch level CNN is shown below
</br>
<img src="./media/mil_stage2_arch.png" width="500" height="200">

## Results

| Metric      | Score |
| ----------- | ----- |
| Sensitivity | 0.63  |
| Specificity | 0.45  |
| Accuracy    | 0.90  |

> **Note**: The newtork was trained for only 25 epochs. 

Performance of the network, when trained and evaluated using the Inbreast dataset as reported in the publication is given below.

|   Metric    | Score |
| ----------- | ----- |
| Sensitivity | 0.96  |
| Specificity | 0.77  |
| Accuracy    | 0.86  |


## Trained Models

Download `.pth` checkpoint for stage1 mass localisation model with the following [link](http://kliv.iitkgp.ac.in/projects/miriad/model_weights/bmi5/checkpoint_stage1.zip).

Download `.pth` checkpoint for stage2 patch level model with the following [link](http://kliv.iitkgp.ac.in/projects/miriad/model_weights/bmi5/checkpoint_stage2.zip)


## System Specifications

* Ubuntu\* 16.04
* Python\* 3.6
* NVidia\* GPU for training
* 16GB RAM for inference

## Train

1. Download the [RBIS-DDSM](https://ieee-dataport.org/documents/re-curated-breast-imaging-subset-ddsm-dataset-rbis-ddsm) dataset
2. Create the directory tree
3. Prepare the training dataset
4. Run the training script

## Code and Directory Organisation

```
mammogram_screening/
	mammogram_screening/
      stage1/
        data_prep_rbis.py
        inference_mass_localization.py
        train_stage1.py
      stage2/
        inference_stage2.py
        step2_get_predictions_for_all.py
        step3_get_patches.py
        train_stage2.py
      train_utils/
        downloader.py
        downloader.py
        export.py
        get_config.py
        loss_functions.py
        models.py
        train_functions.py
        transforms.py
        val_function.py
	configs/
      download_configs.json
      network_stage1_configs.json
      network_stage2_configs.json
	media/
	tests/
      test_1_export.py
      test_2_train.py
      test_3_inference.py
	init_venv.sh
	README.md
	requirements.txt
	setup.py
```

# Using the code

## Create environment

Create a virtual environment with all dependencies using 

`sh init_venv.sh`

Activate the environment using `source venv/bin/activate`


## How to run Stage 1
Follow the below steps to reproduce

### Prepare Training Dataset

`python -m mammogram_screening.stage1.data_prep_rbis`

### Run Training

`python -m mammogram_screening.stage1.train_stage1`

### Run Inference

`python -m mammogram_screening.stage1.inference_mass_localisation`


## How to run Stage 2

### Prepare Training Dataset


`python -m mammogram_screening.stage2.step2_get_predictions_for_all`

`python -m mammogram_screening.stage2.step3_get_patches`

### Run Training

`python -m mammogram_screening.stage2.train_stage2`


### Run Inference

`python -m mammogram_screening.stage2.inference_stage2`


## Run Tests

Necessary unit tests have been provided in the tests directory. The sample/toy dataset to be used in the tests can also be downloaded from [here](http://miriad.digital-health.one/sample_data/bmi5/rbis_ddsm_sample.zip).

## **Acknowledgement**

This work is undertaken as part of Intel India Grand Challenge 2016 Project MIRIAD: Many Incarnations of Screening of Radiology for High Throughput Disease Screening via Multiple Instance Reinforcement Learning with Adversarial Deep Neural Networks, sponsored by Intel Technology India Pvt. Ltd., Bangalore, India.


**Principal Investigators**

<a href="https://www.linkedin.com/in/debdoot/">Dr Debdoot Sheet</a>,<a href="http://www.iitkgp.ac.in/department/EE/faculty/ee-nirmalya"> Dr Nirmalya Ghosh (Co-PI) </a></br>
Department of Electrical Engineering,</br>
Indian Institute of Technology Kharagpur</br>
email: debdoot@ee.iitkgp.ac.in, nirmalya@ee.iitkgp.ac.in

<a href="https://www.linkedin.com/in/ramanathan-sethuraman-27a12aba/">Dr Ramanathan Sethuraman</a>,</br>
Intel Technology India Pvt. Ltd.</br>
email: ramanathan.sethuraman@intel.com

**Contributor**

The codes/model was contributed to the OpenVINO project by

<a href="https://github.com/Rakshith2597"> Rakshith Sathish</a>,</br>
Advanced Technology Development Center,</br>
Indian Institute of Technology Kharagpur</br>
email: rakshith.sathish@kgpian.iitkgp.ac.in</br>
Github username: Rakshith2597

<a href="https://www.linkedin.com/in/arunava-chakravarty-b1736b158/">Arunava Chakravarty</a>, </br>
Department of Electrical Engineering, </br>
Indian Institute of Technology Kharagpur</br>
email: arunavachakravarty1206@gmail.com </br>
