# EfficientNet based Deep Neural Architecture Search for Chest X-Ray Screening


Chest radiographs are primarily employed for the screening of pulmonary and cardio-/thoracic conditions. Being undertaken at primary healthcare centers, they require the presence of an on-premise reporting Radiologist, which is a challenge in low and middle income countries. This has inspired the development of machine learning based automation of the screening process. While recent efforts demonstrate a performance benchmark using an ensemble of deep convolutional neural networks (CNN), our systematic search over multiple standard CNN architectures identified single candidate CNN models whose classification performances were found to be at par with ensembles.

  

This code is to perform architecture search for models trained to detect multiple comorbid chest diseases in chest X-rays. In this code, we performed experiments to fine-tune the DenseNet-121 ([paper](https://arxiv.org/pdf/1608.06993.pdf))CNN architecture for this task ([link](https://arxiv.org/pdf/2004.11693.pdf)). The method in this repository differs from the paper in a few aspects; In the paper, the authors classify an X-ray image into one or more of the 14 classes following a multi-hot encoding on account of co-morbidity of diseases, while in this repository we present the approach to classify a Chest X-ray image into any of the applicable 3 classes. This modified DenseNet-121 is retrained using the publicly available 2018 RSNA Pneumonia Detection Challenge dataset ([link](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)).

  

Few example images from the dataset


**![](https://lh6.googleusercontent.com/4T5-Oc1pyJxXEmXsujHYvNdC6cK_Behf7S5FPT-B6UJgpyAqRcli1_lVgepup0VMpjRyPtvAZdXv5FVRCOhKsJVJCOy1iQWs-F_XroXktc25fSf6ZqqrUBXWR1A62WWHN_qpdVL9){:height="250" width="250"}
Class = 0**

**![](https://lh5.googleusercontent.com/Gen7YqQBLbOl4adwqc449f-0RBt9f3d1Vdlob5c5miYV4UMpEAmCTSzlOU0JCOGxz5aCM-lLihnh5-BIAGUrThWzpJdsR3n94QAfwbHsv4CgGFbXBTwYYOGpP35IYaelLUty4EGD){:height="250" width="250"}
Class = 1**

**![](https://lh3.googleusercontent.com/UUPBhjzjRc01fJBuQfyGkWQ5RbtnwVOHggVj06GB70ltNL1K0G4R63rVbOWZNnZwR6n1-xS4h2eUzCzUo0ocHEo4fI9ZO6KqUi4RDlQfJEyAqBBWx-c1ze6R-Ph098rcd7OnHLxo){:height="250" width="250"}
Class = 2**

A systematic search was performed over a set of CNN architectures by scaling the width and depth of the standard DenseNet using the EfficientNet approach [link](http://proceedings.mlr.press/v97/tan19a/tan19a.pdf). The details of our experiments based on EfficientNet is summarized in the image below

![](https://lh5.googleusercontent.com/gzodIpXuS8SAsvhNJkFKrztuzTF77B_pipyhkCy3LrVENcQUe5z4s6Lk62A9iM3MICCrOaM5EnzvrgN-MR2QGvnTUS1xwBuq6DzlHQlC2m8bbOcngyMf1LmomPRq_EA-fSWIM-aY)

## Network Architecture:

We have used a DenseNet-121 as the base architecture.

![6: A schematic illustration of the DenseNet-121 ...](https://lh6.googleusercontent.com/ziwy55LfUqzzErhcy0Cw418LbeCWpfH_liD3dXNrae8yVnV91rnCWsokLUVO0NVUuUeNHIG6bnkV3J7jNNT5U6DDr2Y78Z60NW-2ACUEuY53k6B7C6x1Q9HFrJ-1yJZNM1vyMPdg)

  

## Results

AUROC scores for each class and Mean AUROC score is reported for Pneumonia detection.
| Class | AUROC Score  |
|--|--|
| Lung Opacity | 0.553 |
| Normal | 0.4699 |
| No Lung Opacity/ Not Normal | 0.5185 |

Mean AUROC score: 0.5138

  

AUROC Score for the base network, when trained and evaluated using the CheXpert dataset provided by Stanford University [link](https://stanfordmlgroup.github.io/competitions/chexpert/), is given below.

| Class | AUROC Score  |
|--|--|
| Lung Opacity | 0.55 |
| Atelectasis | 0.55 |
| Enlarged Cardiomediastinum | 0.15 |
| Cardiomegaly | 0.45 |
| Pleural Effusion | 0.79 |
| Edema | 0.69 |
| Consolidation | 0.83 |
| Pneumonia | 0.11 |
| Pneumothorax | 0.63 |
| Lung lesion | 0.79 |
| Pleural Other | 0.04 |


  

### Model
Download checkpoint with the following [link]()

The OpenVINO IR can be found at [link]()

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* 3.6
* NVidia\* GPU for training
* 16GB RAM for inference

## Train

1. Download the [RSNA Dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)
2. Create the directory tree
3. Prepare the training dataset
4. Run the training

## **Code Structure**

1. `train.py` in main directory contains the code for training the model
3. `generate.py` CNN model defenition is defined in this file. Additionally it also computes and returns the FLOPs for the CNN model.
4. Inside tools directory, `test.py` and `export_onnx.py` file is provided which can be used for testing the model and generating the onnx representation of the trained model.

### Creating the Directory Tree

The data directory should contain two subdirectories: preprocessed data for training and original
[data](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge).

```
+-- data
|   +-- original
|   +-- preprocessed
  | +-- train_data
  | +-- test_data
```
Model weights will be stored in your current/parent directory.
### Prepare the Training Dataset
```
python tools/prepare_training_dataset.py \
  --input_path data/original \
  --output_path data/preprocessed \
```
You should get a set of folders in the `output_path` with preprocessed data. 

### Run Training

Run the `main.py` script:
```
python3 train.py \
  --alpha \
  --beta \
  --phi \
  --checkpoint \
  --bs \
  --lr test_run \
  --imgpath \
  --epochs 
```

## How to Perform Prediction

Ensure that the test directory contains a series of X-ray samples in the JPEG format with the `.jpg` extension.

### Run Test
```
python3 test.py \
  --alpha \
  --beta \
  --phi \
  --checkpoint \
  --bs \
  --imgpath \

```
## Acknowledgement
This work is undertaken as part of Intel India Grand Challenge 2016 Project MIRIAD: Many Incarnations of Screening of Radiology for High Throughput Disease Screening via Multiple Instance Reinforcement Learning with Adversarial Deep Neural Networks, sponsored by Intel Technology India Pvt. Ltd., Bangalore, India 

**Principal Investigators**
Dr Debdoot Sheet, Dr Nirmalya Ghosh
Department of Electrical Engineering, Indian Institute of Technology Kharagpur
Dr Ramanathan Sethuraman, Intel Technology India Pvt. Ltd.

## References

[1] Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. &quot;Densely connected convolutional networks.&quot; In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pp. 4700-4708. 2017.

[2] Tan, Mingxing, and Quoc V. Le. &quot;EfficientNet: Rethinking model scaling for convolutional neural networks.&quot; , ICML, pp. 6105-6114. 2019.

[3] Mitra, Arka, Arunava Chakravarty, Nirmalya Ghosh, Tandra Sarkar, Ramanathan Sethuraman, and Debdoot Sheet. &quot;A Systematic Search over Deep Convolutional Neural Network Architectures for Screening Chest Radiographs.&quot; _arXiv preprint arXiv:2004.11693_ (2020).

[4] Irvin, Jeremy, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund et al. &quot;Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison.&quot; In _Proceedings of the AAAI Conference on Artificial Intelligence_, vol. 33, pp. 590-597. 2019.