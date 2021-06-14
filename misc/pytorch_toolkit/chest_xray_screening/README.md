# Deep Neural Network for Chest X-Ray Screening 

<div id="abs">

Chest radiographs are primarily employed for the screening of pulmonary and cardio-thoracic conditions. Being undertaken at primary healthcare centers, they require the presence of an on-premise reporting Radiologist, which is a challenge in low and middle income countries. This has inspired the development of machine learning based automation of the screening process. While recent efforts demonstrate a performance benchmark using an ensemble of deep convolutional neural networks (CNN), our systematic search over multiple standard CNN architectures identified single candidate CNN models whose classification performances were found to be at par with ensembles.

This code is to perform architecture search for models trained to detect multiple comorbid chest diseases in chest X-rays. In this code, we performed experiments to fine-tune the DenseNet-121[[1](#densenet)] CNN architecture for this task [[2](#embc)]. The method in this repository differs from the paper in a few aspects; In the paper, the authors classify an X-ray image into one or more of the 14 classes following a multi-hot encoding on account of co-morbidity of diseases, while in this repository we present the approach to classify a Chest X-ray image into any of the applicable 3 classes. This modified DenseNet-121 is retrained using the publicly available 2018 RSNA Pneumonia Detection Challenge dataset ([link](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)).
</div>


Few example images from the dataset
<table >
<tr>
<td align='center'> Class 0 (Lung Opacity)</td>
<td align='center'> Class 1 (Normal)</td>
<td align='center'> Class 2 (No Lung Opacity/ Not Normal)</td>
</tr>
<tr>
<td align='center'><img src="./media/class0.jpg" width="250" height="200"></td>
<td align='center'> <img src="./media/class1.jpg" width="250" height="200"></td>
<td align='center'> <img src="./media/class2.jpg" width="250" height="200"> </td>
</tr>
</table>

A systematic search was performed over a set of CNN architectures by scaling the width and depth of the standard DenseNet using the EfficientNet approach[[3](efficientnet)]. The details of our experiments based on EfficientNet is summarized in the image below.

<img src = "./media/efficientnet.png" width=650>

## Network Architecture:

We have used a DenseNet-121 as the base architecture.

![](https://lh6.googleusercontent.com/ziwy55LfUqzzErhcy0Cw418LbeCWpfH_liD3dXNrae8yVnV91rnCWsokLUVO0NVUuUeNHIG6bnkV3J7jNNT5U6DDr2Y78Z60NW-2ACUEuY53k6B7C6x1Q9HFrJ-1yJZNM1vyMPdg)

  

## Results

AUROC scores for each class and Mean AUROC score is reported for Pneumonia detection.
| Class | AUROC Score  |
|--|--|
| Lung Opacity | 0.7586 |
| Normal | 0.80753 |
| No Lung Opacity/ Not Normal | 0.63076 |

**Mean AUROC score**: 0.7323


Note: The newtork was trained for 25 epochs. 

AUROC Score for the same network, when trained and evaluated using the CheXpert dataset provided by Stanford University [[3](#chexpert)], is given below.

| Class | AUROC Score  |
|--|--|
| Lung Opacity | 0.916 |
| Atelectasis | 0.807 |
| Enlarged Cardiomediastinum | 0.714 |
| Cardiomegaly | 0.800 |
| Pleural Effusion | 0.924 |
| Edema | 0.917 |
| Consolidation | 0.924 |
| Pneumonia | 0.723 |
| Pneumothorax | 0.777 |
| Lung lesion | 0.584 |
| Pleural Other | 0.918 |

**Mean AUROC score**: 0.87

AUROC scores of the efficient-net based model for each class and Mean AUROC score is reported for Pneumonia detection.
| Class | AUROC Score  |
|--|--|
| Lung Opacity | 0.578 |
| Normal | 0.569 |
| No Lung Opacity/ Not Normal | 0.511 |

**Mean AUROC score:** 0.5531
The network when trained and evaluated using the CheXpert dataset with same alpha, beta, and phi values (given below) was able to classify with a **Mean AUROC score of 0.7877**
| Variable | Value |
| -- | -- |
| α | 1.833 |
| β | 1.044 |
| ϕ | -0.10 | 


## **Model**
Download checkpoint with the following [link]()

The OpenVINO IR can be found [here]().

Download checkpoint for optimised model with the following [link](https://drive.google.com/file/d/1q9OYgK1y-eWeBljTH5G4NlJUuF4sB6bU/view?usp=sharing)

The OpenVINO IR can be found [here](https://drive.google.com/file/d/1SoSWLbitdh0AfpUeUyI9-0vr_-dUvuHq/view?usp=sharing).

## **Demo**
An example for using the ONNX models for inference can be found [here]().

An example for using the ONNX model of optimised network for inference can be found [here](https://drive.google.com/drive/folders/1cUAbfbRvbSmb4fXiQwHyUaNch8w4ExXl?usp=sharing).

## **Setup**

### Prerequisites

* Ubuntu\* 16.04
* Python\* 3.6
* NVidia\* GPU for training
* 16GB RAM for inference

## **Train**

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
## **Acknowledgement**
This work is undertaken as part of Intel India Grand Challenge 2016 Project MIRIAD: Many Incarnations of Screening of Radiology for High Throughput Disease Screening via Multiple Instance Reinforcement Learning with Adversarial Deep Neural Networks, sponsored by Intel Technology India Pvt. Ltd., Bangalore, India.

**Contributor**

The codes/model was contributed to the OpenVINO project by

<a href="https://www.linkedin.com/in/rakshith-sathish/">Rakshith Sathish</a>, </br>
Advanced Technology Development Centre, </br>
Indian Institute of Technology Kharagpur</br>
email: rakshith.sathish@kgpian.iitkgp.ac.in</br>
Github username: Rakshith2597


**Principal Investigators**

<a href="https://www.linkedin.com/in/debdoot/">Dr Debdoot Sheet</a>, <a href="http://www.iitkgp.ac.in/department/EE/faculty/ee-nirmalya">Dr Nirmalya Ghosh</a>,</br>
Department of Electrical Engineering,</br>
Indian Institute of Technology Kharagpur</br>
email: debdoot@ee.iitkgp.ac.in, nirmalya@ee.iitkgp.ac.in

<a href="https://www.linkedin.com/in/ramanathan-sethuraman-27a12aba/">Dr Ramanathan Sethuraman</a>,</br>
Intel Technology India Pvt. Ltd.
email: ramanathan.sethuraman@intel.com

## **References**

<div id="densenet">
<a href="#abs">[1]</a> Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition_, pp. 4700-4708. 2017. <a href="https://arxiv.org/pdf/1608.06993.pdf"> (link) </a> 
</div>

<div id="embc">
<a href="#abs">[2]</a> A. Mitra, A. Chakravarty, N. Ghosh, T. Sarkar, R. Sethuraman and D. Sheet, "A Systematic Search over Deep Convolutional Neural Network Architectures for Screening Chest Radiographs," 2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), Montreal, QC, Canada, 2020, pp. 1225-1228, doi: 10.1109/EMBC44109.2020.9175246. <a href="https://ieeexplore.ieee.org/document/9175246"> (link) </a>

</div>

<div id="chexpert">
<a href="#results">[3]</a>  Irvin, Jeremy, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund et al. &quot;Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison.&quot; In _Proceedings of the AAAI Conference on Artificial Intelligence_, vol. 33, pp. 590-597. 2019. <a href="https://arxiv.org/abs/1901.07031"> (link) </a>
</div>