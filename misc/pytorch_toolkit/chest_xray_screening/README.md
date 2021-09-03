# Deep Neural Network for Chest X-Ray Screening

<div id="abs">

Chest radiographs are primarily employed for the screening of pulmonary and cardio-thoracic conditions. Being undertaken at primary healthcare centers, they require the presence of an on-premise reporting Radiologist, which is a challenge in low and middle income countries. This has inspired the development of machine learning based automation of the screening process. While recent efforts demonstrate a performance benchmark using an ensemble of deep convolutional neural networks (CNN), our systematic search over multiple standard CNN architectures identified single candidate CNN models whose classification performances were found to be at par with ensembles.

This code is to perform architecture search for models trained to detect multiple comorbid chest diseases in chest X-rays. In this code, we performed experiments to fine-tune the DenseNet-121[[1](#densenet)] CNN architecture for this task [[2](#embc)]. The method in this repository differs from the paper in a few aspects; In the paper, the authors classify an X-ray image into one or more of the 14 classes following a multi-hot encoding on account of co-morbidity of diseases, while in this repository we present the approach to classify a Chest X-ray image into any of the applicable 3 classes. This modified DenseNet-121 is retrained using the publicly available 2018 RSNA Pneumonia Detection Challenge dataset[[5]](#rsnadataset) [(link)](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pneumonia-detection-challenge-2018) [(license)](https://www.rsna.org/-/media/Files/RSNA/Education/AI-resources-and-training/AI-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.ashx?la=en&hash=FF7A635F6DFFAD31A30C8715DFA3B8FC21131543).
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

A systematic search was performed over a set of CNN architectures by scaling the width and depth of the standard DenseNet using the EfficientNet approach[[3](#efficientnet)]. The details of our experiments based on EfficientNet is summarized in the image below.

<img src = "./media/efficientnet.png" width=650>

## Network Architecture:

We have used a DenseNet-121 as the base architecture.

![](https://lh6.googleusercontent.com/ziwy55LfUqzzErhcy0Cw418LbeCWpfH_liD3dXNrae8yVnV91rnCWsokLUVO0NVUuUeNHIG6bnkV3J7jNNT5U6DDr2Y78Z60NW-2ACUEuY53k6B7C6x1Q9HFrJ-1yJZNM1vyMPdg)

  

## Results

AUROC scores for each class and Mean AUROC score
| Class | AUROC Score  |
|--|--|
| Lung Opacity | 0.7586 |
| Normal | 0.80753 |
| No Lung Opacity/ Not Normal | 0.63076 |

**Mean AUROC score**: 0.7323


Note: The newtork was trained for 25 epochs. 

AUROC Score for the same network, when trained and evaluated using the CheXpert dataset provided by Stanford University [[4](#chexpert)], is given below.

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

The network when trained and evaluated using the CheXpert dataset with same alpha, beta, and phi values (given below) was able to classify with a **Mean AUROC score:** 0.7877
| Variable | Value |
| -- | -- |
| α | 1.833 |
| β | 1.044 |
| ϕ | -0.10 | 


## **Model**
Download checkpoint for densenet, ONNX checkpoint, and OpenVINO IR with the following [link](https://drive.google.com/drive/folders/1Qa88btQeeBszcCtQOYp8X7C051fdOOm9?usp=sharing).

Download checkpoint for optimised model, ONNX checkpoint, and OpenVINO IR with the following [link](https://drive.google.com/drive/folders/1qJGDHDXSrJ8WyhXOnJB49MJHBtfzQQcB?usp=sharing)


## **Demo**
An example for using the ONNX models for inference can be found [here](https://drive.google.com/drive/folders/1EdgO3ZLZejoPojGXZxElOYnSOUM7zHkn?usp=sharing).

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
4. Run the training script

## **Code and Directory Organisation**

```
chest_xray_screening/
	chest_xray_screening/
      utils/
        data_prep.py
        downloader.py
        download_weights.py
        exporter.py
        generate.py
        get_config.py
        model.py
        score.py
      export.py
      inference.py
      train.py
	configs/
      densenet121_config.json
      densenet121eff_config.json
      gdrive_config.json
	media/
	tests/
      test_export.py
      test_inference.py
      test_train.py
	init_venv.sh
	README.md
	requirements.txt
	setup.py
```

## **Code Structure**

1. `train.py` in chest_xray_screening directory contains the code for training the model.
2. `inference.py` in chest_xray_screening directory contains the code for evaluating the model with test set.
3. `export.py` in chest_xray_screening directory generating the ONNX and Openvino IR of the trained model.
4. All dependencies are provided in **utils** folder.

5. **tests** directory contains  unittests.
6. **config** directory contains model configs.

## **Creating the Dataset Directory Tree**

Create a folder in the root directory and name it 'dataset'. Create a directoy tree as shown below.

```
+-- dataset
|   +-- original
|   +-- processed_data
```
Download the dataset from [link](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) and place it in the dataset/original directory.

### Prepare the Training Dataset

```
python utils/data_prep.py --dpath absolute/path/to/dataset/directory 
```

On completion processed data will be stored in the 'processed_data' subfolder. Download the dataset split from [link](https://drive.google.com/drive/folders/17owTvg51wo2MORecTLiEopOf3D3YZV_k?usp=sharing) and place the numpy files in dataset/data_splits directory.

### Run Training

Run the `train.py` script:
```
python train.py \
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

### Run Inference
```
python inference.py \
  --alpha \
  --beta \
  --phi \
  --checkpoint \
  --bs \
  --imgpath \

```

### Run Tests

Necessary unit tests have been provided in the tests directory. The sample/toy dataset to be used in the tests is provided in the folder 'data' (at root of this directory). It can also be downloaded from [here](https://drive.google.com/drive/folders/1aAZvdST531WUQIbKxSedifv8WfM3Wbqm?usp=sharing).

## **Acknowledgement**

This work is undertaken as part of Intel India Grand Challenge 2016 Project MIRIAD: Many Incarnations of Screening of Radiology for High Throughput Disease Screening via Multiple Instance Reinforcement Learning with Adversarial Deep Neural Networks, sponsored by Intel Technology India Pvt. Ltd., Bangalore, India.

## **References**

<div id="densenet">
<a href="#abs">[1]</a> Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition_, pp. 4700-4708. 2017. <a href="https://arxiv.org/pdf/1608.06993.pdf"> (link) </a> 
</div>

<div id="embc">
<a href="#abs">[2]</a> A. Mitra, A. Chakravarty, N. Ghosh, T. Sarkar, R. Sethuraman and D. Sheet, "A Systematic Search over Deep Convolutional Neural Network Architectures for Screening Chest Radiographs," 2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), Montreal, QC, Canada, 2020, pp. 1225-1228, doi: 10.1109/EMBC44109.2020.9175246. <a href="https://ieeexplore.ieee.org/document/9175246"> (link) </a>

</div>
<div id="efficientnet">
<a href="#abs">[3]</a>  Tan, Mingxing, and Quoc V. Le. &quot;EfficientNet: Rethinking model scaling for convolutional neural networks.&quot; , ICML, pp. 6105-6114. 2019. <a href="http://proceedings.mlr.press/v97/tan19a/tan19a.pdf"> (link) </a>

</div>
<div id="chexpert">
<a href="#results">[4]</a>  Irvin, Jeremy, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund et al. &quot;Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison.&quot; In _Proceedings of the AAAI Conference on Artificial Intelligence_, vol. 33, pp. 590-597. 2019. <a href="https://arxiv.org/abs/1901.07031"> (link) </a>
</div>
<div id="rsnadataset">
<a href=#>[5] </a>George Shih , Carol C. Wu, Safwan S. Halabi, Marc D. Kohli, Luciano M. Prevedello, 
Tessa S. Cook, Arjun Sharma, Judith K. Amorosa, Veronica Arteaga, Maya GalperinAizenberg, Ritu R. Gill, Myrna C.B. Godoy, Stephen Hobbs, Jean Jeudy, Archana 
Laroia, Palmi N. Shah, Dharshan Vummidi, Kavitha Yaddanapudi, Anouk Stein, 
Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert 
Annotations of Possible Pneumonia, Radiology: AI, January 30, 2019, https://doi.org/10.1148/ryai.2019180041
</dice>
