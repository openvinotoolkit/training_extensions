**Effecient Net based Architecture Search for Chest Radiograph Screening**

In this code, we performed experiments to fine-tune the DenseNet-121 [1] CNN architecture which was found to have a good performance on this task [3]. The same CNN architecture was retrained using a publicly available dataset provided by RSNA to detect pneumonia.

A systematic search was performed over a set of CNN architectures by scaling the width and depth of DenseNet using the EfficientNet method [2]. The optimal architecture was achieved with the scaling parameters of. The details of our experiments based on EfficientNet is summarized in the slides available at: [https://drive.google.com/file/d/1BQsCIeYFHZlMGmUS5GYIYQmgiMqPwPc4/view?usp=sharing](https://drive.google.com/file/d/1BQsCIeYFHZlMGmUS5GYIYQmgiMqPwPc4/view?usp=sharing)

### Model
Download checkpoint with the following [link]() 
The OpenvinoIR can be found at [link]()


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
	|	+-- train_data
	|	+-- test_data
```
Model weights will be stored in your current/parent directory.
```
### Prepare the Training Dataset

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

## **References**

[1] Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. &quot;Densely connected convolutional networks.&quot; In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pp. 4700-4708. 2017.

[2] Tan, Mingxing, and Quoc V. Le. &quot;EfficientNet: Rethinking model scaling for convolutional neural networks.&quot; , ICML, pp. 6105-6114. 2019.

[3] Mitra, Arka, Arunava Chakravarty, Nirmalya Ghosh, Tandra Sarkar, Ramanathan Sethuraman, and Debdoot Sheet. &quot;A Systematic Search over Deep Convolutional Neural Network Architectures for Screening Chest Radiographs.&quot; _arXiv preprint arXiv:2004.11693_ (2020).

[4] Irvin, Jeremy, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund et al. &quot;Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison.&quot; In _Proceedings of the AAAI Conference on Artificial Intelligence_, vol. 33, pp. 590-597. 2019.