# Deep Neural Hashing for Medical Image Retrieval

<div id="abs">

The rapid digitalization of the healthcare sector has resulted in a large corpus of various forms of data on different parts of the human body, mostly acquired using X-rays, Magnetic Resonance (MR), Computed Tomography (CT), and Ultrasound. This large repository of data has the potential to transform the way treatment is administered by clinicians based on evidence. Evidence-based medicine (EBM) [[1](#EBM)] needs to integrate better clinical expertise, patient values, and the best research evidence with the help of modern technologies. Content-based medical image retrieval (CBMIR) is the subset of content-based image retrieval (CBIR) [[2](#CBIR)], which specially focuses on medical image retrieval. The simple approach of  CBIR is to use a query image to find the previously stored images in databases that are the closest neighbors. CBIR technology appears to be very limited in regular clinical practice or biomedical research. Clinicians can prescribe the same treatment to all the patients which reports are retrieved from the medical database using a CBIR system that are similar to query image. DNHN is an efficient and automatic technique for CBIR by automatically and visually retrieving semantically similar images to the
query being studied from an existing image database which can helps doctors and clinicians make for final diagnosis.

This code is to perform to retrieves medical images with semantic similarity of organs and their associated pathology. In this code, we performed experiments on LeNet-121[[3](#LeNet)] CNN architecture for this task. This work’s methods contain three steps during the training procedure to generate authentic hash codes from images. First, minimize the classifying error of input
images. Second, maximize the ability of the discriminator to identify two similar images, and third, minimize and maximize the Hamming distance with cosine similarity for two similar and two dissimilar images respectively, using Cauchy cross-entropy loss [[3](#cauchy)]. This  model is trained using a subset of the publicly available benchmark MedMNIST v2. [[5]](#medmnist) [(link)](https://zenodo.org/record/6496656)) [(license)](change the licence link). We have evaluted our performence using two metrices a  mean average precision (mAP) and normalized discounted cumulative gain (nDCG).
</div>


Few example images from the dataset
<table >
<tr>
<td align='center'> Breast,  class 1(normal or benign)</td>
<td align='center'> Chest, class 0( atelectasis)</td>
<td align='center'> Retina,  class 2 (drusen)</td>
<td align='center'> Tissue,  class 7 (    Thick Ascending Limb​)</td>
</tr>
<tr>
<td align='center'><img src="./media/Breast_1.png" width="250" height="250"></td>
<td align='center'> <img src="./media/chest_0.png" width="250" height="250"></td>
<td align='center'> <img src="./media/oct_2.png" width="250" height="250"> </td>
<td align='center'> <img src="./media/tissue_7.png" width="250" height="250"> </td>
</tr>
</table>

Minimizing classification loss:
<img src = "./media/classifier.png" width=650>


Cauchy entropy loss:
<img src = "./media/cauchyloss.png" width=650>

Discriminator loss:
<img src = "./media/discrim.png" width=650>

## Network Architecture:

We have used a LeNet-5 as the base architecture.


## Hyperparameters

| Variable | Value |
| -- | -- |
| Epoch | 100|
| lr | 0.001|
| α1 | 0.5 |
| α2 | 0.5 |
| β  | 0.6 |
| γ  |  1   | 

## Results

Score of mean avergae precision for hash code length 48 bit.
| mAP@p |  Score  |
|--|--|
| mAP@10 | 0.7586 |
| mAP@100 | 0.8075 |
| mAP@1000 | 0.6307 |


Score of normalized discounted cumulative gain for hash code length 48 bit.
| mAP@p |  Score  |
|--|--|
| nDCG@10 | 0.5148 |
| nDCG@100 | 0.5793 |
| nDCG@1000 | 0.6151 | 

Note: The newtork was trained for 100 epochs. 

## **Model**

Download `.pkl` checkpoint for densenet with the following [link](http://kliv.iitkgp.ac.in/projects/miriad/model_weights/bmi12/densenet.zip).


## **Setup**

### Prerequisites

* Ubuntu\* 16.04
* Python\* 3.6
* NVidia\* GPU for training
* 16GB RAM for inference

## **Train**

1. Download the [MedMNISTv2 Dataset](https://zenodo.org/record/6496656)
2. Create the directory tree
3. Prepare the training dataset and store
4. Run the training script

## **Code and Directory Organisation**

```

openvino_Hashing/
  src
    utils
      dataset
        train
        val
        gallery
        query
      model_weights
        encoder-100.pkl
        encoder.bin
        encoder.onnx
        encoder.xml
      dataloader.py
      exporter.py
      get_config.py
      network.py
      downloader.py
      vectorHandle.py
    export.py
    inference.py
    train.py
  test
    test_export.py
    test_inference.py
    test_train.py
  init_venv.sh
  README.md
  requirements.txt
  setup.py
```

Note: The directory `model_weights` is downloaded and created using the `downloader.py` script

## **Code Structure**


1. `train.py` in src directory contains the code for training the model.
2. `inference.py` in src directory contains the code for evaluating the model with test set.
3. `export.py` in src directory generating the ONNX and Openvino IR of the trained model.
4. All dependencies are provided in **utils** folder.

5. **tests** directory contains  unit tests.
6. **config** directory contains model configs.
7. `encoder-100.pkl` in model_weights directory generates hash code for an input images.


## **Creating the Dataset Directory Tree**

The ChestMNIST, BreastMNIST, RetinaMNIST, TissueMNIST from MedMNIST benchmark is used for the experiments. Each sample/item in the dataset is processed into `Organ_disease_count.npy` format, where organ comes from dataset it belonged to and the disease comes from labels provided in the datasets. The dataset is then divided into four subfolder train, val, gallery, and query with ratio 6:1:2:1. Dataset tree is given below.

```
+-- dataset
|   +-- train
          +--Breast
          +--Chest
          +--Retina
          +--Tissue
        test
          +--Breast
          +--Chest
          +--Retina
          +--Tissue
        gallery
|       query

```
Download the dataset from [link](https://zenodo.org/record/6496656). 

### Run Training

Run the `train.py` script:
```
python train.py \
  --alpha 1\
  --alpha 2\
  --beta \
  --gamma \
  --checkpoint \
  --bs \
  --lr test_run \
  --dpath \
  --spath \
  --epochs \
  --zSize \
  --clscount \
```

## How to Perform Prediction

Ensure that the test directory contains a series of  samples in the numpy format with the `.npy` extension.

### Run Inference
```
python inference.py \
  --checkpoint \
  --dpath \
  --modelpath \
  --zSize \

```

### Run Tests

Necessary unit tests are `test_train.py`, `test_inference.py`, `test_export.py` and have been provided in the test directory. The sample/toy dataset is downloaded and stored in `src/utils/dataset` folder.

## **Acknowledgement**

```
This work is undertaken as part of Intel India Grand Challenge 2016 Project MIRIAD: Many Incarnations of Screening of Radiology for High Throughput Disease Screening via Multiple Instance Reinforcement Learning with Adversarial Deep Neural Networks, sponsored by Intel Technology India Pvt. Ltd., Bangalore, India.
```

**Principal Investigators**

<a href="https://www.linkedin.com/in/debdoot/">Dr Debdoot Sheet</a>
Department of Electrical Engineering,</br>
Indian Institute of Technology Kharagpur</br>
email: debdoot@ee.iitkgp.ac.in, 

<a href="https://www.linkedin.com/in/ramanathan-sethuraman-27a12aba/">Dr Ramanathan Sethuraman</a>,</br>
Intel Technology India Pvt. Ltd.</br>
email: ramanathan.sethuraman@intel.com

**Contributor**

The codes/model was contributed to the OpenVINO project by

<a href="https://www.linkedin.com/in/asimmanna17/">Asim Manna</a>, </br>
Centre of Excellence in Artificial Intelligence, </br>
Indian Institute of Technology Kharagpur </br>
email: asimmanna17@kgpian.iitkgp.ac.in </br> 


<a href="https://github.com/Rakshith2597"> Rakshith Sathish</a>,</br>
Advanced Technology Development Center,</br>
Indian Institute of Technology Kharagpur</br>
email: rakshith.sathish@kgpian.iitkgp.ac.in</br>
Github username: Rakshith2597

## **References**

<div id="EBM">
<a href="#abs">[1]</a> R. B. Haynes, D. L. Sackett, W. S. Richardson, W. Rosenberg, G. R.
Langley, Evidence-based medicine: How to practice & teach ebm,
Canadian Medical Association. Journal 157 (6) (1997) 788. </a> 
</div>

<div id="CBIR">
<a href="#abs">[2]</a> F. Long, H. Zhang, D. D. Feng, Fundamentals of content-based image
retrieval, in: Multimedia information retrieval and management,
Springer, 2003, pp. 1–26 </a>

</div>
<div id="LeNet">
<a href="#abs">[3]</a>  A. El-Sawy, E.-B. Hazem, M. Loey, Cnn for handwritten arabic digits recognition based on lenet-5, in: International conference on advanced
intelligent systems and informatics, Springer, 2016, pp. 566–575. </a>

</div>
<div id="cauchy">
<a href="#results">[4]</a>  Y. Cao, M. Long, B. Liu, J. Wang, Deep cauchy hashing for hamming space retrieval, in: Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition, 2018, pp. 1229–1237 </a>
</div>

</div>
<div id="medmnist">
<a href="#results">[5]</a> 
Yang, Jiancheng, et al. "Medmnist v2: A large-scale lightweight benchmark for 2d and 3d biomedical image classification." arXiv preprint arXiv:2110.14795 (2021).</a>
</div>