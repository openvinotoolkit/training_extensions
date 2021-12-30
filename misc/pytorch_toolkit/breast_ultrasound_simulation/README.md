# Multi-frequency Ultrasound Simulation

Ultrasound simulators are used by clinicians and by the scientific community of practitioners and educators. Ultrasound image formation depends on factors like frequency and size of the transducer, tissue characteristics, nonlinear attenuation, diffraction, and scattering due to the medium. Prior approaches try to incorporate various factors by modeling their physics and solving wave equations which leads to computationally expensive solutions. In this work, we propose a fast simulation approach using Convolutional Neural Network (CNN) to model the non-linearity of signal interaction with ultrasound image formation. The network is trained in an adversarial manner using a Turing test discriminator. The simulation process consists of two stages, Stage 0 is a pseudo B-mode simulator that provides an initial estimate of the speckle map, Stage 1 is a multi-linear separable CNN which further refines the speckle map to provide the final output. In order to ensure reproducibility, we have initialized the experiments with predefined random seeds.
We further derive the relationship between frequency of the transducer and resolution of the ultrasound image in order to be able to simulate and form images at frequencies other than at which the model is trained. This is achieved by interpolation between dilated kernels in a multi-linear separable CNN that is used to control the resolution of the generated image allowing for zero-shot adaptation.
Given a  CNN trained on a  2D ultrasound dataset acquired using a transducer of a particular frequency, we can simulate  ultrasound images corresponding to various transducer frequencies without the need for availability of additional training data at different frequencies. We have performed an  extensive validation of our approach using both real and simulated ultrasound images. Using a regression analysis we have concluded that there exists a linear relationship between the proposed kernel interpolation factor and the transducer frequency, thereby laying the foundation of a multi-frequency ultrasound simulator.

## Proposed Approach

<p align="center"><img src="./media/graphic.jpg" alt="drawing" width="500"/></p>

## Results

<p align="center"><img src="./media/cart.jpg" alt="drawing" width="500"/></p>

## Setup
### Prerequisites
 * python 3.7.3
 * torch                  1.10.0+cu102
 * torchvision            0.6.0+cu101
 * numpy                  1.19.2
 * opencv-python          4.5.1.48
 * Pillow                 8.2.0

```
sh init_env.sh
```
## Datasets
The network is trained on [Breast Ultrasound Images](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) dataset[1]. Data augmentation is applied as explain in [2,3,4] and images are processed by [stage 0](https://in.mathworks.com/matlabcentral/fileexchange/34199-pseudo-b-mode-ultrasound-image-simulator). We have directly provided the processed dataset for 2D simulation [here](http://miriad.digital-health.one/extra_data/data_bmi6.zip). Place the processed data as shown below.
```
+-- data
|   +-- stage0
    |   +-- benign (1)_f_00.png
    |   +-- benign (1)_f_01.png
                        .
                        .
|   +-- real_images
    |   +-- benign (1)_f_00.png
    |   +-- benign (1)_f_01.png
                        .
                        .
```

## **Model**

Download `.pth` checkpoint for alexnet with the following [link](http://miriad.digital-health.one/models/bmi6/model.zip).

Inference models will be made available in the [open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public) as well.


## Training
```
cd src
python train.py --name <EXP_NAME> --stage0_data <STAGE0_DATA_PATH> --realUS_data <REAL_IMAGE_DATA_PATH>
```

Intermediate models will be stored in ```checkpoints``` directory and logs in ```logs``` directory.

## Testing

```
cd src
python infer.py --name <EXP_NAME> --model_name <NAME OF MODEL TO RESTORE> --infer_data BUS  --dilation_factor 0 --stage0_data <STAGE0_DATA_PATH> --realUS_data <REAL_IMAGE_DATA_PATH>
```
The results will be stored in ```infer_results```. In order to simulate images corresponding to different frequency set ```dilation_factor```  between 0 and 1.

A demo code has also been provided [here](http://miriad.digital-health.one/demo/bmi6/model.zip) which does not require any data to be downloaded. The expected output is shown below.

```
python demo.py --dilation_factor 0
```
<p align="center"><img src="./media/output.png" alt="drawing" width="200"/></p>


## Metric

In order to measure the change in resolution of image we used Fourier transform. The experiments are conducted for the simulated 2D IVUS images. First we calculate N point DFT of the image and it is filtered using band pass filter H. The power(P) of the resulting signal after normalizing it, is used as a metric. For further details please refer to the paper[1]. The formulation for the same is given below.
<p align="center"><img src=https://user-images.githubusercontent.com/22493755/120058143-75c56500-c066-11eb-8153-ac8d93ea83c8.png alt="fourier transform" width="350"/></p>
<p align="center"><img src=https://user-images.githubusercontent.com/22493755/123139050-6540b880-d473-11eb-94cd-6d742240de1c.png alt="fourier transform" width="350"/></p>


 where u is the mean of the value of the image and is used to normalize P. The normalized power captures the power of signal in high frequency band, hence it should decrease with increasing dilation factor. The value of P for different dilation factor are given below
 
 <div align="center">
    
 Dilation factor| 0           |    0.25       |     0.50       |     0.75       |       1       |
| :---          |    :----:   |      :----:   |    :----:      |      :----:    |         :----:   |
| Normalized Power x1000| 2694.16       | 2304.32    |    2105.62     |      2012.28  |       1755.84    |


</div>

In order to reproduce the above values generate the simulated images using ```infer.py``` and then use ```metric.py``` to compute the metric. An example for dilation factor 0 is given below.

```
cd src
python infer.py --name dil_0 --model_name <NAME OF MODEL TO RESTORE> --infer_data BUS  --dilation_factor 0 --stage0_data <STAGE0_DATA_PATH> --realUS_data <REAL_IMAGE_DATA_PATH>
python metric.py --dir ../infer_results/dil_0 
```

## Acknowledgement

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

[Vidit Goel](https://vidit98.github.io/)</br>
Department of Electrical Engineering,</br>
Indian Institute of Technology Kharagpur</br>
email: vidit.goel9816@gmail.com 

[Rakshith Sathish](https://github.com/Rakshith2597)</br>
Advanced Technology Development Center,</br>
Indian Institute of Technology Kharagpur</br>
email: rakshith.sathish@kgpian.iitkgp.ac.in</br>
Github username: Rakshith2597

[Raj Krishan Ghosh](https://github.com/Raj-Krishan-Ghosh)</br>
Centre of Excellence in Artificial Intelligence,</br>
Indian Institute of Technology Kharagpur</br>
email: rajkrishanghosh@gmail.com 


## Publications
 1. Al-Dhabyani, Walid, Mohammed Gomaa, Hussien Khaled, and Aly Fahmy. "Dataset of breast ultrasound images." Data in brief 28 (2020): 104863. [[link](https://www.sciencedirect.com/science/article/pii/S2352340919312181)]
 2. Vidit Goel, Harsh Maheshwari, Raj Krishan Ghosh, Anand Mooga, Ramanathan Sethuraman, Debdoot Sheet "Fast Simulation of Ultrasound Images using Multilinear Separable Deep Convolution Neural Network with Kernel Dilation" submitted in IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control
 3. Anand Mooga, Ramanathan Sethuraman, Debdoot Sheet "Zero-Shot Adaptation to Simulate 3D Ultrasound Volume by Learning a Multilinear Separable 2D Convolutional Neural Network" in 2020 IEEE 17th International Symposium on Biomedical Imaging [[link](https://ieeexplore.ieee.org/abstract/document/9098479/)]
 4. Francis Tom, Debdoot Sheet "Simulating Patho-realistic Ultrasound Images using Deep Generative Networks with Adversarial Learning" in 2018 IEEE 15th International Symposium on Biomedical Imaging [[link](https://arxiv.org/abs/1712.07881)]