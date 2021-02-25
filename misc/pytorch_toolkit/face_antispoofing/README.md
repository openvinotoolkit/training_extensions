# Lightweight Face Anti Spoofing
Towards the solving anti-spoofing problem on RGB only data.
## Introduction
This repository contains a training and evaluation pipeline with different regularization methods for face anti-spoofing network. There are a few models available for training purposes, based on MobileNetv2 (MN2) and MobileNetv3 (MN3). Project supports natively three datasets: [CelebA Spoof](https://github.com/Davidzhangyuanhan/CelebA-Spoof), [LCC FASD](https://csit.am/2019/proceedings/PRIP/PRIP3.pdf), [CASIA-SURF CeFA](https://arxiv.org/pdf/2003.05136.pdf). Also, you may want to train or validate with your own data. Final model based on MN3 trained on the CelebA Spoof dataset. The model has 3.72 times fewer parameters and 24.3 times fewer GFlops than AENET from the original paper, at the same time MN3 better generalizes on cross-domain. The code contains a demo that you can launch in real-time with your webcam or on the provided video. You can check out the short video on how it works on the [goole drive](https://drive.google.com/drive/u/0/folders/1A6wa3AlrdjyNPkXT81knIzXxR7SAYm1q). Also, the code supports conversion to the ONNX format.
You can follow the links to the configuration files with smaller models to train them as-is and obtain metrics below.

| model name | dataset | AUC | EER% | APCER% | BPCER% | ACER% | MParam | GFlops | Link to snapshot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MN3_large |CelebA-Spoof| 0.998 | 2.26 | 0.69 | 6.92 | 3.8 | 3.02 |  0.15 | [snapshot](https://drive.google.com/drive/u/0/folders/1A6wa3AlrdjyNPkXT81knIzXxR7SAYm1q) |
| AENET | CelebA-Spoof | 0.999 | 1.12 | 0.23 | 6.27 | 3.25 | 11.22 | 3.64  | [link to repo](https://github.com/Davidzhangyuanhan/CelebA-Spoof) |
| MN3_large_075 |CelebA-Spoof| 0.997 | 2.79 | 1.21 | 5.44 | 3.32 | 1.86 | 0.1 | [config](./configs/config_large_075.py) |
| MN3_small |CelebA-Spoof| 0.994 | 3.84 | 1.47 | 8.63 | 5.05 | 1.0 | 0.04 | [config](./configs/config_small.py) |
| MN3_small_075 |CelebA-Spoof| 0.991 | 4.74 | 1.62 | 10.55 | 6.09 | 0.6 | 0.03 | [config](./configs/config_small_075.py) |
| MN3_large | LCC_FASD | 0.921 | 16.13 | 17.26 | 15.4 | 16.33 | 3.02 | 0.15 | [snapshot](https://drive.google.com/drive/u/0/folders/1A6wa3AlrdjyNPkXT81knIzXxR7SAYm1q) |
| AENET | LCC_FASD | 0.868 | 20.91 | 12.52 | 32.7 | 22.61 | 11.22 | 3.64 | [link to repo](https://github.com/Davidzhangyuanhan/CelebA-Spoof) |
| MN3_large_075 | LCC_FASD | 0.892 | 19.42 | 28.34 | 12.18 | 20.26 | 1.86 | 0.1 | [config](./configs/config_large_075.py) |
| MN3_small | LCC_FASD | 0.889 | 18.7 | 14.79 | 24.6 | 19.69 | 1.0 | 0.04 | [config](./configs/config_small.py) |
| MN3_small_075 | LCC_FASD | 0.879 | 21.07 | 22.77 | 19.3 | 21.04 | 0.6 | 0.03 | [config](./configs/config_small_075.py) |

## Setup
### Prerequisites

* Python 3.6.9
* OpenVINO™ 2020 R3 (or newer) with Python API

### Installation

1. Create a virtual environment:
```bash
bash init_venv.sh
```

2. Activate the virtual environment:
```bash
. venv/bin/activate
```
### Data Preparation
For training or evaluating on the CelebA Spoof dataset you need to download the dataset (you can do it from the [official repository](https://github.com/Davidzhangyuanhan/CelebA-Spoof)) and then run the following script being located in the root folder of the project:
```bash
cd /data_preparation/
python prepare_celeba_json.py
```
To train on or evaluate the LCC FASD dataset you need to download it (link is available in the [original paper](https://csit.am/2019/proceedings/PRIP/PRIP3.pdf)). Then you need to get the OpenVINO™ face detector model. You can use [model downloader](https://docs.openvinotoolkit.org/latest/omz_tools_downloader_README.html) to do that. The name of the model that you are looking for is `face-detection-0100.xml`, activate OpenVINO™ environment, and run the following script:
```bash
python prepare_LCC_FASD.py --fd_model <path to `.xml` face detector model> --root_dir <path to root dir of LCC_FASD>
```
This script will cut faces tighter than it is in the original dataset and get rid of some garbage crops. For running this script you need to activate OpenVINO™ environment. Refer to the official documentation.

You can use the LCC FASD without doing this at all, but it seems to enhance performance, so I recommend doing this.
Note that the new folder will be created and named as `<old name>cropped`. So to train or test the model with cropped data, please, set path to that new folder, which will be located in the same directory as the script.

To train on or evaluate the CASIA CEFA you just need to download it. The reader for this dataset supports not only RGB modality but the depth and IR too. Nevertheless, it's not the purpose of this project.

If you want to use your own data, the next steps should be done:
1) Prepare the reader for your dataset.
2) Import reader object to datasets/database.py file. Substitute `do_nothing` with your object in `external_reader=do_nothing` (35 line).
3) In config, write any kwargs for train, validation, test protocol. If you do not have test data, you can just add the same parameters as for validation.
Example: `external = dict(train=dict(data_root='...', mode='train', whatever=...), val=dict(data_root='...', mode='val', whatever=...), test=dict(...))`

Now you are ready to launch the training process!

### Configuration file
The script for training and inference uses a configuration file. This is [default one](./configs/config.py). You need to specify paths to datasets. The training pipeline supports the following methods, which you can switch on and tune hyperparameters while training:
* **dataset** - this is an indicator which dataset you will be using during training. Available options are 'celeba-spoof', 'LCC_FASD', 'Casia', 'multi_dataset', 'external'
* **multi_task_learning** - specify whether or not to train with multitasking loss. **It is available for the CelebA-Spoof dataset only!**
* **evaluation** - it is the flag to perform the assessment at the end of training and write metrics to a file
* **test_dataset** - this is an indicator on which dataset you want to test. Options are the same as for dataset parameter
* **external** - parameters for constructing external dataset reader. See Data Preparation section.
* **img_norm_cfg** - parameters for data normalization
* **scheduler** - scheduler for dropping learning rate
* **data.sampler** - if it is true, then will be generated weights for `WeightedRandomSampler` object to uniform distribution of two classes
* **resize** - resize of the image
* **checkpoint** - the name of the checkpoint to save and the path to the experiment folder where checkpoint, tensorboard logs and eval metrics will be kept
* **loss** - there are available two possible losses: `amsoftmax` with `cos`, `arcos`, `cross_enropy` margins and `soft_triple` with different number of inner classes. For more details about this soft triple loss see in [paper](https://arxiv.org/pdf/1909.05235.pdf)
* **loss.amsoftmax.ratio**  - there is availability to use different m for different classes. The ratio is the weights on which provided `m` will be divided for a specific class. For example ratio = [1,2] means that m for the first class will equal to m, but for the second will equal to m/2
* **loss.amsoftmax.gamma** - if this constant differs from 0 then the focal loss will be switched on with the corresponding gamma
* **For soft triple loss**: `Cn` - number of classes, `K` - number of proxies for each class, `tau` - parameter for regularization number of proxies
* **model** - there are parameters concerning model. `pretrained` means that you want to train with the imagenet weights (you can download weights from [google drive](https://drive.google.com/drive/u/0/folders/1A6wa3AlrdjyNPkXT81knIzXxR7SAYm1q) and specify the path to it in the `imagenet weights` parameter. **model_type** - type of the model, 'Mobilenet3' and 'Mobilenet2' are available. **size** param means the size of the mobilenetv3, there are 'large' and 'small' options. Note that this will change mobilenev3 only. **embeding_dim** - the size of the embeding (vector of features after average pooling). **width_mult** - the width scaling parameter of the model. Note, that you will need the appropriate imagenet weights if you want to train your model with transfer learning. On google drive weights with 0.75, 1.0 values of this parameter are available
* **aug** - there are some advanced augmentations are available. You can specify `cutmix` or `mixup` and appropriate params for them. `alpha` and `beta` are used for choosing `lambda` from beta distribution, `aug_prob` response for the probability of applying augmentation on the image.
* **curves** - you can specify the name of the curves, then set option `--draw_graph` to `True` when evaluating with eval_protocol.py script
* **dropout** - `bernoulli` and `gaussian` dropouts are available with respective parameters
* **data_parallel** - you can train your network on several GPU
* **RSC** - representation self-challenging, applied before global average pooling. p, b - quantile and probability applying it on an image in the current batch
* **conv_cd** - this is the option to switch on central difference convolutions instead of vanilla one changing the value of theta from 0
* **test_steps** - if you set this parameter for some int number, the algorithm will execute that many iterations for one epoch and stop. This will help you to test all processes (train, val, test)

## Training
To start training create a config file based on the default one and run 'train.py':
```bash
python train.py --config <path to config>;
```
For additional parameters, you can refer to help (`--help`). For example, you can specify on which GPU you want to train your model. If for some reason you want to train on CPU, specify `--device` to `cpu`. The default device is `cuda 0`.

## Testing
To test your model set 'test_dataset' in config file to one of preferable dataset (available params: 'celeba-spoof', 'LCC_FASD', 'Casia'). Then run script:
```bash
python eval_protocol.py --config <path to config>;
```
The default device to do it is `cuda 0`.

## Convert a PyTorch Model to the OpenVINO™ Format
To convert the obtained model, run the following command:
```bash
python convert_model.py --config <path to config>; --model_path <path to where save the model>;
```
By default, the output model path is 'MobileNetv3.onnx'

Now you obtain '.onnx' format. Then go to <OPENVINO_INSTALL_DIR>/deployment_tools/model_optimizer/install_prerequisites directory and run:
```bash
install_prerequisites_onnx.sh
```
Use the `mo_onnx.py` script from the <INSTALL_DIR>/deployment_tools/model_optimizer directory to run the Model Optimizer:
```bash
python mo_onnx.py --input_model <INPUT_MODEL.onnx> --mean_values [151.2405,119.595,107.8395] --scale_values [63.0105,56.457,55.0035] --reverse_input_channels
```
Note, that parameters of the mean and scale values should be in the [0,255] range (byte format)
To check that there are no mistakes with the conversion you can launch `conversion_checker.py` by writing the following command:
```bash
python conversion_checker.py --config <path to config>; --spf_model_torch <path to torch model> --spf_model_openvino <path to OpenVINO model>;
```
You will see the mean difference (L1 metric distance) on the first and second predicted class. If it's 10e-6 or less then it's all good.

## Demo
![demo.png](./demo/demo.png)
To start demo you need to [download] OpenVINO™ face detector model. Concretely, you will need `face-detection-0100` version.
On [google drive](https://drive.google.com/drive/u/0/folders/1A6wa3AlrdjyNPkXT81knIzXxR7SAYm1q) you will see a trained antispoofing model that you can download and run, or choose your own trained model. Use OpenVINO™ format to obtain the best performance speed, but PyTorch format will work as well.

After preparation start demo by running:
```bash
python demo/demo.py --fd_model /path_to_face_detecor.xml --spf_model /path_to_antispoofing_model.xml(.pth.tar) --cam_id 0 --config config.py;
```
Refer to `--help` for additional parameters. If you are using the PyTorch model then you need to specify training config with the `--config` option. To run the demo on the video, you should change `--cam_id` on `--video` option and specify your_video.mp4
