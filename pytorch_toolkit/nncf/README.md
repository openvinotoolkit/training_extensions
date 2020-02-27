# Neural Network Compression Framework (NNCF)

This module contains a PyTorch\*-based framework and samples for neural networks compression. The framework is organized as a Python\* module that can be built and used in a standalone mode. The framework architecture is unified to make it easy to add different compression methods. The samples demonstrate the usage of compression algorithms for three different use cases on public models and datasets: Image Classification, Object Detection and Semantic Segmentation.

## Key Features

- Support of quantization, binarization, and sparsity algorithms with fine-tuning
- Automatic model graph transformation. The model is wrapped by the custom class and additional layers are inserted in the graph. The transformations are configurable.
- Common interface for compression methods
- GPU-accelerated layers for fast model fine-tuning
- Distributed training support
- Configuration file examples for sparsity, quantization, and sparsity with quantization. Each type of compression requires only one additional fine-tuning stage.
- Export models to ONNX\* format that is supported by the [OpenVINO&trade; toolkit](https://github.com/opencv/dldt).

For more details about framework architecture, refer to the [NNCF description](nncf/README.md).

## Usage
The NNCF can be used in two different ways:
- A standalone package that can be installed and integrated into a custom training pipeline. For example, the NNCF can be used with the [mmdetection](https://github.com/open-mmlab/mmdetection) pip package into the Python environment. For more information about NNCF standalone usage, refer to this [manual](./docs/PackageUsage.md).
- Training sample that demonstrates model compression capabilities

Any of these two options implies providing a configuration file which contains hyperparameters of training and compression algorithms. Each of compression samples contains examples of configuration files which implements compression of standard DL models. For more information about configuration files please refer to the corresponding [manual file](docs/Configuration.md).

### Model Compression Samples
To run the samples please refer to the corresponding tutorials:
- [Image Classification sample](examples/classification/README.md)
- [Object Detection sample](examples/object_detection/README.md)
- [Semantic Segmentation sample](examples/semantic_segmentation/README.md)

### Third-party repository integration
NNCF may be straightforwardly integrated into training/evaluation pipelines of third-party repositories. See [third_party_integration](./third_party_integration) for examples of code modifications necessary to integrate NNCF into select repositories (Git patches and base commit IDs are provided).
### System requirements
- Ubuntu\* 16.04 or later (64-bit)
- Python\* 3.5 or later
- NVidia CUDA\* Toolkit 10.0 or later
- PyTorch\* 1.2.0 or higher.

### Installation
We suggest to install or use the package in the [Python virtual environment](https://docs.python.org/3/tutorial/venv.html).
- Install the following dependencies: `sudo apt-get install python3-dev`
- Activate environment and install the project dependencies running `pip install -r requirements.txt`
- Use as a standalone package:
   1. Clone repository
   2. In the project folder, run `python setup.py   bdist_wheel` to build package
   3. Install package running `pip install dist/nncf-<version>.whl`
- Use project samples
   1. Clone repository
   2. In the project folder run `python setup.py develop` to install NNCF in your environment

## Recent Results

|Model|Compression algorithm|Dataset|PyTorch FP32 baseline|PyTorch compressed accuracy|
| :---: | :---: | :---: | :---: | :---: |
|ResNet-50|None|ImageNet|-|76.13|
|ResNet-50|INT8|ImageNet|76.13|76.54|
|ResNet-50|INT8 + Sparsity 61% (RB)|ImageNet|76.13|75.28|
|Inception V3|None|ImageNet|-|77.32|
|Inception V3|INT8|ImageNet|77.32|77.46|
|Inception V3|INT8 + Sparsity 61% (RB)|ImageNet|77.32|77.02|
|MobileNet V2|None|ImageNet|-|71.81|
|MobileNet V2|INT8|ImageNet|71.81|71.33|
|MobileNet V2|INT8 + Sparsity 52% (RB)|ImageNet|71.81|71.04|
|SqueezeNet V1.1|None|ImageNet|-|58.18|
|SqueezeNet V1.1|INT8|ImageNet|58.18|58.31|
|ResNet-18|None|ImageNet|-|69.76|
|ResNet-18|XNOR (weights), scale/threshold (activations)|ImageNet|69.76|61.58|
|ResNet-18|DoReFa (weights), scale/threshold (activations)|ImageNet|69.76|61.57|
|SSD300-BN|None|VOC12+07|-|78.28|
|SSD300-BN|INT8|VOC12+07|78.28|78.02|
|SSD300-BN|INT8 + Sparsity 70% (Magnitude)|VOC12+07|78.28|77.96|
|SSD512-BN|None|VOC12+07|-|80.26|
|SSD512-BN|INT8|VOC12+07|80.26|80.58|
|SSD512-BN|INT8 + Sparsity 70% (Magnitude)|VOC12+07|80.26|80.11|
|UNet|None|CamVid|-|71.95|
|UNet|INT8|CamVid|71.95|71.66|
|UNet|INT8 + Sparsity 60% (Magnitude)|CamVid|71.95|71.72|
|ICNet|None|CamVid|-|67.89|
|ICNet|INT8|CamVid|67.89|67.78|
|ICNet|INT8 + Sparsity 60% (Magnitude)|CamVid|67.89|67.55|
|UNet|None|Mapillary|-|56.23|
|UNet|INT8|Mapillary|56.23|56.12|
|UNet|INT8 + Sparsity 60% (Magnitude)|Mapillary|56.23|56.0|
