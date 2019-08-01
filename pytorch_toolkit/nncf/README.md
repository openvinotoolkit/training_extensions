
# Neural Network Compression Framework  (NNCF)

This repository contains a PyTorch-based framework and samples for neural networks compression. The framework organized as a Python module that can be built and used in a standalone mode. The framework architecture is unified to make it easy to add different compression methods. The samples demonstrate the usage of compression algorithms for three different use cases on public models and datasets: Image Classification, Object Detection and Semantic Segmentation.

## Key features:

- Support of quantization and sparsity algorithms with fine-tuning.
- Automatic model graph transformation. The model is wrapped by the custom class and additional layers are inserted in the graph. The transformations are configurable.
- Common interface for compression methods.
- GPU-accelerated layers for fast model fine-tuning.
- Distributed training support.
- Configuration file examples for sparsity, quantization and sparsity with quantization. Each type of compression requires only one additional fine-tuning stage.
- Export models to ONNX format that is supported by [OpenVINO](https://github.com/opencv/dldt) Toolkit.

For more details about framework architecture please refer to [NNCF description](nncf/README.md).

## Usage
It is assumed that NNCF can be used in two different ways:
- A standalone package that can installed and integrated into a custom training pipeline. For example, a user has a training sample for a model that solves Action Recognition problem so the best option is to install the NNCF pip package into the Python environment. For more information about NNCF standalone usage refer to this [manual](./docs/PackageUsage.md).
- Training sample that demonstrates model compression capabilities.

Any of these two options implies providing a configuration file which contains hyperparameters of training and compression algorithms. Each of compression samples contains examples of configuration files which implements compression of standard DL models. For more information about configuration files please refer to the corresponding [manual file](docs/Configuration.md).

### Model compression samples
To run the samples please refer to the corresponding tutorials:
- [Image Classification sample](examples/classification/README.md)
- [Object Detection sample](examples/object_detection/README.md)
- [Semantic Segmentation sample](examples/segmentation/README.md)

### System requirements
- Ubuntu 16.04 or later (64-bit)
- Python 3.5 or later
- NVidia CUDA Toolkit 9.0 or later

### Installation
We suggest to install or use the package into the [Python virtual environment](https://docs.python.org/3/tutorial/venv.html).
- Install the following dependencies: `sudo apt-get install python3-dev`
- Activate environment and install the project dependencies running `pip install -r requirements.txt`
- Use as a standalone package:
   - Clone repository
   - In the project folder run `python setup.py bdist_wheel` to build package
   - Then you can install package running `pip install dist/nncf-<vesion>.whl`
- Use project samples
   - Clone repository
   - In the project folder run `python setup.py develop` to install NNCF in your environment

## Some recent results

| Model | Dataset | PyTorch FP32 baseline | PyTorch compressed accuracy  | 
|--|--|--|--|
| ResNet-50 INT8 quantized | ImageNet | 76.13 | 76.49 | 
| ResNet-50 INT8 w/ 60% of sparsity (RB) | ImageNet | 76.13 | 75.2 | 
| Inception v3 INT8 quantized | ImageNet | 77.32 | 78.06 | 
| Inception v3 INT8 w/ 60% of sparsity (RB) | ImageNet | 77.32 | 76.8 |
| MobileNet v2 INT8 quantized | ImageNet | 71.8 | 71.3 | 
| MobileNet v2 INT8 w/ 51% of sparsity (RB) | ImageNet | 71.8 | 70.9 | 
| SqueezeNet v1.1 INT8 quantized | ImageNet | 58.19 | 57.9 | 
| SSD300-BN INT8 quantized | VOC12+07 | 78.28 | 78.18 |
| SSD300-BN INT8 w/ 70% of sparsity (Magnitude) | VOC12+07 | 78.28 | 77.63 |
| SSD512-BN INT8 quantized | VOC12+07 | 80.26 | 80.32 |
| SSD512-BN INT8 w/ 70% of sparsity (Magnitude) | VOC12+07 | 80.26 | 79.90 |
| UNet INT8 quantized | CamVid | 71.95 | 71.82 |
| UNet INT8 w/ 60% of sparsity (Magnitude) | CamVid | 71.95 | 71.90 |
| UNet INT8 quantized | Mapillary | 56.23 | 56.16 |
| UNet INT8 w/ 60% of sparsity (Magnitude) | Mapillary | 56.23 | 54.30 |
| ICNet INT8 quantized | CamVid | 67.89 | 67.69 |
| ICNet INT8 w/ 60% of sparsity (Magnitude) | CamVid | 67.89 | 67.53 |

