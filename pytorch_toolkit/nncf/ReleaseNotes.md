# Release Notes

## Introduction
*Neural Network Compression Framework  (NNCF)* is a toolset for Neural Networks model compression. The framework organized as a Python module that can be built and used standalone or within samples distributed with the code.  The samples demonstrate the usage of compression methods on public models and datasets for three different use cases: Image Classification, Object Detection, and Semantic Segmentation.

## New in Release 1.0:

- Support of symmetric quantization and two sparsity algorithms with fine-tuning.
- Automatic model graph transformation. The model is wrapped by the custom class and additional layers are inserted in the graph. The transformations are configurable.
- Three training samples which demonstrate usage of compression methods from NNCF:
	- Image classification:  torchvision models for classification and custom models on ImageNet and CIFAR10/100 datasets.
	- Object Detection: SSD300, SSD512, MobileNet SSD on VOC07+12 and COCO datasets. 
	- Semantic Segmentation: UNet, ICENet on CamVid and Mapillary Vistas datasets.
- Unified interface for compression methods.	
- GPU-accelerated *Quantization* layer for fast model fine-tuning.
- Distributed training support in all samples.
- Configuration file examples for sparsity, quantization and sparsity with quantization for all three samples. Each type of compression requires only one additional stage of fine-tuning.
- Export models to ONNX format that is supported by the [OpenVINO](https://github.com/opencv/dldt) Toolkit.
