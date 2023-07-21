Models Optimization
===================

OpenVINO™ Training Extensions provides two types of optimization algorithms: `Post-training Optimization Tool (POT) <https://docs.openvino.ai/latest/pot_introduction.html#doxid-pot-introduction>`_ and `Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`_.

*******************************
Post-training Optimization Tool
*******************************

POT is designed to optimize the inference of models by applying post-training methods that do not require model retraining or fine-tuning. If you want to know more details about how POT works and to be more familiar with model optimization methods, please refer to `documentation <https://docs.openvino.ai/latest/pot_introduction.html#doxid-pot-introduction>`_.

To run Post-training optimization it is required to convert the model to OpenVINO™ intermediate representation (IR) first. To perform fast and accurate quantization we use ``DefaultQuantization Algorithm`` for each task. Please, see the `DefaultQuantization Parameters <https://docs.openvino.ai/latest/pot_compression_algorithms_quantization_default_README.html#doxid-pot-compression-algorithms-quantization-default-r-e-a-d-m-e>`_ for further information about configuring the optimization.

POT parameters can be found and configured in ``template.yaml`` and ``configuration.yaml`` for each task. For Anomaly and Semantic Segmentation tasks, we have separate configuration files for POT, that can be found in the same directory with ``template.yaml``, for example for `PaDiM <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/algorithms/anomaly/configs/classification/padim/ptq_optimization_config.py>`_, `OCR-Lite-HRNe-18-mod2 <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/algorithms/segmentation/configs/ocr_lite_hrnet_18_mod2/ptq_optimization_config.py>`_ model.

************************************
Neural Network Compression Framework
************************************

NNCF utilizes Training-time Optimization algorithms. It is a set of advanced algorithms for training-time model optimization within the Deep Learning framework such as Pytorch.
The process of optimization is controlled by the NNCF configuration file. A JSON configuration file is used for easier setup of the parameters of the compression algorithm. See `configuration file description <https://github.com/openvinotoolkit/nncf/blob/develop/docs/ConfigFile.md>`_.

You can refer to configuration files for default templates for each task accordingly: `Classification <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/algorithms/classification/configs/efficientnet_b0_cls_incr/compression_config.json>`_, `Object Detection <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/algorithms/detection/configs/detection/mobilenetv2_atss/compression_config.json>`_, `Semantic segmentation <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/algorithms/segmentation/configs/ocr_lite_hrnet_18_mod2/compression_config.json>`_, `Instance segmentation <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/algorithms/detection/configs/instance_segmentation/efficientnetb2b_maskrcnn/compression_config.json>`_, `Anomaly classification <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/algorithms/anomaly/configs/classification/padim/compression_config.json>`_, `Anomaly Detection <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/algorithms/anomaly/configs/detection/padim/compression_config.json>`_, `Anomaly segmentation <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/algorithms/anomaly/configs/segmentation/padim/compression_config.json>`_. Configs for other templates can be found in the same directory.


NNCF tends to provide better quality in terms of preserving accuracy as it uses training compression approaches.
Compression results achievable with the NNCF can be found `here <https://github.com/openvinotoolkit/nncf#nncf-compressed-model-zoo>`_ .
Meanwhile, the POT is faster but can degrade accuracy more than the training-enabled approach.

.. note::
    The main recommendation is to start with post-training compression and use NNCF compression during training if you are not satisfied with the results.

Please, refer to our :doc:`dedicated tutorials <../../tutorials/base/how_to_train/index>` on how to optimize your model using POT or NNCF.