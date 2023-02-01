Classification  model
================================

This tutorial reveals end-to-end solution from installation to model export and optimization for classification task on a specific dataset.
On this page, we show how to train, validate, export and optimize :ref:`MobileNet-V3-large-1x <classificaiton_models>` model on the `flowers dataset <https://www.tensorflow.org/hub/tutorials/image_feature_vector#the_flowers_dataset>`_ from TensorFlow.

.. note::

  To learn how to deploy the trained model, refer to: :doc:`../deploy`.

  To learn how to run the demo and visualize results, refer to: :doc:`../demo`.

The process has been tested on the following configuration.

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-10980XE
- CUDA Toolkit 11.4



*************************
Setup virtual environment
*************************

You can follow the installation process from a :doc:`quick start guide <../../../get_started/installation>` to create a universal virtual environment for OTX.

***************************
Dataset preparation
***************************

1.  Download and prepare a `flowers dataset <https://www.tensorflow.org/hub/tutorials/image_feature_vector#the_flowers_dataset>`_ with the following command:

.. code-block::

  cd data
  wget http://download.tensorflow.org/example_images/flower_photos.tgz
  tar -xzvf archive.tar.gz
  cd ..


.. image:: ../../../../utils/images/flowers.jpg
  :width: 600


This dataset contains images of 5 different flower categories and is stored in the imagenet format which is supported by OTX:

.. code-block::

  flower_photos
    ├── daisy
    ├── dandelion
    ├── roses
    ├── sunflowers
    ├── tulips


2.  Next, we need to create train/validation sets.
We prepared a dedicated script that splits the dataset into 80% for training and 20% for validation. To prepare the dataset run the following command:

.. code-block::

  (otx) ...$ python docs\utils\prepare_classification_dataset.py --data-root data/flower_photos

3.  ``(Optional)`` To simplify the command line functions calling, we may create a ``data.yaml`` file with data roots info and pass it as a ``--data`` parameter to ``otx`` command lines.
The content of the ``data.yaml`` for a dataset should have absolute paths and will be similar to that:

  .. code-block::

    {'data':
      {'train':
        {'data-roots': '/home/<username>/training_extensions/data/flower_photos/train'},
      'val':
        {'data-roots': '/home/<username>/training_extensions/data/flower_photos/val'},
      'test':
        {'data-roots': '/home/<username>/training_extensions/data/flower_photos/val'}
      }
    }

*********
Training
*********

1. First of all, we need to choose which classification model will we train.
The list of supported templates for classification is available with the command line below.

.. note::

  The characteristics and detailed comparison of the models could be found in :doc:`Explanation section <../../../explanation/algorithms/classification/multi_class_classification>`.

  We also can modify the architecture of supported models with various backbones, please refer to the :doc:`advanced tutorial for model customization <../../advanced/backbones>`.

.. code-block::

  (otx) ...$ otx find --template --task CLASSIFICATION

  +----------------+---------------------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
  |      TASK      |                         ID                        |          NAME         |                                        PATH                                       |
  +----------------+---------------------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
  | CLASSIFICATION | Custom_Image_Classification_MobileNet-V3-large-1x | MobileNet-V3-large-1x | otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml |
  | CLASSIFICATION |    Custom_Image_Classification_EfficinetNet-B0    |    EfficientNet-B0    |    otx/algorithms/classification/configs/efficientnet_b0_cls_incr/template.yaml   |
  | CLASSIFICATION |   Custom_Image_Classification_EfficientNet-V2-S   |   EfficientNet-V2-S   |   otx/algorithms/classification/configs/efficientnet_v2_s_cls_incr/template.yaml  |
  +----------------+---------------------------------------------------+-----------------------+-----------------------------------------------------------------------------------+

2. ``otx train`` trains a specific model template
on a dataset and results in two files:

- ``weights.pth`` - a model snapshot
- ``label_schema.json`` - a label schema used in training, created from a dataset

These are needed as inputs for the further commands: ``export``, ``eval``,  ``optimize``,  ``deploy`` and ``demo``.


3. To have a specific example in this tutorial, all commands will be run on the MobileNet-V3-large-1x model.
It's a light model, that achieves competitive accuracy while keeping the inference fast.

The following command line starts training the classification model on the first GPU on our dataset:

.. code-block::

  (otx) ...$ otx train otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml \
                            --train-data-roots data/flower_photos/train \
                            --val-data-roots data/flower_photos/val \
                            --save-model-to model_outputs \
                            --work-dir outputs/logs \
                            --gpus 1

To start multi-gpu training, list the indexes of GPUs you want to train on or omit `gpus` parameter, so training will run on all available GPUs.

If you created ``data.yaml`` file in the previous step, you can simplify the training by passing it in ``--data`` parameter.

.. code-block::

  (otx) ...$ otx train otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml \
                            --data data.yaml \
                            --save-model-to outputs \
                            --work-dir outputs/logs \
                            --gpus 1

You can also pass the ``data.yaml`` for the rest of the OTX CLI commands (eval, export, optimize) that require annotation paths.

4. ``(Optional)`` Additionally, we can tune training parameters such as batch size, learning rate, patience epochs or warm-up iterations.
More about template-specific parameters is in :doc:`quick start <../../../get_started/index>`.

It can be done by manually updating parameters in the ``template.yaml`` file or via the command line.

For example, to decrease the batch size to 4, fix the number of epochs to 100 and disable early stopping, extend the command line above with the following line.

.. code-block::

  params --learning_parameters.batch_size 4 --learning_parameters.num_iters 100 --learning_parameters.enable_early_stopping false


5. The training results are ``weights.pth`` and ``label_schema.json`` files that located in ``model_outputs`` folder,
while training logs and tf_logs for `Tensorboard` visualization can be found in the ``outputs/logs`` dir.

.. code-block::

  2023-02-02 02:56:51,220 | INFO :
  Early Stopping at :31 with best accuracy: 0.9540983581542969
  2023-02-02 02:56:51,220 | INFO : Epoch(val) [32][44]    accuracy_top-1: 0.9464, accuracy_top-5: 1.0000, daisy accuracy: 0.9684, dandelion accuracy: 0.9732, roses accuracy: 0.8562, sunflowers accuracy: 0.9540, tulips accuracy: 0.9648, mean accuracy: 0.9433, accuracy: 0.9464, current_iters: 1408
  2023-02-02 02:56:52,314 | INFO : run task done.
  2023-02-02 02:56:52,406 | INFO : called save_model
  2023-02-02 02:56:52,580 | INFO : Final model performance: Performance(score: 0.9540983581542969, dashboard: (18 metric groups))

The training time highly relies on the hardware characteristics, for example on 1 GeForce 3090 the training took about 8 minutes.

After that, we have the PyTorch classification model trained with OTX, which we can use for evaluation, export, optimization and deployment.

***********
Validation
***********

1. ``otx eval`` runs evaluation of a trained
model on a specific dataset.

The eval function receives test annotation information and model snapshot, trained in the previous step.
Please note, ``label_schema.json`` file contains meta-information about the dataset and it should be located in the same folder as the model snapshot.

otx eval will output a top-1 accuracy score for multi-class classification

2. The command below will run validation on our dataset
and save performance results in ``outputs/performance`` folder:

.. code-block::

  (otx) ...$ otx train otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml \
                            --train-data-roots data/flower_photos/train \
                            --val-data-roots data/flower_photos/val \
                            --save-model-to model_outputs \
                            --work-dir outputs/logs \
                            --gpus 1


If you created ``data.yaml`` file in the previous step, you can simplify the training by passing it in ``--data`` parameter.
Note,  with ``data.yaml``, it runs evaluation on the test data root, not on the validation.

.. code-block::

  (otx) ...$ otx eval otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml \
                            --test-data-roots data/flower_photos/val \
                            --load-weights model_outputs/weights.pth \
                            --save-performance outputs/performance.json

We will get a similar to this validation output:

.. code-block::

  ...

  2023-02-02 03:14:54,488 | INFO : run task done.
  2023-02-02 03:15:03,729 | INFO : called evaluate()
  2023-02-02 03:15:03,745 | INFO : Accuracy after evaluation: 0.9540983606557377
  2023-02-02 03:15:03,745 | INFO : Evaluation completed
  Performance(score: 0.9540983606557377, dashboard: (3 metric groups))

3. The output of ``./outputs/performance.json`` consists
of a dict with the target metric name and its value.

.. code-block::

  ...
  {"Accuracy": 0.9540983606557377}

*********
Export
*********

1. ``otx export`` exports a trained Pytorch `.pth` model to the OpenVINO™ Intermediate Representation (IR) format.
It allows running the model on the Intel hardware much more efficient, especially on the CPU. Also, the resulting IR model is required to run POT optimization in the section below. IR model consists of 2 files: ``openvino.xml`` for weights and ``openvino.bin`` for architecture.

2. We can run the below command line to export the trained model ``model_outputs/weights.pth`` from the previous section
and save the exported model to the ``outputs/openvino`` folder.

.. code-block::

  (otx) ...$ otx export otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml \
                              --load-weights model_outputs/weights.pth \
                              --save-model-to outputs/openvino

  ...

  2023-02-02 03:23:03,057 | INFO : run task done.
  2023-02-02 03:23:03,064 | INFO : Exporting completed


3. We can check the accuracy of the IR model and the consistency between the exported model and the PyTorch model,
using ``otx eval`` and passing the IR model path to the ``--load-weights`` parameter.

.. code-block::

  (otx) ...$ otx eval otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml \
                            --test-data-roots data/flower_photos/val \
                            --load-weights outputs/openvino/openvino.xml \
                            --save-performance outputs/openvino/performance.json

  ...

  Performance(score: 0.9540983606557377, dashboard: (3 metric groups))


*************
Optimization
*************

1. We can further optimize the model with ``otx optimize``.
It uses NNCF or POT depending on the model format.

``NNCF`` optimization is used for trained snapshots in a framework-specific format such as checkpoint (.pth) file from Pytorch. It starts accuracy-aware quantization based on the obtained weights from the training stage. Generally, we will see the same output as during training.

``POT`` optimization is used for models exported in the OpenVINO™ IR format. It decreases the floating-point precision to integer precision of the exported model by performing the post-training optimization.

The function results in the following files, which could be used to run :doc:`otx demo <../demo>`:

- ``weights.pth``
- ``label_schema.json``
- ``openvino.bin``
- ``openvino.xml``

Please, refer to :doc:`optimization explanation <../../../explanation/additional_features/models_optimization>` section to get the intuition of what we use under the hood for optimization purposes.

2. Command example for optimizing
a PyTorch model (`.pth`) with OpenVINO™ NNCF.

.. code-block::

  (otx) ...$ otx optimize otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml \
                            --train-data-roots data/flower_photos/train \
                            --val-data-roots data/flower_photos/val \
                            --save-model-to model_outputs/nncf \
                            --load-weights model_outputs/weights.pth \
                            --save-performance outputs/nncf_performance.json

  ...

  2023-02-02 03:41:27,059 | INFO : run task done.
  2023-02-02 03:41:34,925 | INFO : called evaluate()
  2023-02-02 03:41:34,942 | INFO : Accuracy after evaluation: 0.9475409836065574
  2023-02-02 03:41:34,942 | INFO : Evaluation completed
  Performance(score: 0.9475409836065574, dashboard: (3 metric groups))

The optimization time relies on the hardware characteristics, for example on 1 GeForce 3090 and Intel(R) Core(TM) i9-10980XE it took about 10 minutes.

3.  Command example for optimizing
OpenVINO™ model (.xml) with OpenVINO™ POT.

.. code-block::

  (otx) ...$ otx optimize otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml \
                                --train-data-roots data/flower_photos/train \
                                --val-data-roots data/flower_photos/val \
                                --load-weights outputs/openvino/openvino.xml \
                                --save-model-to outputs/pot

  ...

  Performance(score: 0.9453551912568307, dashboard: (3 metric groups))

Please note, that POT will take some time (generally less than NNCF optimization) without logging to optimize the model.

4. Finally, we can also evaluate the optimized
model by passing it to the ``otx eval`` function.

Now we have fully trained, optimized and exported an efficient model representation ready-to-use classification model.

The following tutorials provide further steps on how to :doc:`deploy <../deploy>` and use your model in the :doc:`demonstration mode <../demo>` and visualize results.
The examples are provided with an object detection model, but it is easy to apply them for classification by substituting the object detection model with classification one.