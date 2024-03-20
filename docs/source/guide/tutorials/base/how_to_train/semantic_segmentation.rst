Semantic Segmentation model
================================

This tutorial demonstrates how to train and optimize a semantic segmentation model using the VOC2012 dataset from the PASCAL Visual Object Classes Challenge 2012.
The trained model will be used to segment images by assigning a label to each pixel of the input image.

To learn more about Segmentation task, refer to :doc:`../../../explanation/algorithms/segmentation/semantic_segmentation`.

.. note::
  To learn more about managing the training process of the model including additional parameters and its modification, refer to :doc:`./detection`.

The process has been tested on the following configuration.

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-11900
- CUDA Toolkit 11.8

*************************
Setup virtual environment
*************************

1. You can follow the installation process from a :doc:`quick start guide <../../../get_started/installation>`
to create a universal virtual environment for OpenVINO™ Training Extensions.

2. Activate your virtual
environment:

.. code-block:: shell

  .otx/bin/activate
  # or by this line, if you created an environment, using tox
  . venv/otx/bin/activate

***************************
Dataset preparation
***************************

Download and prepare `VOC2012 dataset <http://host.robots.ox.ac.uk/pascal/VOC/voc2012>`_ with the following command:

.. code-block::

  cd data
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  tar -xvf VOCtrainval_11-May-2012.tar
  cd ..

.. image:: ../../../../../utils/images/voc_example.png
  :width: 600

The dataset contains a set of RGB images with 20 semantic labels such as aeroplane, bicycle, bird, car, person, etc. The images are stored in the following format:

.. code-block::

  VOC2012
  ├── Annotations
  ├── ImageSets
  ├── JPEGImages
  ├── SegmentationClass
  ├── SegmentationObject



*********
Training
*********

1. First of all, you need to choose which semantic segmentation model you want to train.
The list of supported templates for semantic segmentation is available with the command line below.

.. note::

  The characteristics and detailed comparison of the models could be found in :doc:`Explanation section <../../../explanation/algorithms/segmentation/semantic_segmentation>`.

  We also can modify the architecture of supported models with various backbones, please refer to the :doc:`advanced tutorial for model customization <../../advanced/backbones>`.

.. code-block::

  (otx) ...$ otx find --task segmentation

  +--------------+-----------------------------------------------------+--------------------+------------------------------------------------------------------------------+
  |     TASK     |                          ID                         |        NAME        |                                  BASE PATH                                   |
  +--------------+-----------------------------------------------------+--------------------+------------------------------------------------------------------------------+
  | SEGMENTATION |    Custom_Semantic_Segmentation_Lite-HRNet-18_OCR   |   Lite-HRNet-18    |   src/otx/algorithms/segmentation/configs/ocr_lite_hrnet_18/template.yaml    |
  | SEGMENTATION | Custom_Semantic_Segmentation_Lite-HRNet-18-mod2_OCR | Lite-HRNet-18-mod2 | src/otx/algorithms/segmentation/configs/ocr_lite_hrnet_18_mod2/template.yaml |
  | SEGMENTATION |  Custom_Semantic_Segmentation_Lite-HRNet-s-mod2_OCR | Lite-HRNet-s-mod2  | src/otx/algorithms/segmentation/configs/ocr_lite_hrnet_s_mod2/template.yaml  |
  | SEGMENTATION |  Custom_Semantic_Segmentation_Lite-HRNet-x-mod3_OCR | Lite-HRNet-x-mod3  | src/otx/algorithms/segmentation/configs/ocr_lite_hrnet_x_mod3/template.yaml  |
  +--------------+-----------------------------------------------------+--------------------+------------------------------------------------------------------------------+

.. note::

  We do not attach an OCR head for supported models in default. We remain the suffix '_OCR' in ID just for backward compatibility.

To have a specific example in this tutorial, all commands will be run on the :ref:`Lite-HRNet-18-mod2 <semantic_segmentation_models>`  model. It's a light model, that achieves competitive accuracy while keeping the inference fast.


2.  Next, we need to create train/validation sets.
OpenVINO™ Training Extensions supports auto-split functionality for semantic segmentation.

.. note::

  Currently, OpenVINO™ Training Extensions supports auto-split only for public VOC dataset format in semantic segmentation. We should specify the validation roots in the argument ``--val-data-roots`` when using other supported segmentation dataset. To learn about dataset formats for semantic segmentation, please refer to the :doc:`explanation section <../../../explanation/algorithms/segmentation/semantic_segmentation>`.

Let's prepare an OpenVINO™ Training Extensions semantic segmentation workspace running the following command:

.. code-block::

  (otx) ...$ otx build --train-data-roots data/VOCdevkit/VOC2012 --model Lite-HRNet-18-mod2

  [*] Load Model Template ID: Custom_Semantic_Segmentation_Lite-HRNet-18-mod2_OCR
  [*] Load Model Name: Lite-HRNet-18-mod2

  ...

  [*] Update data configuration file to: otx-workspace-SEGMENTATION/data.yaml

  (otx) ...$ cd ./otx-workspace-SEGMENTATION

It will create **otx-workspace-SEGMENTATION** with all necessary configs for Lite-HRNet-18-mod2, prepared ``data.yaml`` to simplify CLI commands launch and splitted dataset.

3. To start training we need to call ``otx train``
command in our workspace:

.. code-block::

  (otx) ...$ otx train

That's it! The training will return artifacts: ``weights.pth`` and ``label_schema.json``, which are needed as input for the further commands: ``export``, ``eval``,  ``optimize``,  etc.

After that, we have the PyTorch model trained with OpenVINO™ Training Extensions, which we can use for evaluation, export, optimization and deployment.

***********
Validation
***********

1. ``otx eval`` runs evaluation of a trained
model on a specific dataset.
The eval function receives test annotation information and model snapshot, trained in the previous step.
Please note, ``label_schema.json`` file contains meta information about the dataset and it should be located in the same folder as the model snapshot.

``otx eval`` will output a ``mDice`` score for semantic segmentation.

2. The command below will run validation on our splitted dataset. We can use other test dataset as well by specifying the path where test data exists in argument ``--test-data-roots``.
By running this example command, the performance results evaluated by our splitted validation dataset are saved in ``performance.json`` file:

.. code-block::

  (otx) ...$ otx eval --test-data-roots splitted_dataset/val \
                      --load-weights models/weights.pth \
                      --output outputs

Finally, we get the validation output:

.. code-block::

  ...

  2023-02-21 18:09:56,134 | INFO : run task done.
  2023-02-21 18:09:57,807 | INFO : called evaluate()
  2023-02-21 18:09:57,807 | INFO : Computing mDice
  2023-02-21 18:09:58,508 | INFO : mDice after evaluation: 0.9659400544959128
  Performance(score: 0.9659400544959128, dashboard: (1 metric groups))

In ``outputs/performance.json`` file, the validation output score is saved as:

.. code-block::

  {"Dice Average": 0.9659400544959128}


*********
Export
*********

1. ``otx export`` exports a trained Pytorch `.pth` model to the OpenVINO™ Intermediate Representation (IR) format.
It allows running the model on the Intel hardware much more efficient, especially on the CPU. Also, the resulting IR model is required to run PTQ optimization. IR model consists of 2 files: ``openvino.xml`` for weights and ``openvino.bin`` for architecture.

2. We can run the below command line to export the trained model
and save the exported model to the ``openvino_model`` folder.

.. code-block::

  (otx) ...$ otx export --load-weights models/weights.pth \
                        --output openvino_model

  ...

  2023-02-02 03:23:03,057 | INFO : run task done.
  2023-02-02 03:23:03,064 | INFO : Exporting completed


3. We can check the ``mDice`` score of the IR model and the consistency between the exported model and the PyTorch model,
using ``otx eval`` and passing the IR model path to the ``--load-weights`` parameter.

.. code-block::

  (otx) ...$ otx eval --test-data-roots splitted_dataset/val \
                      --load-weights openvino_model/openvino.xml \
                      --output openvino_model

  ...

  Performance(score: 0.9659400544959128, dashboard: (1 metric groups))


*************
Optimization
*************

1. We can further optimize the model with ``otx optimize``.
It uses NNCF or PTQ depending on the model and transforms it to ``INT8`` format.

Please, refer to :doc:`optimization explanation <../../../explanation/additional_features/models_optimization>` section to get the intuition of what we use under the hood for optimization purposes.

2. Command example for optimizing
a PyTorch model (`.pth`) with OpenVINO™ NNCF.

.. code-block::

  (otx) ...$ otx optimize --load-weights models/weights.pth --output nncf_model

  ...

  INFO:nncf:Loaded 5286/5286 parameters
  2023-02-21 18:09:56,134 | INFO : run task done.
  2023-02-21 18:09:57,807 | INFO : called evaluate()
  2023-02-21 18:09:57,807 | INFO : Computing mDice
  2023-02-21 18:09:58,508 | INFO : mDice after evaluation: 0.9659400544959128
  Performance(score: 0.9659400544959128, dashboard: (1 metric groups))

The optimization time relies on the hardware characteristics, for example on 1 NVIDIA GeForce RTX 3090 and Intel(R) Core(TM) i9-10980XE it took about 15 minutes.

3.  Command example for optimizing
OpenVINO™ model (.xml) with OpenVINO™ PTQ.

.. code-block::

  (otx) ...$ otx optimize --load-weights openvino_model/openvino.xml \
                          --output ptq_model

  ...

  Performance(score: 0.9577656675749319, dashboard: (1 metric groups))

Please note, that PTQ will take some time (generally less than NNCF optimization) without logging to optimize the model.

4. Now we have fully trained, optimized and exported an
efficient model representation ready-to-use semantic segmentation model.

The following tutorials provide further steps on how to :doc:`deploy <../deploy>` and use your model in the :doc:`demonstration mode <../demo>` and visualize results.
The examples are provided with an object detection model, but it is easy to apply them for semantic segmentation by substituting the object detection model with segmentation one.