Action Classification model
================================

This live example shows how to easily train, validate, optimize and export classification model on the `HMDB51 <https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/>`_.
To learn more about Action Classification task, refer to :doc:`../../../explanation/algorithms/action/action_classification`.

.. note::
  To learn more about managing the training process of the model including additional parameters and modification, refer to :doc:`./detection`.

  To learn how to deploy the trained model, refer to: :doc:`../deploy`.

  To learn how to run the demo and visualize results, refer to: :doc:`../demo`.

The process has been tested on the following configuration.

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-10980XE
- CUDA Toolkit 11.6

.. note::

  To learn more about the model, algorithm and dataset format, refer to :doc:`action classification explanation <../../../explanation/algorithms/action/action_classification>`.


*************************
Setup virtual environment
*************************

1. You can follow the installation process from a :doc:`quick start guide <../../../get_started/installation>`
to create a universal virtual environment for OpenVINO™ Training Extensions.

2. Activate your virtual
environment:

.. code-block::

  .otx/bin/activate
  # or by this line, if you created an environment, using tox
  . venv/otx/bin/activate

***************************
Dataset preparation
***************************

According to the `documentation <https://mmaction2.readthedocs.io/en/latest/supported_datasets.html#hmdb51>`_ provided by mmaction2, you need to ensure that the `HMDB51 <https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/>`_ dataset is structured as follows:

.. code-block::

    training_extensions
    ├── data
    │   ├── hmdb51
    │   │   ├── hmdb51_{train,val}_split_{1,2,3}_rawframes.txt
    │   │   ├── hmdb51_{train,val}_split_{1,2,3}_videos.txt
    │   │   ├── annotations
    │   │   ├── videos
    │   │   │   ├── brush_hair
    │   │   │   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0.avi
    │   │   │   ├── wave
    │   │   │   │   ├── 20060723sfjffbartsinger_wave_f_cm_np1_ba_med_0.avi
    │   │   ├── rawframes
    │   │   │   ├── brush_hair
    │   │   │   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0
    │   │   │   │   │   ├── img_00001.jpg
    │   │   │   │   │   ├── img_00002.jpg
    │   │   │   │   │   ├── ...
    │   │   │   │   │   ├── flow_x_00001.jpg
    │   │   │   │   │   ├── flow_x_00002.jpg
    │   │   │   │   │   ├── ...
    │   │   │   │   │   ├── flow_y_00001.jpg
    │   │   │   │   │   ├── flow_y_00002.jpg
    │   │   │   ├── ...
    │   │   │   ├── wave
    │   │   │   │   ├── 20060723sfjffbartsinger_wave_f_cm_np1_ba_med_0
    │   │   │   │   ├── ...
    │   │   │   │   ├── winKen_wave_u_cm_np1_ri_bad_1

Once you have the dataset structured properly, copy ``mmaction2/data`` folder, which contains hmdb51 dataset, to ``training_extensions/data``.
Then, you can now convert it to the `CVAT <https://www.cvat.ai/>`_ format using the following command:

.. code-block::

  (otx) ...$ python3 src/otx/algorithms/action/utils/convert_public_data_to_cvat.py \
                     --task action_classification \
                     --src_path ./data/hmdb51/rawframes \
                     --dst_path ./data/hmdb51/CVAT/train \
                     --ann_file ./data/hmdb51/hmdb51_train_split_1_rawframes.txt \
                     --label_map ./data/hmdb51/label_map.txt

The resulting folder structure will be as follows:

.. code-block::

    hmdb51
    ├── rawframes
    ├── videos
    ├── annotations
    └── CVAT
        ├── train (3570 videos)
        │    ├── Video_0
        │    │   ├── annotations.xml
        │    │   └── images [101 frames]
        │    ├── Video_1
        │    │   ├── annotations.xml
        │    │   └── images [105 frames]
        │    └── Video_2
        │        ├── annotations.xml
        │        └── images [64 frames]
        │
        └── valid (1530 videos)
            ├── Video_0
            │   ├── annotations.xml
            │   └── images [85 frames]
            ├── Video_1
            │   ├── annotations.xml
            │   └── images [89 frames]
            └── Video_2
                ├── annotations.xml
                └── images [60 frames]

*********
Training
*********

1. You need to choose, which action classification model you want to train.
To see the list of supported templates, run the following command:

.. note::

  OpenVINO™ Training Extensions supports X3D and MoViNet template now, other architecture will be supported in future.

.. code-block::

  (otx) ...$ otx find --task action_classification

  +-----------------------+--------------------------------------+---------+-----------------------------------------------------------------------+
  |          TASK         |                  ID                  |   NAME  |                               BASE PATH                               |
  +-----------------------+--------------------------------------+---------+-----------------------------------------------------------------------+
  | ACTION_CLASSIFICATION |   Custom_Action_Classification_X3D   |   X3D   |   ../otx/algorithms/action/configs/classification/x3d/template.yaml   |
  | ACTION_CLASSIFICATION | Custom_Action_Classification_MoViNet | MoViNet | ../otx/algorithms/action/configs/classification/movinet/template.yaml |
  +-----------------------+--------------------------------------+---------+-----------------------------------------------------------------------+

All commands will be run on the X3D model. It's a light model, that achieves competitive accuracy while keeping the inference fast.

2. Prepare an OpenVINO™ Training Extensions workspace for
the action classification task by running the following command:

.. code-block::

  (otx) ...$ otx build --task action_classification --train-data-roots data/hmdb51/CVAT/train/ --val-data-roots data/hmdb51/CVAT/valid
  [*] Workspace Path: otx-workspace-ACTION_CLASSIFICATION
  [*] Load Model Template ID: Custom_Action_Classification_X3D
  [*] Load Model Name: X3D
  [*]     - Updated: otx-workspace-ACTION_CLASSIFICATION/model.py
  [*]     - Updated: otx-workspace-ACTION_CLASSIFICATION/data_pipeline.py
  [*] Update data configuration file to: otx-workspace-ACTION_CLASSIFICATION/data.yaml

  (otx) ...$ cd ./otx-workspace-ACTION_CLASSIFICATION

It will create **otx-workspace-ACTION_CLASSIFICATION** with all necessary configs for X3D and prepare ``data.yaml`` to simplify CLI commands.


3. To begin training, simply run ``otx train``
from **within the workspace directory**:

.. code-block::

  (otx) ...$ otx train

That's it! The training will return artifacts: ``weights.pth`` and ``label_schema.json``, which are needed as input for the further commands: ``export``, ``eval``,  ``optimize``,  etc.

The training time highly relies on the hardware characteristics. For example, the training took about 10 minutes on a single NVIDIA GeForce RTX 3090.

After that, you have the PyTorch action classification model trained with OpenVINO™ Training Extensions, which you can use for evaluation, export, optimization and deployment.

***********
Validation
***********

1. To evaluate the trained model on a specific dataset, use the ``otx eval`` command with
the following arguments:

The eval function receives test annotation information and model snapshot, trained in the previous step.
Keep in mind that ``label_schema.json`` file contains meta information about the dataset and it should be in the same folder as the model snapshot.

``otx eval`` will output a frame-wise accuracy for action classification. Note, that top-1 accuracy during training is video-wise accuracy.

2. The command below will run validation on the dataset
and save performance results in ``outputs/performance.json`` file:

.. code-block::

  (otx) ...$ otx eval --test-data-roots ../data/hmdb51/CVAT/valid \
                      --load-weights models/weights.pth \
                      --output outputs

You will get a similar validation output:

.. code-block::

  ...

    2023-02-22 00:08:45,156 - mmaction - INFO - Model architecture: X3D
    2023-02-22 00:08:56,766 - mmaction - INFO - Inference completed
    2023-02-22 00:08:56,766 - mmaction - INFO - called evaluate()
    2023-02-22 00:08:59,469 - mmaction - INFO - Final model performance: Performance(score: 0.6646406490691239, dashboard: (3 metric groups))
    2023-02-22 00:08:59,470 - mmaction - INFO - Evaluation completed
    Performance(score: 0.6646406490691239, dashboard: (3 metric groups))

*********
Export
*********

1. ``otx export`` exports a trained Pytorch `.pth` model to the OpenVINO™ Intermediate Representation (IR) format.
It allows running the model on the Intel hardware much more efficiently, especially on the CPU. Also, the resulting IR model is required to run POT optimization. IR model consists of two files: ``openvino.xml`` for weights and ``openvino.bin`` for architecture.

2. Run the command line below to export the trained model
and save the exported model to the ``openvino`` folder.

.. code-block::

  (otx) ...$ otx export --load-weights models/weights.pth \
                        --output openvino

  ...
  2023-02-21 22:54:32,518 - mmaction - INFO - Model architecture: X3D
  Successfully exported ONNX model: /tmp/OTX-task-a7wekgbc/openvino.onnx
  mo --input_model=/tmp/OTX-task-a7wekgbc/openvino.onnx --mean_values=[0.0, 0.0, 0.0] --scale_values=[255.0, 255.0, 255.0] --output_dir=/tmp/OTX-task-a7wekgbc --output=logits --data_type=FP32 --source_layout=??c??? --input_shape=[1, 1, 3, 8, 224, 224]
  [ WARNING ]  Use of deprecated cli option --data_type detected. Option use in the following releases will be fatal.
  [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
  Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
  [ SUCCESS ] Generated IR version 11 model.
  [ SUCCESS ] XML file: /tmp/OTX-task-a7wekgbc/openvino.xml
  [ SUCCESS ] BIN file: /tmp/OTX-task-a7wekgbc/openvino.bin
  2023-02-21 22:54:35,424 - mmaction - INFO - Exporting completed


3. Check the accuracy of the IR model and the consistency between the exported model and the PyTorch model,
using ``otx eval`` and passing the IR model path to the ``--load-weights`` parameter.

.. code-block::

  (otx) ...$ otx eval --test-data-roots ../data/hmdb51/CVAT/valid \
                      --load-weights openvino/openvino.xml \
                      --output outputs/openvino

  ...

  Performance(score: 0.6357698983041397, dashboard: (3 metric groups))


*************
Optimization
*************

1. You can further optimize the model with ``otx optimize``.
Currently, quantization jobs that include POT is supported for X3D template. MoViNet will be supported in near future.
Refer to :doc:`optimization explanation <../../../explanation/additional_features/models_optimization>` section for more details on model optimization.

2. Example command for optimizing
OpenVINO™ model (.xml) with OpenVINO™ POT.

.. code-block::

  (otx) ...$ otx optimize --load-weights openvino/openvino.xml \
                          --output pot_model

  ...

  Performance(score: 0.6252587703095486, dashboard: (3 metric groups))

Keep in mind that POT will take some time (generally less than NNCF optimization) without logging to optimize the model.

3. Now, you have fully trained, optimized and exported an
efficient model representation ready-to-use action classification model.

The following tutorials provide further steps on how to :doc:`deploy <../deploy>` and use your model in the :doc:`demonstration mode <../demo>` and visualize results.
The examples are provided with an object detection model, but it is easy to apply them for action classification by substituting the object detection model with classification one.
