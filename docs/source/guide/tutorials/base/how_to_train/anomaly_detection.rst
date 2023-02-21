Anomaly Detection model
================================

This tutorial demonstrates how to train, evaluate, and deploy a classification, detection, or segmentation model for anomaly detection in industrial or medical applications.

The process has been tested with the following configuration:

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-10980XE
- CUDA Toolkit 11.1


*****************************
Setup the Virtual environment
*****************************

To create a universal virtual environment for OTX, please follow the installation process in the :doc:`quick start guide <../../../get_started/quick_start_guide/installation>`. 

Alternatively, if you want to only train anomaly models then you can just use ``pip install -e .[anomaly]``

**************************
Dataset Preparation
**************************

Since the anomaly tasks in OTX depend on `Anomalib <https://github.com/openvinotoolkit/anomalib>`_, let's use a `Toy Dataset <https://openvinotoolkit.github.io/anomalib/data/hazelnut_toy.html>`_ from Anomalib to demonstrate the process.

You can download and extract the dataset by running the following commands:

.. code-block:: bash

    wget https://openvinotoolkit.github.io/anomalib/_downloads/3f2af1d7748194b18c2177a34c03a2c4/hazelnut_toy.zip
    unzip hazelnut_toy.zip
    mv hazelnut_toy data/anomaly

The last command moves the dataset to the ``data/anomaly`` directory.

This is how it should look like in your file system:

.. code-block:: bash

    data/anomaly/hazelnut_toy/
    ├── colour
    │   ├── 00.jpg
    │   ├── 01.jpg
    │   ...
    ├── crack
    │   ├── 01.jpg
    │   ...
    ├── good
    │   ├── 00.jpg
    │   ├── 01.jpg
    │   ...
    ├── LICENSE
    └── mask
        ├── colour
        │   ├── 00.jpg
        │   ├── 01.jpg
        │   ...
        └── crack
            ├── 01.jpg
            ├── 02.jpg
            ...

***************************
Training
***************************

For this example let's look at the anomaly detection tasks

.. code-block:: bash

    otx find --task anomaly_detection

::

    +-------------------+-----------------------------+-------+--------------------------------------------------------------+
    |        TASK       |              ID             |  NAME |                          BASE PATH                           |
    +-------------------+-----------------------------+-------+--------------------------------------------------------------+
    | ANOMALY_DETECTION | ote_anomaly_detection_stfpm | STFPM | otx/algorithms/anomaly/configs/detection/stfpm/template.yaml |
    | ANOMALY_DETECTION | ote_anomaly_detection_padim | PADIM | otx/algorithms/anomaly/configs/detection/padim/template.yaml |
    +-------------------+-----------------------------+-------+--------------------------------------------------------------+

You can see two anomaly detection models, STFPM and PADIM. For more detail on each model, refer to Anomalib's `STFPM <https://openvinotoolkit.github.io/anomalib/reference_guide/algorithms/stfpm.html>`_ and `PADIM <https://openvinotoolkit.github.io/anomalib/reference_guide/algorithms/padim.html>`_ documentation.

Let's proceed with PADIM for this example. 

.. code-block:: bash

    otx train ote_anomaly_detection_padim --train-data-roots data/anomaly/hazelnut_toy --val-data-roots data/anomaly/hazelnut_toy

This will start training and generate artifacts for commands such as ``export`` and ``optimize``. You will notice the ``otx-workspace-ANOMALY_DETECTION`` directory in your current working directory. This is where all the artifacts are stored.

**************
Evaluation
**************

Now that we have trained the model, let's see how it performs on the a specific dataset. In this example we will use the same dataset to generate evaluation metrics. To perform evaluation you need to run the following commands:

.. code-block:: bash

    otx eval ote_anomaly_detection_padim \
        --test-data-roots data/anomaly/hazelnut_toy \
        --load-weights otx-workspace-ANOMALY_DETECTION/models/weights.pth \
        --save-performance otx-workspace-ANOMALY_DETECTION/performance.json

You should see an output similar to the following::

    MultiScorePerformance(score: 0.9032258064516128, primary_metric: None, additional_metrics: (1 metrics), dashboard: (1 metric groups))

******
Export
******

1. ``otx export`` exports a trained Pytorch `.pth` model to the OpenVINO™ Intermediate Representation (IR) format.
It allows running the model on the Intel hardware much more efficient, especially on the CPU. Also, the resulting IR model is required to run POT optimization. IR model consists of 2 files: ``openvino.xml`` for weights and ``openvino.bin`` for architecture.

2. We can run the below command line to export the trained model
and save the exported model to the ``openvino_models`` folder.

.. code-block::

    otx export ote_anomaly_detection_padim \
        --load-weights otx-workspace-ANOMALY_DETECTION/models/weights.pth \
        --save-model-to otx-workspace-ANOMALY_DETECTION/openvino_models

You will see the outputs similar to the following::

    [INFO] 2023-02-21 16:42:43,207 - otx.algorithms.anomaly.tasks.inference - Initializing the task environment.
    [INFO] 2023-02-21 16:42:43,632 - otx.algorithms.anomaly.tasks.train - Loaded model weights from Task Environment
    [WARNING] 2023-02-21 16:42:43,639 - otx.algorithms.anomaly.tasks.inference - Ommitting feature dumping is not implemented.The saliency maps and representation vector outputs will be dumped in the exported model.
    [INFO] 2023-02-21 16:42:43,640 - otx.algorithms.anomaly.tasks.inference - Exporting the OpenVINO model.
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /tmp/otx-anomaliba3imqkmo/onnx_model.xml
    [ SUCCESS ] BIN file: /tmp/otx-anomaliba3imqkmo/onnx_model.bin

Now that we have the exported model, let's check its performance using ``otx eval``

.. code-block:: bash

    otx eval ote_anomaly_detection_padim \
        --test-data-roots data/anomaly/hazelnut_toy \
        --load-weights otx-workspace-ANOMALY_DETECTION/openvino_models/openvino.xml \
        --save-performance otx-workspace-ANOMALY_DETECTION/openvino_models/performance.json

This gives the following results::

    MultiScorePerformance(score: 0.8974358974358974, primary_metric: None, additional_metrics: (1 metrics), dashboard: (1 metric groups))

************
Optimization
************

Anomaly tasks can be optimized either in POT or NNCF format. For more information refer to the :doc:`optimization explanation <../../../explanation/additional_features/models_optimization>` section.


----------------
POT Optimization
----------------

Let's start with POT optimization.

.. code-block:: bash

    otx optimize ote_anomaly_detection_padim \
        --train-data-roots data/anomaly/hazelnut_toy/ \
        --load-weights otx-workspace-ANOMALY_DETECTION/openvino_models/openvino.xml \
        --save-model-to otx-workspace-ANOMALY_DETECTION/pot_model

-----------------
NNCF Optimization
-----------------

To perform NNCF optimization, pass the torch ``pth`` weights to the ``opitmize`` command.

.. code-block:: bash

    otx optimize ote_anomaly_detection_padim \
        --train-data-roots data/anomaly/hazelnut_toy/ \
        --load-weights otx-workspace-ANOMALY_DETECTION/models/weights.pth \
        --save-model-to otx-workspace-ANOMALY_DETECTION/nncf_model


*******************************
Segmentation and Classification
*******************************

While the above example shows Anomaly Detection, you can also train Anomaly Segmentation and Classification models. To see what tasks are available, you can pass ``anomaly_segmentation`` and ``anomaly_classification`` to ``otx find`` mentioned in the `Training`_ section. You can then use the same commands to train, evaluate, export and optimize the models.