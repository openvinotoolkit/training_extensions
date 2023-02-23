Anomaly Detection Tutorial
================================

This tutorial demonstrates how to train, evaluate, and deploy a classification, detection, or segmentation model for anomaly detection in industrial or medical applications. Read :doc:`../../../explanation/algorithms/anomaly/index` for more information about the Anomaly tasks.

.. note::

    To learn how to deploy the trained model, refer to: :doc:`../deploy`.

    To learn how to run the demo and visualize results, refer to: :doc:`../demo`.

The process has been tested with the following configuration:

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-10980XE
- CUDA Toolkit 11.1


*****************************
Setup the Virtual environment
*****************************

To create a universal virtual environment for OpenVINO™ Training Extensions, please follow the installation process in the :doc:`quick start guide <../../../get_started/quick_start_guide/installation>`. 

Alternatively, if you want to only train anomaly models then you can create a task specific environment.

1. Ensure that you have python 3 installed on your system. You can check this by running.

    .. code-block:: bash

        python3 --version

    It should give a similar result::

        Python 3.8.16

2. Create a virtual environment and activate it.

    .. code-block:: bash

        python3 -m venv anomaly_env
        source anomaly_env/bin/activate

3. Install the prerequisites for OpenVINO™ Training Extensions.

    Install PyTorch according to your system environment. Refer to the `official installation guide <https://pytorch.org/get-started/previous-versions/>`_

    .. note::

        Currently, only torch==1.13.1 was fully validated. torch==2.x will be supported soon. (Earlier versions are not supported due to security issues)

    Example install command for torch==1.13.1+cu111:

    .. code-block::

        pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu111

4. Then, install the task specific OpenVINO™ Training Extensions package you can then use::

    pip install -e .[anomaly]

**************************
Dataset Preparation
**************************

For this example we will use the `MVTec <https://www.mvtec.com/company/research/datasets/mvtec-ad>`_ dataset.

You can download the dataset from the link above. We will use the ``bottle`` category for this tutorial.

This is how it might look like in your file system:

.. code-block:: bash

    datasets/MVTec/bottle
    ├── ground_truth
    │   ├── broken_large
    │   │   ├── 000_mask.png
    │   │   ├── 001_mask.png
    │   │   ├── 002_mask.png
    │   │   ...
    │   ├── broken_small
    │   │   ├── 000_mask.png
    │   │   ├── 001_mask.png
    │   │   ...
    │   └── contamination
    │       ├── 000_mask.png
    │       ├── 001_mask.png
    │       ...
    ├── license.txt
    ├── readme.txt
    ├── test
    │   ├── broken_large
    │   │   ├── 000.png
    │   │   ├── 001.png
    │   │   ...
    │   ├── broken_small
    │   │   ├── 000.png
    │   │   ├── 001.png
    │   │   ...
    │   ├── contamination
    │   │   ├── 000.png
    │   │   ├── 001.png
    │   │   ...
    │   └── good
    │       ├── 000.png
    │       ├── 001.png
    │       ...
    └── train
        └── good
            ├── 000.png
            ├── 001.png
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

    otx train ote_anomaly_detection_padim \
        --train-data-roots datasets/MVTec/bottle/train \
        --val-data-roots datasets/MVTec/bottle/test

This will start training and generate artifacts for commands such as ``export`` and ``optimize``. You will notice the ``otx-workspace-ANOMALY_DETECTION`` directory in your current working directory. This is where all the artifacts are stored.

**************
Evaluation
**************

Now that we have trained the model, let's see how it performs on the a specific dataset. In this example we will use the same dataset to generate evaluation metrics. To perform evaluation you need to run the following commands:

.. code-block:: bash

    otx eval ote_anomaly_detection_padim \
        --test-data-roots datasets/MVTec/bottle/test \
        --load-weights otx-workspace-ANOMALY_DETECTION/models/weights.pth \
        --save-performance otx-workspace-ANOMALY_DETECTION/performance.json

You should see an output similar to the following::

    MultiScorePerformance(score: 0.6356589147286821, primary_metric: ScoreMetric(name=`f-measure`, score=`0.6356589147286821`), additional_metrics: (1 metrics), dashboard: (2 metric groups))


The primary metric here is the f-measure computed against the ground-truth bounding boxes. It is also called the local score. In addition, f-measure is also used to compute the global score. The global score is computed based on global label of the image. That is, the image is anomalous if it contains at least one anomaly. This global score is stored as an additional metric.

.. note::

All task types report Image-level F-measure as the primary metric. In addition, both localization tasks (anomaly detection and anomaly segmentation) also report localization performance (F-measure for anomaly detection and Dice-coefficient for anomaly segmentation).

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
        --test-data-roots datasets/MVTec/bottle/test \
        --load-weights otx-workspace-ANOMALY_DETECTION/openvino_models/openvino.xml \
        --save-performance otx-workspace-ANOMALY_DETECTION/openvino_models/performance.json

This gives the following results::

    MultiScorePerformance(score: 0.6511627906976744, primary_metric: ScoreMetric(name=`f-measure`, score=`0.6511627906976744`), additional_metrics: (1 metrics), dashboard: (2 metric groups))

************
Optimization
************

Anomaly tasks can be optimized either in POT or NNCF format. For more information refer to the :doc:`optimization explanation <../../../explanation/additional_features/models_optimization>` section.


1. Let's start with POT optimization.

    .. code-block:: bash

        otx optimize ote_anomaly_detection_padim \
            --train-data-roots datasets/MVTec/bottle/train \
            --load-weights otx-workspace-ANOMALY_DETECTION/openvino_models/openvino.xml \
            --save-model-to otx-workspace-ANOMALY_DETECTION/pot_model

    This generates the following files that can be used to run :doc:`otx demo <../demo>`.

    - image_threshold
    - pixel_threshold
    - label_schema.json
    - max
    - min
    - openvino.bin
    - openvino.xml

2. To perform NNCF optimization, pass the torch ``pth`` weights to the ``opitmize`` command.

    .. code-block:: bash

        otx optimize ote_anomaly_detection_padim \
            --train-data-roots datasets/MVTec/bottle/train \
            --load-weights otx-workspace-ANOMALY_DETECTION/models/weights.pth \
            --save-model-to otx-workspace-ANOMALY_DETECTION/nncf_model

    Similar to POT optimization, this generates the following files.

    - image_threshold
    - pixel_threshold
    - label_schema.json
    - max
    - min
    - weights.pth


*******************************
Segmentation and Classification
*******************************

While the above example shows Anomaly Detection, you can also train Anomaly Segmentation and Classification models. To see what tasks are available, you can pass ``anomaly_segmentation`` and ``anomaly_classification`` to ``otx find`` mentioned in the `Training`_ section. You can then use the same commands to train, evaluate, export and optimize the models.

.. note::

    The Segmentation and Detection tasks also require that the ``ground_truth`` masks be present to ensure that the localization metrics are computed correctly.
    The ``ground_truth`` masks are not required for the Classification task.

