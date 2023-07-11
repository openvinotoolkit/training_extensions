Anomaly Detection Tutorial
================================

This tutorial demonstrates how to train, evaluate, and deploy a classification, detection, or segmentation model for anomaly detection in industrial or medical applications. 
Read :doc:`../../../explanation/algorithms/anomaly/index` for more information about the Anomaly tasks.

.. note::
  To learn more about managing the training process of the model including additional parameters and its modification, refer to :doc:`./detection`.

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

1. To create a universal virtual environment for OpenVINO™ Training Extensions, 
please follow the installation process in the :doc:`quick start guide <../../../get_started/quick_start_guide/installation>`. 

2. Alternatively, if you want to only train anomaly models, then you can create a task specific environment. 
Then also follow the installation process in the guide above, but substitute ``pip install -e .[anomaly]`` with the following command:

.. code-block::

    pip install -e .[anomaly]

3. Activate your virtual 
environment:

.. code-block::

  .otx/bin/activate
  # or by this line, if you created an environment, using tox
  . venv/otx/bin/activate

**************************
Dataset Preparation
**************************

1. For this example, we will use the `MVTec <https://www.mvtec.com/company/research/datasets/mvtec-ad>`_ dataset.
You can download the dataset from the link above. We will use the ``bottle`` category for this tutorial.

2. This is how it might look like in your 
file system:

.. code-block:: 

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

1. For this example let's look at the 
anomaly detection tasks

.. code-block:: bash

    (otx) ...$  otx find --task anomaly_detection

::

    +-------------------+-----------------------------+-------+--------------------------------------------------------------+
    |        TASK       |              ID             |  NAME |                          BASE PATH                           |
    +-------------------+-----------------------------+-------+--------------------------------------------------------------+
    | ANOMALY_DETECTION | ote_anomaly_detection_stfpm | STFPM | otx/algorithms/anomaly/configs/detection/stfpm/template.yaml |
    | ANOMALY_DETECTION | ote_anomaly_detection_padim | PADIM | otx/algorithms/anomaly/configs/detection/padim/template.yaml |
    +-------------------+-----------------------------+-------+--------------------------------------------------------------+

You can see two anomaly detection models, STFPM and PADIM. For more detail on each model, refer to Anomalib's `STFPM <https://openvinotoolkit.github.io/anomalib/reference_guide/algorithms/stfpm.html>`_ and `PADIM <https://openvinotoolkit.github.io/anomalib/reference_guide/algorithms/padim.html>`_ documentation.

2. Let's proceed with PADIM for 
this example. 

.. code-block:: bash

    (otx) ...$  otx train ote_anomaly_detection_padim \
                          --train-data-roots datasets/MVTec/bottle/train \
                          --val-data-roots datasets/MVTec/bottle/test

This will start training and generate artifacts for commands such as ``export`` and ``optimize``. You will notice the ``otx-workspace-ANOMALY_DETECTION`` directory in your current working directory. This is where all the artifacts are stored.

**************
Evaluation
**************

Now we have trained the model, let's see how it performs on a specific dataset. In this example, we will use the same dataset to generate evaluation metrics. To perform evaluation you need to run the following commands:

.. code-block:: bash

    (otx) ...$ otx eval ote_anomaly_detection_padim \
                        --test-data-roots datasets/MVTec/bottle/test \
                        --load-weights otx-workspace-ANOMALY_DETECTION/models/weights.pth \
                        --output otx-workspace-ANOMALY_DETECTION/outputs

You should see an output similar to the following::

    MultiScorePerformance(score: 0.6356589147286821, primary_metric: ScoreMetric(name=`f-measure`, score=`0.6356589147286821`), additional_metrics: (1 metrics), dashboard: (2 metric groups))


The primary metric here is the f-measure computed against the ground-truth bounding boxes. It is also called the local score. In addition, f-measure is also used to compute the global score. The global score is computed based on the global label of the image. That is, the image is anomalous if it contains at least one anomaly. This global score is stored as an additional metric.

.. note::

    All task types report Image-level F-measure as the primary metric. In addition, both localization tasks (anomaly detection and anomaly segmentation) also report localization performance (F-measure for anomaly detection and Dice-coefficient for anomaly segmentation).

******
Export
******

1. ``otx export`` exports a trained Pytorch `.pth` model to the OpenVINO™ Intermediate Representation (IR) format.
It allows running the model on the Intel hardware much more efficient, especially on the CPU. Also, the resulting IR model is required to run POT optimization. IR model consists of 2 files: ``openvino.xml`` for weights and ``openvino.bin`` for architecture.

2. We can run the below command line to export the trained model
and save the exported model to the ``openvino`` folder:

.. code-block::

    otx export ote_anomaly_detection_padim \
        --load-weights otx-workspace-ANOMALY_DETECTION/models/weights.pth \
        --output otx-workspace-ANOMALY_DETECTION/openvino

You will see the outputs similar to the following:

.. code-block::

    [INFO] 2023-02-21 16:42:43,207 - otx.algorithms.anomaly.tasks.inference - Initializing the task environment.
    [INFO] 2023-02-21 16:42:43,632 - otx.algorithms.anomaly.tasks.train - Loaded model weights from Task Environment
    [WARNING] 2023-02-21 16:42:43,639 - otx.algorithms.anomaly.tasks.inference - Ommitting feature dumping is not implemented.The saliency maps and representation vector outputs will be dumped in the exported model.
    [INFO] 2023-02-21 16:42:43,640 - otx.algorithms.anomaly.tasks.inference - Exporting the OpenVINO model.
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /tmp/otx-anomaliba3imqkmo/onnx_model.xml
    [ SUCCESS ] BIN file: /tmp/otx-anomaliba3imqkmo/onnx_model.bin

Now that we have the exported model, let's check its performance using ``otx eval``:

.. code-block:: bash

    otx eval ote_anomaly_detection_padim \
        --test-data-roots datasets/MVTec/bottle/test \
        --load-weights otx-workspace-ANOMALY_DETECTION/openvino/openvino.xml \
        --output otx-workspace-ANOMALY_DETECTION/openvino

This gives the following results:

.. code-block::

    MultiScorePerformance(score: 0.6511627906976744, primary_metric: ScoreMetric(name=`f-measure`, score=`0.6511627906976744`), additional_metrics: (1 metrics), dashboard: (2 metric groups))

************
Optimization
************

Anomaly tasks can be optimized either in POT or NNCF format. For more information refer to the :doc:`optimization explanation <../../../explanation/additional_features/models_optimization>` section.


1. Let's start with POT 
optimization.

.. code-block::

    otx optimize ote_anomaly_detection_padim \
        --train-data-roots datasets/MVTec/bottle/train \
        --load-weights otx-workspace-ANOMALY_DETECTION/openvino/openvino.xml \
        --output otx-workspace-ANOMALY_DETECTION/pot_model

This command generates the following files that can be used to run :doc:`otx demo <../demo>`:

- image_threshold
- pixel_threshold
- label_schema.json
- max
- min
- openvino.bin
- openvino.xml

2. To perform NNCF optimization, pass the torch ``pth`` 
weights to the ``opitmize`` command:

.. code-block:: 

    otx optimize ote_anomaly_detection_padim \
        --train-data-roots datasets/MVTec/bottle/train \
        --load-weights otx-workspace-ANOMALY_DETECTION/models/weights.pth \
        --output otx-workspace-ANOMALY_DETECTION/nncf_model

Similar to POT optimization, it generates the following files:

- image_threshold
- pixel_threshold
- label_schema.json
- max
- min
- weights.pth


*******************************
Segmentation and Classification
*******************************

While the above example shows Anomaly Detection, you can also train Anomaly Segmentation and Classification models. 
To see what tasks are available, you can pass ``anomaly_segmentation`` and ``anomaly_classification`` to ``otx find`` mentioned in the `Training`_ section. You can then use the same commands to train, evaluate, export and optimize the models.

.. note::

    The Segmentation and Detection tasks also require that the ``ground_truth`` masks be present to ensure that the localization metrics are computed correctly.
    The ``ground_truth`` masks are not required for the Classification task.

