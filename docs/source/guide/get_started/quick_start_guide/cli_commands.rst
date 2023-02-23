OpenVINO™ Training Extensions CLI commands
=================

Below, all possible OpenVINO™ Training Extensions CLI commands are presented with some general examples of how to run specific functionality. We also have :doc:`dedicated tutorials <../../tutorials/base/how_to_train/index>` in our documentation with life-practical examples on specific datasets for each task.

.. note::

    To run CLI commands we need to prepare a dataset. Each task requires specific data formats. To know more about which formats are supported by each task, refer to :doc:`explanation section <../../explanation/index>` in the documentation.

*****
Find
*****

``otx find`` lists model templates and backbones available for the given task. Specify the task name with ``--task`` option. Use ``--backbone BACKBONE`` to find the backbone from supported frameworks.

.. code-block::

    (otx) ...$ otx find --help
    usage: otx find [-h] [--task TASK] [--template] [--backbone BACKBONE [BACKBONE ...]]

    optional arguments:
      -h, --help            show this help message and exit
      --task TASK           The currently supported options: ('CLASSIFICATION', 'DETECTION', 'ROTATED_DETECTION', 'INSTANCE_SEGMENTATION', 'SEGMENTATION', 'ACTION_CLASSIFICATION', 'ACTION_DETECTION',
                            'ANOMALY_CLASSIFICATION', 'ANOMALY_DETECTION', 'ANOMALY_SEGMENTATION').
      --template            Shows a list of templates that can be used immediately.
      --backbone BACKBONE [BACKBONE ...]
                            The currently supported options: ['otx', 'mmcls', 'mmdet', 'mmseg', 'torchvision', 'pytorchcv', 'omz.mmcls'].


Example to find ready-to-use templates for the detection task:

.. code-block::

    (otx) ...$ otx find --task detection
    +-----------+-----------------------------------+-------+---------------------------------------------------------------------------+
    |    TASK   |                 ID                |  NAME |                                 BASE PATH                                 |
    +-----------+-----------------------------------+-------+---------------------------------------------------------------------------+
    | DETECTION | Custom_Object_Detection_Gen3_ATSS |  ATSS | otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml |
    | DETECTION |  Custom_Object_Detection_Gen3_SSD |  SSD  |  otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml |
    | DETECTION |   Custom_Object_Detection_YOLOX   | YOLOX | otx/algorithms/detection/configs/detection/cspdarknet_yolox/template.yaml |
    +-----------+-----------------------------------+-------+---------------------------------------------------------------------------+


Example to find supported torchvision backbones for the detection task:

.. code-block::

    (otx) ...$ otx find --task detection --backbone torchvision
    +-------+--------------------------------+---------------+---------+
    | Index |         Backbone Type          | Required-Args | Options |
    +-------+--------------------------------+---------------+---------+
    |   1   |      torchvision.alexnet       |               |         |
    |   2   |      torchvision.resnet18      |               |         |
    |   3   |      torchvision.resnet34      |               |         |
    |   4   |      torchvision.resnet50      |               |         |
    ...
    |   33  | torchvision.shufflenet_v2_x1_0 |               |         |
    |   34  | torchvision.shufflenet_v2_x1_5 |               |         |
    |   35  | torchvision.shufflenet_v2_x2_0 |               |         |
    +-------+--------------------------------+---------------+---------+



*************************
Building workspace folder
*************************

``otx build`` creates a workspace with a particular model template and all the necessary components for training, evaluation, optimization, etc. This option is also used for modifying the backbone of the model.

.. code-block::

    (otx) ...$ otx build --help
    usage: otx build [-h] [--train-data-roots TRAIN_DATA_ROOTS] [--val-data-roots VAL_DATA_ROOTS] [--test-data-roots TEST_DATA_ROOTS] [--unlabeled-data-roots UNLABELED_DATA_ROOTS]
                    [--unlabeled-file-list UNLABELED_FILE_LIST] [--task TASK] [--train-type TRAIN_TYPE] [--work-dir WORK_DIR] [--model MODEL] [--backbone BACKBONE]
                    [template]

    positional arguments:
      template              Enter the path or ID or name of the template file.
                            This can be omitted if you have train-data-roots or run inside a workspace.

    optional arguments:
      -h, --help            show this help message and exit
      --train-data-roots TRAIN_DATA_ROOTS
                            Comma-separated paths to training data folders.
      --val-data-roots VAL_DATA_ROOTS
                            Comma-separated paths to validation data folders.
      --test-data-roots TEST_DATA_ROOTS
                            Comma-separated paths to test data folders.
      --unlabeled-data-roots UNLABELED_DATA_ROOTS
                            Comma-separated paths to unlabeled data folders
      --unlabeled-file-list UNLABELED_FILE_LIST
                            Comma-separated paths to unlabeled file list
      --task TASK           The currently supported options: ('CLASSIFICATION', 'DETECTION', 'INSTANCE_SEGMENTATION', 'SEGMENTATION', 'ACTION_CLASSIFICATION', 'ACTION_DETECTION', 'ANOMALY_CLASSIFICATION', 'ANOMALY_DETECTION', 'ANOMALY_SEGMENTATION').
      --train-type TRAIN_TYPE
                            The currently supported options: dict_keys(['INCREMENTAL', 'SEMISUPERVISED', 'SELFSUPERVISED']).
      --work-dir WORK_DIR   Location where the workspace.
      --model MODEL         Enter the name of the model you want to use. (Ex. EfficientNet-B0).
      --backbone BACKBONE   Available Backbone Type can be found using 'otx find --backbone {framework}'.
                            If there is an already created backbone configuration yaml file, enter the corresponding path.


For example, the following command line will create an object detection ``Custom_Object_Detection_Gen3_ATSS`` model template with ResNet backbone from `mmdetection <https://github.com/open-mmlab/mmdetection>`_:
To learn more about backbone replacement, please refer to the :doc:`following advanced tutorial <../../tutorials/advanced/backbones>`.

.. code-block::

    (otx) ...$ otx build Custom_Object_Detection_Gen3_ATSS --backbone mmdet.ResNet --train-data-roots <path/to/train/root> --val-data-roots <path/to/val/root>

----------------
Dataset handling
----------------

If the train dataset root and validation dataset root are the same - pass the same path to both options. For example, you have a standard COCO format for object detection:

.. code-block::

    coco_data_root
      |---- annotations
        |---- instances_train.json
        |---- instances_val.json
      |---- images
        |---- train
          |---- 000.jpg
          ....
      |---- val
          |---- 000.jpg
          ....


Then pass the path to ``coco_data_root`` to both root options:

.. code-block::

  --train-data-roots coco_data_root --val-data-roots coco_data_root

However, if you store your training set and validation separately - provide paths to both accordingly.
OpenVINO™ Training Extensions supports also auto-split functionality. If you don't have a prepared validation set - the Datumaro manager will run a random auto-split and will save the final dataset to ``splitted_dataset`` folder inside the workspace folder. This split can be further used for training.

.. note::

    Not all of the tasks support the auto-split feature. If the task isn't supported - unexpected behavior or errors may appear. Please, refer to :doc:`auto-configuration <../../explanation/additional_features/auto_configuration>` documentation.


*********
Training
*********

``otx train`` trains a model (a particular model template) on a dataset and saves results in two files:

- ``weights.pth`` - a model snapshot
- ``label_schema.json`` - a label schema used in training, created from a dataset

The results will be saved in ``./model`` folder by default. The output folder can be modified by ``--save-model-to`` option. These files are used by other commands: ``export``, ``eval``, ``demo``, etc.

``otx train`` receives ``template`` as a positional argument. ``template`` can be a path to the specific ``template.yaml`` file, template name or template ID. Also, the path to train and val data root should be passed to the CLI to start training.

However, if you created a workspace with ``otx build``, the training process can be started (in the workspace directory) just with ``otx train`` command without any additional options. OpenVINO™ Training Extensions will fetch everything else automatically.

.. code-block::

    otx train --help
    usage: otx train [-h] [--train-data-roots TRAIN_DATA_ROOTS] [--val-data-roots VAL_DATA_ROOTS] [--unlabeled-data-roots UNLABELED_DATA_ROOTS] [--unlabeled-file-list UNLABELED_FILE_LIST]
                    [--load-weights LOAD_WEIGHTS] [--resume-from RESUME_FROM] [--save-model-to SAVE_MODEL_TO] [--work-dir WORK_DIR] [--enable-hpo] [--hpo-time-ratio HPO_TIME_RATIO] [--gpus GPUS]
                    [--rdzv-endpoint RDZV_ENDPOINT] [--base-rank BASE_RANK] [--world-size WORLD_SIZE] [--data DATA]
                    [template] {params} ...

    positional arguments:
      template              Enter the path or ID or name of the template file.
                            This can be omitted if you have train-data-roots or run inside a workspace.
      {params}              sub-command help
        params              Hyper parameters defined in template file.

    optional arguments:
      -h, --help            show this help message and exit
      --train-data-roots TRAIN_DATA_ROOTS
                            Comma-separated paths to training data folders.
      --val-data-roots VAL_DATA_ROOTS
                            Comma-separated paths to validation data folders.
      --unlabeled-data-roots UNLABELED_DATA_ROOTS
                            Comma-separated paths to unlabeled data folders
      --unlabeled-file-list UNLABELED_FILE_LIST
                            Comma-separated paths to unlabeled file list
      --load-weights LOAD_WEIGHTS
                            Load model weights from previously saved checkpoint.
      --resume-from RESUME_FROM
                            Resume training from previously saved checkpoint
      --save-model-to SAVE_MODEL_TO
                            Location where trained model will be stored.
      --work-dir WORK_DIR   Location where the intermediate output of the training will be stored.
      --enable-hpo          Execute hyper parameters optimization (HPO) before training.
      --hpo-time-ratio HPO_TIME_RATIO
                            Expected ratio of total time to run HPO to time taken for full fine-tuning.
      --gpus GPUS           Comma-separated indices of GPU.               If there are more than one available GPU, then model is trained with multi GPUs.
      --rdzv-endpoint RDZV_ENDPOINT
                            Rendezvous endpoint for multi-node training.
      --base-rank BASE_RANK
                            Base rank of the current node workers.
      --world-size WORLD_SIZE
                            Total number of workers in a worker group.
      --data DATA           The data.yaml path want to use in train task.



Example of the command line to start object detection training:

.. code-block::

    (otx) ...$ otx train SSD  --train-data-roots <path/to/train/root> --val-data-roots <path/to/val/root>


.. note::
  We also can visualize the training using ``Tensorboard`` as these logs are located in ``<work_dir>/tf_logs``.

It is also possible to start training by omitting the template and just passing the paths to dataset roots, then the :doc:`auto-configuration <../../explanation/additional_features/auto_configuration>` will be enabled. Based on the dataset, OpenVINO™ Training Extensions will choose the task type and template with the best accuracy/speed trade-off.

We also can modify model template-specific parameters through the command line. To print all the available parameters the following command can be executed:

.. code-block::

    (otx) ...$ otx train TEMPLATE params --help



For example, that is how we can change the learning rate and the batch size for the SSD model:

.. code-block::

    (otx) ...$ otx train SSD --train-data-roots <path/to/train/root> \
                             --val-data-roots <path/to/val/root> \
                             params \
                             --learning_parameters.batch_size 16 \
                             --learning_parameters.learning_rate 0.001


As can be seen from the parameters list, the model can be trained using multiple GPUs. To do so, you simply need to specify a comma-separated list of GPU indices after the ``--gpus`` argument. It will start the distributed data-parallel training with the GPUs you have specified.

.. note::

    Multi-GPU training is currently supported for all tasks except for action tasks and semi/self-supervised learning methods. We'll add support for them in the near future.

**********
Exporting
**********

``otx export`` exports a trained model to the OpenVINO™ IR format to efficiently run it on Intel hardware.

With the ``--help`` command, you can list additional information, such as its parameters common to all model templates:

.. code-block::

    (otx) ...$ otx export --help
    usage: otx export [-h] [--load-weights LOAD_WEIGHTS] [--save-model-to SAVE_MODEL_TO] [template]

    positional arguments:
      template              Enter the path or ID or name of the template file.
                            This can be omitted if you have train-data-roots or run inside a workspace.

    optional arguments:
      -h, --help            show this help message and exit
      --load-weights LOAD_WEIGHTS
                            Load model weights from previously saved checkpoint.
      --save-model-to SAVE_MODEL_TO
                            Location where exported model will be stored.


The command below performs exporting to the ``outputs/openvino`` path.

.. code-block::

    (otx) ...$ otx export Custom_Object_Detection_Gen3_SSD --load-weights <path/to/trained/weights.pth> --save-model-to outputs/openvino

The command results in ``openvino.xml``, ``openvino.bin`` and ``label_schema.json``


************
Optimization
************

``otx optimize`` optimizes a model using `NNCF <https://github.com/openvinotoolkit/nncf>`_ or `POT <https://docs.openvino.ai/latest/pot_introduction.html>`_ depending on the model format.

- NNCF optimization used for trained snapshots in a framework-specific format such as checkpoint (.pth) file from Pytorch
- POT optimization used for models exported in the OpenVINO™ IR format

With the ``--help`` command, you can list additional information:

.. code-block::

    usage: otx optimize [-h] [--train-data-roots TRAIN_DATA_ROOTS] [--val-data-roots VAL_DATA_ROOTS] [--load-weights LOAD_WEIGHTS] [--save-model-to SAVE_MODEL_TO] [--save-performance SAVE_PERFORMANCE]
                        [--work-dir WORK_DIR]
                        [template] {params} ...

    positional arguments:
      template              Enter the path or ID or name of the template file.
                            This can be omitted if you have train-data-roots or run inside a workspace.
      {params}              sub-command help
        params              Hyper parameters defined in template file.

    optional arguments:
      -h, --help            show this help message and exit
      --train-data-roots TRAIN_DATA_ROOTS
                            Comma-separated paths to training data folders.
      --val-data-roots VAL_DATA_ROOTS
                            Comma-separated paths to validation data folders.
      --load-weights LOAD_WEIGHTS
                            Load weights of trained model
      --save-model-to SAVE_MODEL_TO
                            Location where trained model will be stored.
      --save-performance SAVE_PERFORMANCE
                            Path to a json file where computed performance will be stored.
      --work-dir WORK_DIR   Location where the intermediate output of the task will be stored.

Command example for optimizing a PyTorch model (.pth) with OpenVINO™ NNCF:

.. code-block::

    (otx) ...$ otx optimize SSD --load-weights <path/to/trained/weights.pth> \
                                --train-data-roots <path/to/train/root> \
                                --val-data-roots <path/to/val/root> \
                                --save-model-to outputs/nncf


Command example for optimizing OpenVINO™ model (.xml) with OpenVINO™ POT:

.. code-block::

    (otx) ...$ otx optimize SSD --load-weights <path/to/openvino.xml> \
                                --val-data-roots <path/to/val/root> \
                                --save-model-to outputs/pot


Thus, to use POT pass the path to exported IR (.xml) model, to use NNCF pass the path to the PyTorch (.pth) weights.


***********
Evaluation
***********

``otx eval`` runs the evaluation of a model on the specific dataset.

With the ``--help`` command, you can list additional information, such as its parameters common to all model templates:

.. code-block::

    (otx) ...$ otx eval --help
    usage: otx eval [-h] [--test-data-roots TEST_DATA_ROOTS] [--load-weights LOAD_WEIGHTS] [--save-performance SAVE_PERFORMANCE] [--work-dir WORK_DIR] [template] {params} ...

    positional arguments:
      template              Enter the path or ID or name of the template file.
                            This can be omitted if you have train-data-roots or run inside a workspace.
      {params}              sub-command help
        params              Hyper parameters defined in template file.

    optional arguments:
      -h, --help            show this help message and exit
      --test-data-roots TEST_DATA_ROOTS
                            Comma-separated paths to test data folders.
      --load-weights LOAD_WEIGHTS
                            Load model weights from previously saved checkpoint.It could be a trained/optimized model (POT only) or exported model.
      --save-performance SAVE_PERFORMANCE
                            Path to a json file where computed performance will be stored.
      --work-dir WORK_DIR   Location where the intermediate output of the task will be stored.


The command below will evaluate the trained model on the provided dataset:

.. code-block::

    (otx) ...$ otx eval SSD --test-data-roots <path/to/test/root> \
                            --load-weights <path/to/model_weghts> \
                            --save-performance outputs/performance.json

.. note::

    It is possible to pass both PyTorch weights ``.pth`` or OpenVINO™ IR ``openvino.xml`` to ``--load-weights`` option.


***********
Explanation
***********

``otx explain`` runs the explanation algorithm of a model on the specific dataset. It helps explain the model's decision-making process in a way that is easily understood by humans.

With the ``--help`` command, you can list additional information, such as its parameters common to all model templates:

.. code-block::

    (otx) ...$ otx explain --help
    usage: otx explain [-h] --explain-data-roots EXPLAIN_DATA_ROOTS [--save-explanation-to SAVE_EXPLANATION] --load-weights LOAD_WEIGHTS [--explain-algorithm EXPLAIN_ALGORITHM] [--overlay-weight OVERLAY_WEIGHT] [template] {params} ...

    positional arguments:
      template              Enter the path or ID or name of the template file.
                            This can be omitted if you have train-data-roots or run inside a workspace.
      {params}              sub-command help
        params              Hyper parameters defined in template file.

    optional arguments:
      -h, --help            show this help message and exit
      --explain-data-roots EXPLAIN_DATA_ROOTS
                            Comma-separated paths to explain data folders.
      --save-explanation-to SAVE_EXPLANATION_TO
                            Output path for explanation images.
      --load-weights LOAD_WEIGHTS
                            Load model weights from previously saved checkpoint.
      --explain-algorithm EXPLAIN_ALGORITHM
                            Explain algorithm name, currently support ['activationmap', 'eigencam', 'classwisesaliencymap']. For Openvino task, default method will be selected.
      --overlay-weight OVERLAY_WEIGHT
                            Weight of the saliency map when overlaying the saliency map.


The command below will generate saliency maps (heatmaps with read colored areas of focus) of the trained model on the provided dataset and save the resulting images to ``save-explanation-to`` path:

.. code-block::

    (otx) ...$ otx explain SSD --explain-data-roots <path/to/explain/root> \
                               --load-weights <path/to/model_weights> \
                               --save-explanation-to <path/to/output/root> \
                               --explain-algorithm classwisesaliencymap \
                               --overlay-weight 0.5

.. note::

    It is possible to pass both PyTorch weights ``.pth`` or OpenVINO™ IR ``openvino.xml`` to ``--load-weights`` option.


*************
Demonstration
*************

``otx demo`` runs model inference on images, videos, or webcam streams to show how it works with the user's data.

.. note::

  ``otx demo`` command requires GUI backend to your system for displaying inference results.

  Only the OpenVINO™ IR model can be used for the ``otx demo`` command.

.. code-block::

    (otx) ...$ otx demo --help
    usage: otx demo [-h] -i INPUT --load-weights LOAD_WEIGHTS [--fit-to-size FIT_TO_SIZE FIT_TO_SIZE] [--loop] [--delay DELAY] [--display-perf] [template] {params} ...

    positional arguments:
      template              Enter the path or ID or name of the template file.
                            This can be omitted if you have train-data-roots or run inside a workspace.
      {params}              sub-command help
        params              Hyper parameters defined in template file.

    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT, --input INPUT
                            Source of input data: images folder, image, webcam and video.
      --load-weights LOAD_WEIGHTS
                            Load model weights from previously saved checkpoint.It could be a trained/optimized model (POT only) or exported model.
      --fit-to-size FIT_TO_SIZE FIT_TO_SIZE
                            Width and Height space-separated values. Fits displayed images to window with specified Width and Height. This options applies to result visualisation only.
      --loop                Enable reading the input in a loop.
      --delay DELAY         Frame visualization time in ms.
      --display-perf        This option enables writing performance metrics on displayed frame. These metrics take into account not only model inference time, but also frame reading, pre-processing and post-processing.


Command example of the demonstration:

.. code-block::

    (otx) ...$ otx demo SSD --input INPUT \
                            --load-weights <path/to/openvino.xml> \
                            --display-perf \
                            --delay 1000


Input can be a folder with images, a single image, a webcam ID or a video. The inference results of a model will be displayed to the GUI window with a 1-second interval.

.. note::
 
  If you execute this command from the remote environment (e.g., using text-only SSH via terminal) without having remote GUI client software, you can meet some error messages from this command.


***********
Deployment
***********

``otx deploy`` creates ``openvino.zip`` with a self-contained python package, a demo application, and an exported model. As follows from the zip archive name, the ``deploy`` can be used only with the OpenVINO™ IR model.

With the ``--help`` command, you can list additional information, such as its parameters common to all model templates:

.. code-block::

    (otx) ...$ otx deploy --help
    usage: otx deploy [-h] [--load-weights LOAD_WEIGHTS] [--save-model-to SAVE_MODEL_TO] [template]

    positional arguments:
      template              Enter the path or ID or name of the template file.
                            This can be omitted if you have train-data-roots or run inside a workspace.

    optional arguments:
      -h, --help            show this help message and exit
      --load-weights LOAD_WEIGHTS
                            Load model weights from previously saved checkpoint.
      --save-model-to SAVE_MODEL_TO
                            Location where openvino.zip will be stored.


Command example:

.. code-block::

    (otx) ...$ otx deploy SSD --load-weights <path/to/openvino.xml> \
                              --save-model-to outputs/deploy

