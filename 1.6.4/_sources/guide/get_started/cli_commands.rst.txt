OpenVINO™ Training Extensions CLI commands
==========================================

All possible OpenVINO™ Training Extensions CLI commands are presented below along with some general examples of how to run specific functionality. There are :doc:`dedicated tutorials <../tutorials/base/how_to_train/index>` in our documentation with life-practical examples on specific datasets for each task.

.. note::

    To run CLI commands you need to prepare a dataset. Each task requires specific data formats. To know more about which formats are supported by each task, refer to :doc:`explanation section <../explanation/algorithms/index>` in the documentation.

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
    +-----------+-----------------------------------+------------------+------------------------------------------------------------------------------------+
    |    TASK   |                 ID                |       NAME       |                                     BASE PATH                                      |
    +-----------+-----------------------------------+------------------+------------------------------------------------------------------------------------+
    | DETECTION | Custom_Object_Detection_Gen3_ATSS | MobileNetV2-ATSS |   src/otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml    |
    | DETECTION |  Object_Detection_ResNeXt101_ATSS | ResNeXt101-ATSS  |    src/otx/algorithms/detection/configs/detection/resnext101_atss/template.yaml    |
    | DETECTION |  Custom_Object_Detection_Gen3_SSD |       SSD        |    src/otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml    |
    | DETECTION |      Object_Detection_YOLOX_L     |     YOLOX-L      |  src/otx/algorithms/detection/configs/detection/cspdarknet_yolox_l/template.yaml   |
    | DETECTION |      Object_Detection_YOLOX_S     |     YOLOX-S      |  src/otx/algorithms/detection/configs/detection/cspdarknet_yolox_s/template.yaml   |
    | DETECTION |   Custom_Object_Detection_YOLOX   |    YOLOX-TINY    | src/otx/algorithms/detection/configs/detection/cspdarknet_yolox_tiny/template.yaml |
    | DETECTION |      Object_Detection_YOLOX_X     |     YOLOX-X      |  src/otx/algorithms/detection/configs/detection/cspdarknet_yolox_x/template.yaml   |
    +-----------+-----------------------------------+------------------+------------------------------------------------------------------------------------+

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
                    [--unlabeled-file-list UNLABELED_FILE_LIST] [--task TASK] [--train-type TRAIN_TYPE] [--workspace WORKSPACE] [--model MODEL] [--backbone BACKBONE]
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
                            The currently supported options: dict_keys(['Incremental', 'Semisupervised', 'Selfsupervised']).
      --workspace WORKSPACE   Location where the workspace.
      --model MODEL         Enter the name of the model you want to use. (Ex. EfficientNet-B0).
      --backbone BACKBONE   Available Backbone Type can be found using 'otx find --backbone {framework}'.
                            If there is an already created backbone configuration yaml file, enter the corresponding path.
      --deterministic       Set deterministic to True, default=False.
      --seed SEED           Set seed for configuration.


For example, the following command line will create an object detection ``Custom_Object_Detection_Gen3_ATSS`` model template with ResNet backbone from `mmdetection <https://github.com/open-mmlab/mmdetection>`_:
To learn more about backbone replacement, please refer to the :doc:`following advanced tutorial <../tutorials/advanced/backbones>`.

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

    Not all of the tasks support the auto-split feature. If the task isn't supported - unexpected behavior or errors may appear. Please, refer to :doc:`auto-configuration <../explanation/additional_features/auto_configuration>` documentation.

If you have multiple annotation files like below, add additional argument (``--train-ann-files``). Then, you could use the annotation what you selected.
OpenVINO™ Training Extensions could randomly selects the train annotation file if you do not use additional argument (``--train-ann-files``)

.. code-block::

  coco_data_root
    |---- annotations
      |---- instances_train.json
      |---- instances_train_1percent.json
      |---- instances_train_10percent.json
      |---- instances_val.json
    |---- images
      |---- train
        |---- 000.jpg
        ....
    |---- val
        |---- 000.jpg
        ....

.. code-block::

  --train-data-roots coco_data_root --train-ann-files coco_data_root/annotations/instances_train_10percent.json

.. note::

   For now, only COCO format data could be used for direct annotation input

*********
Training
*********

``otx train`` trains a model (a particular model template) on a dataset and saves results in two files:

- ``weights.pth`` - a model snapshot
- ``label_schema.json`` - a label schema used in training, created from a dataset

The results will be saved in ``./outputs/`` folder by default. The output folder can be modified by ``--output`` option. These files are used by other commands: ``export``, ``eval``, ``demo``, etc.

``otx train`` receives ``template`` as a positional argument. ``template`` can be a path to the specific ``template.yaml`` file, template name or template ID. Also, the path to train and val data root should be passed to the CLI to start training.

However, if you created a workspace with ``otx build``, the training process can be started (in the workspace directory) just with ``otx train`` command without any additional options. OpenVINO™ Training Extensions will fetch everything else automatically.

.. code-block::

    otx train --help
    usage: otx train [-h] [--train-data-roots TRAIN_DATA_ROOTS] [--val-data-roots VAL_DATA_ROOTS] [--unlabeled-data-roots UNLABELED_DATA_ROOTS] [--unlabeled-file-list UNLABELED_FILE_LIST]
                    [--load-weights LOAD_WEIGHTS] [--resume-from RESUME_FROM] [-o OUTPUT] [--workspace WORKSPACE] [--enable-hpo] [--hpo-time-ratio HPO_TIME_RATIO] [--gpus GPUS]
                    [--rdzv-endpoint RDZV_ENDPOINT] [--base-rank BASE_RANK] [--world-size WORLD_SIZE] [--mem-cache-size PARAMS.ALGO_BACKEND.MEM_CACHE_SIZE] [--data DATA]
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
      --train-type TRAIN_TYPE
                            The currently supported options: dict_keys(['Incremental', 'Semisupervised', 'Selfsupervised']).
      --load-weights LOAD_WEIGHTS
                            Load model weights from previously saved checkpoint.
      --resume-from RESUME_FROM
                            Resume training from previously saved checkpoint
      -o OUTPUT, --output OUTPUT
                            Location where trained model will be stored.
      --workspace WORKSPACE   Location where the intermediate output of the training will be stored.
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
      --mem-cache-size PARAMS.ALGO_BACKEND.MEM_CACHE_SIZE
                            Size of memory pool for caching decoded data to load data faster. For example, you can use digits for bytes size (e.g. 1024) or a string with size units (e.g. 7KiB = 7 * 2^10, 3MB = 3 * 10^6, and 2G = 2 * 2^30).
      --deterministic       Set deterministic to True, default=False.
      --seed SEED           Change seed for training.
      --data DATA           The data.yaml path want to use in train task.



Example of the command line to start object detection training:

.. code-block::

    (otx) ...$ otx train SSD  --train-data-roots <path/to/train/root> --val-data-roots <path/to/val/root>


.. note::
  You also can visualize the training using ``Tensorboard`` as these logs are located in ``<work_dir>/tf_logs``.

.. note::
  ``--mem-cache-size`` provides in-memory caching for decoded images in main memory.
  If the batch size is large, such as for classification tasks, or if your dataset contains high-resolution images,
  image decoding can account for a non-negligible overhead in data pre-processing.
  This option can be useful for maximizing GPU utilization and reducing model training time in those cases.
  If your machine has enough main memory, we recommend increasing this value as much as possible.
  For example, you can cache approximately 10,000 of ``500x375~500x439`` sized images with ``--mem-cache-size=8GB``.

It is also possible to start training by omitting the template and just passing the paths to dataset roots, then the :doc:`auto-configuration <../explanation/additional_features/auto_configuration>` will be enabled. Based on the dataset, OpenVINO™ Training Extensions will choose the task type and template with the best accuracy/speed trade-off.

You also can modify model template-specific parameters through the command line. To print all the available parameters the following command can be executed:

.. code-block::

    (otx) ...$ otx train TEMPLATE params --help



For example, that is how you can change the learning rate and the batch size for the SSD model:

.. code-block::

    (otx) ...$ otx train SSD --train-data-roots <path/to/train/root> \
                             --val-data-roots <path/to/val/root> \
                             params \
                             --learning_parameters.batch_size 16 \
                             --learning_parameters.learning_rate 0.001

You could also enable storage caching to boost data loading at the expanse of storage:

.. code-block::

    (otx) ...$ otx train SSD --train-data-roots <path/to/train/root> \
                             --val-data-roots <path/to/val/root> \
                             params \
                             --algo_backend.storage_cache_scheme JPEG/75

.. note::
  Not all templates support stroage cache. We are working on extending supported templates.


As can be seen from the parameters list, the model can be trained using multiple GPUs. To do so, you simply need to specify a comma-separated list of GPU indices after the ``--gpus`` argument. It will start the distributed data-parallel training with the GPUs you have specified.

.. note::

    Multi-GPU training is currently supported for all tasks except for action tasks. We'll add support for them in the near future.

**********
Exporting
**********

``otx export`` exports a trained model to the OpenVINO™ IR format to efficiently run it on Intel hardware.

With the ``--help`` command, you can list additional information, such as its parameters common to all model templates:

.. code-block::

    (otx) ...$ otx export --help
    usage: otx export [-h] [--load-weights LOAD_WEIGHTS] [-o OUTPUT] [--workspace WORKSPACE] [--dump-features] [--half-precision] [template]

    positional arguments:
      template              Enter the path or ID or name of the template file.
                            This can be omitted if you have train-data-roots or run inside a workspace.

    optional arguments:
      -h, --help            show this help message and exit
      --load-weights LOAD_WEIGHTS
                            Load model weights from previously saved checkpoint.
      -o OUTPUT, --output OUTPUT
                            Location where exported model will be stored.
      --workspace WORKSPACE   Location where the intermediate output of the export will be stored.
      --dump-features       Whether to return feature vector and saliency map for explanation purposes.
      --half-precision      This flag indicated if model is exported in half precision (FP16).


The command below performs exporting to the ``outputs/openvino`` path.

.. code-block::

    (otx) ...$ otx export Custom_Object_Detection_Gen3_SSD --load-weights <path/to/trained/weights.pth> --output outputs/openvino

The command results in ``openvino.xml``, ``openvino.bin`` and ``label_schema.json``

To use the exported model as an input for ``otx explain``, please dump additional outputs with internal information, using ``--dump-features``:

.. code-block::

    (otx) ...$ otx export Custom_Object_Detection_Gen3_SSD --load-weights <path/to/trained/weights.pth> --output outputs/openvino/with_features --dump-features


************
Optimization
************

``otx optimize`` optimizes a model using `NNCF <https://github.com/openvinotoolkit/nncf>`_ or `PTQ <https://github.com/openvinotoolkit/nncf#post-training-quantization>`_ depending on the model and transforms it to ``INT8`` format.

- NNCF optimization used for trained snapshots in a framework-specific format such as checkpoint (.pth) file from Pytorch
- PTQ optimization used for models exported in the OpenVINO™ IR format

With the ``--help`` command, you can list additional information:

.. code-block::

    usage: otx optimize [-h] [--train-data-roots TRAIN_DATA_ROOTS] [--val-data-roots VAL_DATA_ROOTS] [--load-weights LOAD_WEIGHTS] [-o OUTPUT]
                        [--workspace WORKSPACE]
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
      -o OUTPUT, --output OUTPUT
                            Location where optimized model will be stored.
      --workspace WORKSPACE   Location where the intermediate output of the task will be stored.

Command example for optimizing a PyTorch model (.pth) with OpenVINO™ NNCF:

.. code-block::

    (otx) ...$ otx optimize SSD --load-weights <path/to/trained/weights.pth> \
                                --train-data-roots <path/to/train/root> \
                                --val-data-roots <path/to/val/root> \
                                --output outputs/nncf


Command example for optimizing OpenVINO™ model (.xml) with OpenVINO™ PTQ:

.. code-block::

    (otx) ...$ otx optimize SSD --load-weights <path/to/openvino.xml> \
                                --val-data-roots <path/to/val/root> \
                                --output outputs/ptq


Thus, to use PTQ pass the path to exported IR (.xml) model, to use NNCF pass the path to the PyTorch (.pth) weights.


***********
Evaluation
***********

``otx eval`` runs the evaluation of a model on the specific dataset.

With the ``--help`` command, you can list additional information, such as its parameters common to all model templates:

.. code-block::

    (otx) ...$ otx eval --help
    usage: otx eval [-h] [--test-data-roots TEST_DATA_ROOTS] [--load-weights LOAD_WEIGHTS] [-o OUTPUT] [--workspace WORKSPACE] [template] {params} ...

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
                            Load model weights from previously saved checkpoint. It could be a trained/optimized model (with PTQ only) or exported model.
      -o OUTPUT, --output OUTPUT
                            Location where the intermediate output of the task will be stored.
      --workspace WORKSPACE   Path to the workspace where the command will run.


The command below will evaluate the trained model on the provided dataset:

.. code-block::

    (otx) ...$ otx eval SSD --test-data-roots <path/to/test/root> \
                            --load-weights <path/to/model_weghts> \
                            --output <path/to/outputs>

.. note::

    It is possible to pass both PyTorch weights ``.pth`` or OpenVINO™ IR ``openvino.xml`` to ``--load-weights`` option.


***********
Explanation
***********

``otx explain`` runs the explainable AI (XAI) algorithm on a specific model-dataset pair. It helps explain the model's decision-making process in a way that is easily understood by humans.

With the ``--help`` command, you can list additional information, such as its parameters common to all model templates:

.. code-block::

    (otx) ...$ otx explain --help
    usage: otx explain [-h] --input INPUT [--output OUTPUT] --load-weights LOAD_WEIGHTS [--explain-algorithm EXPLAIN_ALGORITHM] [--overlay-weight OVERLAY_WEIGHT] [template] {params} ...

    positional arguments:
      template              Enter the path or ID or name of the template file.
                            This can be omitted if you have train-data-roots or run inside a workspace.
      {params}              sub-command help
        params              Hyper parameters defined in template file.

    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT, --input INPUT
                            Comma-separated paths to explain data folders.
      -o OUTPUT, --output OUTPUT
                            Output path for explanation images.
      --load-weights LOAD_WEIGHTS
                            Load model weights from previously saved checkpoint.
      --explain-algorithm EXPLAIN_ALGORITHM
                            Explain algorithm name, currently support ['activationmap', 'eigencam', 'classwisesaliencymap']. For Openvino task, default method will be selected.
      --process-saliency-maps PROCESS_SALIENCY_MAPS
                            Processing of saliency map includes (1) resizing to input image resolution and (2) applying a colormap. Depending on the number of targets to explain, this might take significant time.
      --explain-all-classes EXPLAIN_ALL_CLASSES
                            Provides explanations for all classes. Otherwise, explains only predicted classes. This feature is supported by algorithms that can generate explanations per each class.
      --overlay-weight OVERLAY_WEIGHT
                            Weight of the saliency map when overlaying the input image with saliency map.


The command below will generate saliency maps (heatmaps with red colored areas of focus) of the trained model on the provided dataset and save the resulting images to ``output`` path:

.. code-block::

    (otx) ...$ otx explain SSD --input <path/to/explain/root> \
                               --load-weights <path/to/model_weights> \
                               --output <path/to/output/root> \
                               --explain-algorithm classwisesaliencymap \
                               --overlay-weight 0.5

.. note::

    It is possible to pass both PyTorch weights ``.pth`` or OpenVINO™ IR ``openvino.xml`` to ``--load-weights`` option.

By default, the model is exported to the OpenVINO™ IR format without extra feature information needed for the ``explain`` function. To use OpenVINO™ IR model in ``otx explain``, please first export it with ``--dump-features`` parameter:

.. code-block::

    (otx) ...$ otx export SSD --load-weights <path/to/trained/weights.pth> \
                              --output outputs/openvino/with_features \
                              --dump-features
    (otx) ...$ otx explain SSD --input <path/to/explain/root> \
                               --load-weights outputs/openvino/with_features \
                               --output <path/to/output/root> \
                               --explain-algorithm classwisesaliencymap \
                               --overlay-weight 0.5



*************
Demonstration
*************

``otx demo`` runs model inference on images, videos, or webcam streams to show how it works with the user's data.

.. note::

  ``otx demo`` command requires GUI backend to your system for displaying inference results.

  Only the OpenVINO™ IR model can be used for the ``otx demo`` command.

.. code-block::

    (otx) ...$ otx demo --help
    usage: otx demo [-h] -i INPUT --load-weights LOAD_WEIGHTS [--fit-to-size FIT_TO_SIZE FIT_TO_SIZE] [--loop] [--delay DELAY] [--display-perf] [--output OUTPUT] [template] {params} ...

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
                            Load model weights from previously saved checkpoint.It could be a trained/optimized model (with PTQ only) or exported model.
      --fit-to-size FIT_TO_SIZE FIT_TO_SIZE
                            Width and Height space-separated values. Fits displayed images to window with specified Width and Height. This options applies to result visualisation only.
      --loop                Enable reading the input in a loop.
      --delay DELAY         Frame visualization time in ms.
      --display-perf        This option enables writing performance metrics on displayed frame. These metrics take into account not only model inference time, but also frame reading, pre-processing and post-processing.
      --output OUTPUT
                            Output path to save input data with predictions.

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
    usage: otx deploy [-h] [--load-weights LOAD_WEIGHTS] [-o OUTPUT] [template]

    positional arguments:
      template              Enter the path or ID or name of the template file.
                            This can be omitted if you have train-data-roots or run inside a workspace.

    optional arguments:
      -h, --help            show this help message and exit
      --load-weights LOAD_WEIGHTS
                            Load model weights from previously saved checkpoint.
      -o OUTPUT, --output OUTPUT
                            Location where openvino.zip will be stored.


Command example:

.. code-block::

    (otx) ...$ otx deploy SSD --load-weights <path/to/openvino.xml> \
                              --output outputs/deploy

