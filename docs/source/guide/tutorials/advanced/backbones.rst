Backbone Replacement
================================

This tutorial describes an example of how to find an available backbone and how it can be replaced in OpenVINO™ Training Extensions.

The process has been tested on the following configuration.

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-11900
- CUDA Toolkit 11.1, python 3.9

*****************************
Currently supported backbones
*****************************

The following libraries are currently available for backbone replacement.

+-----------------------+-------+-------+-------------+-----------+-----------+
|         Task          | mmdet | mmseg | torchvision | pytorchcv | omz.mmcls |
+=======================+=======+=======+=============+===========+===========+
|    Classification     |   O   |   O   |      O      |     O     |     O     |
+-----------------------+-------+-------+-------------+-----------+-----------+
|       Detection       |   O   |   O   |      O      |     O     |           |
+-----------------------+-------+-------+-------------+-----------+-----------+
|     Segmentation      |   O   |   O   |      O      |     O     |           |
+-----------------------+-------+-------+-------------+-----------+-----------+
| Instance-Segmentation |   O   |   O   |      O      |     O     |           |
+-----------------------+-------+-------+-------------+-----------+-----------+

*************************
Setup virtual environment
*************************

1. You can follow the installation process from a :doc:`quick start guide <../../get_started/quick_start_guide/installation>` 
to create a universal virtual environment for OpenVINO™ Training Extensions.

2. Activate your virtual 
environment:

.. code-block::

  .otx/bin/activate
  # or by this line, if you created an environment, using tox
  . venv/otx/bin/activate

*****************************
Backbone replacement tutorial
*****************************

1. First, we need to configure the workspace 
for the backbone replacement:

.. note::

  You can use the OpenVINO™ Training Extensions workspace to swap out backbones, train, set up configurations, and more.
  Workspaces are created automatically on ``otx build`` or ``otx train``.

.. code-block::

  (otx) ...$ otx build --task classification

  [*] Workspace Path: otx-workspace-CLASSIFICATION
  [*] Load Model Template ID: Custom_Image_Classification_EfficinetNet-B0
  [*] Load Model Name: EfficientNet-B0
  [*]     - Updated: otx-workspace-CLASSIFICATION/model.py
  [*]     - Updated: otx-workspace-CLASSIFICATION/data_pipeline.py
  [*]     - Updated: otx-workspace-CLASSIFICATION/deployment.py
  [*]     - Updated: otx-workspace-CLASSIFICATION/hpo_config.yaml
  [*]     - Updated: otx-workspace-CLASSIFICATION/model_hierarchical.py
  [*]     - Updated: otx-workspace-CLASSIFICATION/model_multilabel.py
  [*]     - Updated: otx-workspace-CLASSIFICATION/compression_config.json
  [*] Update data configuration file to: otx-workspace-CLASSIFICATION/data.yaml

  (otx) ...$ cd otx-workspace-CLASSIFICATION

2. Next, we can find the backbone
we want to replace via ``otx find``:

.. note::

  We can use ``otx find`` to find templates and available backbones.
  Each backbone may have a required argument. If the backbone has options for required arguments, ``otx build`` provide the first option as default.

.. code-block::

  (otx) ...$ otx find --backbone mmdet

  +-------+-------------------------+---------------+-------------------------------+
  | Index |      Backbone Type      | Required-Args |            Options            |
  +-------+-------------------------+---------------+-------------------------------+
  |   1   |       mmdet.RegNet      |      arch     | regnetx_400mf, regnetx_800mf, |
  |       |                         |               | regnetx_1.6gf, regnetx_3.2gf, |
  |       |                         |               | regnetx_4.0gf, regnetx_6.4gf, |
  |       |                         |               |  regnetx_8.0gf, regnetx_12gf  |
  |   2   |       mmdet.ResNet      |     depth     |      18, 34, 50, 101, 152     |
  |   3   |     mmdet.ResNetV1d     |     depth     |      18, 34, 50, 101, 152     |
  |   4   |      mmdet.ResNeXt      |     depth     |          50, 101, 152         |
  |   5   |       mmdet.SSDVGG      |   input_size  |            300, 512           |
  |       |                         |     depth     |           11, 16, 19          |
  |   6   |       mmdet.HRNet       |     extra     |                               |
  |   7   |      mmdet.Res2Net      |     depth     |          50, 101, 152         |
  |   8   |  mmdet.DetectoRS_ResNet |     depth     |          50, 101, 152         |
  |   9   | mmdet.DetectoRS_ResNeXt |     depth     |          50, 101, 152         |
  |   10  |      mmdet.Darknet      |               |                               |
  |   11  |      mmdet.ResNeSt      |     depth     |       50, 101, 152, 200       |
  |   12  |     mmdet.CSPDarknet    |               |                               |
  +-------+-------------------------+---------------+-------------------------------+

3. We need to run the command below to replace 
the backbone:

In this example, we'll replace the classification model using the default EfficientNet with ``mmdet.ResNet``.
You can use the ``Backbone Type`` in the table output from ``otx find --backbone`` to use a different backbone.

.. code-block::

  (otx) ...$ otx build --backbone mmdet.RegNet

  [*] Backbone Config: mmdet.RegNet
  [*] mmdet.RegNet requires the argument : ['arch']
  [*] Please refer to /venv/lib/python3.9/site-packages/mmdet/models/backbones/regnet.py
  [*] 'arch' can choose between: ['regnetx_400mf', 'regnetx_800mf', 'regnetx_1.6gf', 'regnetx_3.2gf', 'regnetx_4.0gf', 'regnetx_6.4gf', 'regnetx_8.0gf', 'regnetx_12gf']
  [*] 'arch' default value: regnetx_400mf
  [*] Save backbone configuration: otx-workspace-CLASSIFICATION/backbone.yaml
  [*] Update model.py with backbone.yaml
          Target Model: SAMImageClassifier
          Target Backbone: mmdet.RegNet
          Backbone config: {'arch': 'regnetx_400mf', 'avg_down': False, 'base_channels': 32, 'conv_cfg': None, 'dcn': None, 'deep_stem': False, 'dilations': (1, 1, 1, 1), 'frozen_stages': -1, 'in_channels': 3, 'init_cfg': None, 'norm_cfg': {'requires_grad': True, 'type': 'BN'}, 'norm_eval': True, 'out_indices': (0, 1, 2, 3), 'plugins': None, 'pretrained': None, 'stage_with_dcn': (False, False, False, False), 'stem_channels': 32, 'strides': (2, 2, 2, 2), 'style': 'pytorch', 'type': 'mmdet.RegNet', 'with_cp': False, 'zero_init_residual': True}
  [*] Save model configuration: model.py

Then we get ``model.py``, which has been changed to ``mmdet.ResNet``.

.. note::

  If you get a log like this, follow the steps below:

  .. code-block::

    [!] mmseg.HRNet backbone has inputs that the user must enter.
    [!] Edit backbone.yaml and run 'otx build --backbone backbone.yaml'.

  Please modify the available configuration file directly (``backbone.yaml``).

  You can then update the model with the command below:

  .. code-block::

    (otx) ...$ otx build --backbone backbone.yaml

4. After that, you can use any other OpenVINO™ Training Extensions command with the 
new model: :doc:`quick start guide <../../get_started/quick_start_guide/installation>`

You can use the backbones provided by ``mmdet``, ``mmseg``, ``torchvision``, and ``omz.mmcls`` in the same way as above.

.. warning::
  Depending on your backbone, your data may require multiple hyperparameter optimizations. Custom models, except for TEMPLATE, are not yet guaranteed to be accurate.
