Installation
============

**************
Prerequisites
**************

The current version of OpenVINO™ Training Extensions was tested in the following environment:

- Ubuntu 20.04
- Python 3.8 ~ 3.10
- (Optional) To use the NVidia GPU for the training: `CUDA Toolkit 11.7 <https://developer.nvidia.com/cuda-11-7-0-download-archive>`_

.. note::

        If using CUDA, make sure you are using a proper driver version. To do so, use ``ls -la /usr/local | grep cuda``.

        If necessary, `install CUDA 11.7 <https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local>`_ (requires 'sudo' permission) and select it with ``export CUDA_HOME=/usr/local/cuda-11.7``.

******************************************************
Install OpenVINO™ Training Extensions for users (CUDA)
******************************************************

1. Clone the training_extensions
repository with the following command:

.. code-block::

    git clone https://github.com/openvinotoolkit/training_extensions.git
    cd training_extensions
    git checkout develop

2. Set up a
virtual environment.

.. code-block::

    # Create virtual env.
    python -m venv .otx

    # Activate virtual env.
    source .otx/bin/activate

3. Install PyTorch according to your system environment.
Refer to the `official installation guide <https://pytorch.org/get-started/previous-versions/>`_

.. note::

    Currently, only torch==1.13.1 ~ 2.0.1 have been fully validated.
    (Older versions are not supported due to the security issues. Newer versions might not work correctly)

.. code-block::

    # Install command for torch==2.0.1 for CUDA 11.7:
    pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu117

    # Or, install command for torch==1.13.1 for CUDA 11.7:
    pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117

    # On CPU only systems:
    pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cpu

4. Install OpenVINO™ Training Extensions
package from either:

* A local source in development mode

.. code-block::

    pip install -e .[full]

* PyPI

.. code-block::

    pip install otx[full]

5. Once the package is installed in the virtual environment, you can use full
OpenVINO™ Training Extensions command line functionality.

******************************************************
Install OpenVINO™ Training Extensions for users (XPU)
******************************************************

1. Proceed with 1-2 steps from the above instructions
cloning the repository and setting up the environment

2. Install Intel® Extension
for PyTorch (IPEX):

Please, refer to `official documentation <https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.10%2Bxpu>`_
to ensure that necessary prerequisites are done.

.. code-block::

    python -m pip install torch==2.1.0.post0 torchvision==0.16.0.post0 torchaudio==2.1.0.post0 intel-extension-for-pytorch==2.1.20+xpu oneccl_bind_pt==2.1.200+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

3. Install mmcv and apply a custom patch
to make it compatible with XPU devices:

* From PyPI

.. code-block::

    curl -o mmcv-full-1.7.0.tar.gz https://files.pythonhosted.org/packages/a1/81/89120850923f4c8b49efba81af30160e7b1b305fdfa9671a661705a8abbf/mmcv-full-1.7.0.tar.gz
    tar -zxvf mmcv-full-1.7.0.tar.gz
    sed -i 's/c++14/c++17/g' mmcv-full-1.7.0/setup.py
    MMCV_WITH_OPS=1 pip install ./mmcv-full-1.7.0
    rm -r mmcv-full-1.7.0 mmcv-full-1.7.0.tar.gz

* From source

.. code-block::

    git clone https://github.com/open-mmlab/mmcv
    cd mmcv
    git checkout v1.7.0
    wget https://gist.githubusercontent.com/sovrasov/44c41202c09ab38d657e796fccc86181/raw/146a50c5f99c8721c0bcc0fcc25b19064c4b29a2/mmcv_1_7_0_setup.patch
    git apply mmcv_1_7_0_setup.patch
    rm mmcv_1_7_0_setup.patch
    MMCV_WITH_OPS=1 python setup.py develop --user

4. Install OpenVINO™ Training Extensions
package from source:

.. code-block::

    pip install -e .[full]

5. Activate Intel OneAPI
environment:

.. code-block::

    source /path/to/intel/oneapi/setvars.sh

6. Currently, IPEX may stop during training with a segmentation error (core dump).
To avoid this, please, execute:

.. code-block::

    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30

.. note::

    Currently, OTX fully supports classification, object detection, and anomaly tasks with XPU devices.
    (Instance Segmentation and Semantic Segmentation tasks may work with accuracy and performance problems)

****************************************************
Install OpenVINO™ Training Extensions for developers
****************************************************

Install ``tox`` and create a development environment:

.. code-block::

    pip install tox
    # -- need to replace '310' below if another python version needed
    tox devenv venv/otx -e tests-all-py310
    source venv/otx/bin/activate

Then you may change code, and all fixes will be directly applied to the editable package.

*****************************************************
Install OpenVINO™ Training Extensions by using Docker
*****************************************************

To build a docker image with Python 3.9, run a command below from the working copy of the OpenVINO training extensions.

.. code-block::

    # build a docker image (otx/cpu/python3.9:latest) with Python 3.9 (default)
    training_extensions$ ./docker/build.sh
    # or, with other version of Python e.g., 3.10
    training_extensions$ ./docker/build.sh --python 3.10

.. note::

    When the docker image build script completed successfully, the image will be named and tagged as `otx/cpu/python<py-version-string>:latest`.
    You can check it using the command `docker images` on the terminal.

To start the OpenVINO training extensions container using the image built in above, run a command below.

.. code-block::

    # start a container from `otx/cpu/python3.9:latest' image.
    $ docker run \
        -it \ # enter interactive terminal
        --rm \ # remove container after use
        -v "$(pwd):/mnt/shared:rw" \ # mount current folder on host machine to the container
        --shm-size=4g \ # increase mounted shared memory
        otx/cpu/python3.9:latest    # name of the docker image to be used to create container

Enjoy OpenVINO training extensions!

.. code-block::

    # find all templates for the classification task
    root@fc01132c3753:/training_extensions# otx find --task classification
    +----------------+---------------------------------------------------+-----------------------+---------------------------------------------------------------------------------------+
    |      TASK      |                         ID                        |          NAME         |                                       BASE PATH                                       |
    +----------------+---------------------------------------------------+-----------------------+---------------------------------------------------------------------------------------+
    | CLASSIFICATION |       Custom_Image_Classification_DeiT-Tiny       |       DeiT-Tiny       |           src/otx/algorithms/classification/configs/deit_tiny/template.yaml           |
    | CLASSIFICATION |    Custom_Image_Classification_EfficinetNet-B0    |    EfficientNet-B0    |    src/otx/algorithms/classification/configs/efficientnet_b0_cls_incr/template.yaml   |
    | CLASSIFICATION |   Custom_Image_Classification_EfficientNet-V2-S   |   EfficientNet-V2-S   |   src/otx/algorithms/classification/configs/efficientnet_v2_s_cls_incr/template.yaml  |
    | CLASSIFICATION | Custom_Image_Classification_MobileNet-V3-large-1x | MobileNet-V3-large-1x | src/otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml |
    +----------------+---------------------------------------------------+-----------------------+---------------------------------------------------------------------------------------+

*********
Run tests
*********

To run some tests, need to have development environment on your host. The development requirements file (requirements/dev.txt)
would be used to setup them.

.. code-block::

    $ pip install -r requirements/dev.txt
    $ pytest tests/

Another option to run the tests is using the testing automation tool `tox <https://tox.wiki/en/latest/index.html>`_. Following commands will install
the tool ``tox`` to your host and run all test codes inside of ``tests/`` folder.

.. code-block::

    $ pip install tox
    $ tox -e tests-all-py310-pt1 -- tests/

.. note::

    When running the ``tox`` command above first time, it will create virtual env by installing all dependencies of this project into
    the newly created environment for your testing before running the actual testing. So, it is expected to wait more than 10 minutes
    before to see the actual testing results.

***************
Troubleshooting
***************

1. If you have problems when you try to use ``pip install`` command,
please update pip version by following command:

.. code-block::

    python -m pip install --upgrade pip

2. If you're facing a problem with ``torch`` or ``mmcv`` installation, please check that your CUDA version is compatible with torch version.
Consider updating CUDA and CUDA drivers if needed.
Check the `command example <https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local>`_ to install CUDA 11.7 with drivers on Ubuntu 20.04.

3. If you use Anaconda environment, you should consider that OpenVINO has limited `Conda support <https://docs.openvino.ai/2021.4/openvino_docs_install_guides_installing_openvino_conda.html>`_ for Python 3.6 and 3.7 versions only.
So to use these python versions, please use other tools to create the environment (like ``venv`` or ``virtualenv``) and use ``pip`` as a package manager.

4. If you have access to the Internet through the proxy server only,
please use pip with proxy call as demonstrated by command below:

.. code-block::

    python -m pip install --proxy http://<usr_name>:<password>@<proxyserver_name>:<port#> <pkg_name>

5. If you get ``mmcv`` kernel compilation error message, e.g. ModuleNotFoundEffor: no module named 'mmcv._ext',
please try to delete the pre-compiled MMCV wheel from the cache directory, and then try again.
Then the kernels would be compiled on your environment.

.. code-block::

    find ~/.cache/pip/wheels/ -name "mmcv*" -delete
    pip uninstall mmcv-full
    pip install otx[full]  # pip install -e .[full]
