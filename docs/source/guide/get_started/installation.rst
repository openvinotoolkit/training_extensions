:octicon:`package` Installation
====================================

**************
Prerequisites
**************

The current version of OpenVINO™ Training Extensions was tested in the following environment:

- Ubuntu 20.04
- Python >= 3.10


**********************************************************
Install OpenVINO™ Training Extensions for users (CUDA/CPU)
**********************************************************

1. Install OpenVINO™ Training Extensions
package:

* A local source in development mode

.. tab-set::

    .. tab-item:: PyPI

        .. code-block:: shell

            pip install otx

    .. tab-item:: Source

        .. code-block:: shell

            # Clone the training_extensions repository with the following command:
            git clone https://github.com/openvinotoolkit/training_extensions.git
            cd training_extensions

            # Set up a virtual environment.
            python -m venv .otx
            source .otx/bin/activate

            pip install -e .

2. Install PyTorch & Requirements for training
according to your system environment.

.. tab-set::

    .. tab-item:: Minimum requirements

        .. code-block:: shell

            pip install '.[base]'

        .. note::

            Models from mmlab are not available for this environment. If you want to use mmlab models, you must install them with Full Requirements.
            Also, some tasks may not be supported by minimum requirements.

    .. tab-item:: Full Requirements

        .. code-block:: shell

            otx install -v

[Optional] Refer to the `torch official installation guide <https://pytorch.org/get-started/previous-versions/>`_

.. note::

    Currently, only torch==2.1.1 was fully validated. (older versions are not supported due to security issues).


3. Once the package is installed in the virtual environment, you can use full
OpenVINO™ Training Extensions command line functionality.

*************************************************************
Install OpenVINO™ Training Extensions for users (XPU devices)
*************************************************************

1. Install OpenVINO™ Training Extensions
package:

* A local source in development mode

.. tab-set::

    .. tab-item:: PyPI

        .. code-block:: shell

            pip install otx

    .. tab-item:: Source

        .. code-block:: shell

            # Clone the training_extensions repository with the following command:
            git clone https://github.com/openvinotoolkit/training_extensions.git
            cd training_extensions

            # Set up a virtual environment.
            python -m venv .otx
            source .otx/bin/activate

            pip install -e .

2. Install Intel Extensions for Pytorch & Requirements
for training according to your system environment.

.. tab-set::

    .. tab-item:: Minimum requirements

        .. code-block:: shell

            pip install '.[xpu]' --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

        .. note::

            Models from mmlab are not available for this environment. If you want to use mmlab models, you must install them with Full Requirements.
            Also, some tasks may not be supported by minimum requirements.

    .. tab-item:: Full Requirements

        .. code-block:: shell
            python -m pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30+xpu oneccl_bind_pt==2.1.300+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
            git clone https://github.com/open-mmlab/mmcv
            cd mmcv
            git checkout v2.1.0
            MMCV_WITH_OPS=1 pip install -e .
            cd ..
            otx install -v --do-not-install-torch

[Optional] Refer to the `Intel® Extension for PyTorch documentation guide <https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.10%2bxpu&os=linux%2fwsl2&package=pip>`_

3. Activate OneAPI environment
and export required IPEX system variables

.. code-block:: shell

    source /path/to/intel/oneapi/setvars.sh
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30
    export IPEX_FP32_MATH_MODE=TF32

3. Once the package is installed in the virtual environment, you can use full
OpenVINO™ Training Extensions command line functionality.

.. code-block:: shell

    otx --help

****************************************************
Install OpenVINO™ Training Extensions for developers
****************************************************

Install ``tox`` and create a development environment:

.. code-block:: shell

    pip install tox
    # -- need to replace '310' below if another python version needed
    tox devenv venv/otx -e unit-test-py310
    source venv/otx/bin/activate

Then you may change code, and all fixes will be directly applied to the editable package.

*****************************************************
Install OpenVINO™ Training Extensions by using Docker
*****************************************************

1. By executing the following commands, it will build two
Docker images: ``otx:${OTX_VERSION}-cuda`` and ``otx:${OTX_VERSION}-cuda-pretrained-ready``.

.. code-block:: shell

    git clone https://github.com/openvinotoolkit/training_extensions.git
    cd docker
    ./build.sh

2. After that, you can check whether the
images are built correctly such as

.. code-block:: shell

    docker image ls | grep otx

Example:

.. code-block:: shell

    otx                                           2.0.0-cuda-pretrained-ready                    4f3b5f98f97c   3 minutes ago   14.5GB
    otx                                           2.0.0-cuda                                     8d14caccb29a   8 minutes ago   10.4GB


``otx:${OTX_VERSION}-cuda`` is a minimal Docker image where OTX is installed with CUDA supports. On the other hand, ``otx:${OTX_VERSION}-cuda-pretrained-ready`` includes all the model pre-trained weights that OTX provides in addition to ``otx:${OTX_VERSION}-cuda``.

*********
Run tests
*********

To run some tests, need to have development environment on your host. The development requirements file (requirements/dev.txt)
would be used to setup them.

.. code-block:: shell

    $ otx install --option dev
    $ pytest tests/

Another option to run the tests is using the testing automation tool `tox <https://tox.wiki/en/latest/index.html>`_. Following commands will install
the tool ``tox`` to your host and run all test codes inside of ``tests/`` folder.

.. code-block::

    $ pip install tox
    $ tox -e tests-all-py310 -- tests/

.. note::

    When running the ``tox`` command above first time, it will create virtual env by installing all dependencies of this project into
    the newly created environment for your testing before running the actual testing. So, it is expected to wait more than 10 minutes
    before to see the actual testing results.

***************
Troubleshooting
***************

1. If you have problems when you try to use ``pip install`` command,
please update pip version by following command:

.. code-block:: shell

    python -m pip install --upgrade pip

2. If you're facing a problem with ``torch`` or ``mmcv`` installation, please check that your CUDA version is compatible with torch version.
Consider updating CUDA and CUDA drivers if needed.
Check the `command example <https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local>`_ to install CUDA 11.8 with drivers on Ubuntu 20.04.

3. If you have access to the Internet through the proxy server only,
please use pip with proxy call as demonstrated by command below:

.. code-block:: shell

    python -m pip install --proxy http://<usr_name>:<password>@<proxyserver_name>:<port#> <pkg_name>

4. If you're facing a problem with CLI side of the OTX, please check the help message of the command by using ``--help`` option.
If you still want to see more ``jsonargparse``-related messages, you can set the environment variables like below.

.. code-block:: shell

    export JSONARGPARSE_DEBUG=1 # 0: Off, 1: On
