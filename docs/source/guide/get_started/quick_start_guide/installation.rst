Installation
=============

**************
Prerequisites
**************

The current version of OpenVINO™ Training Extensions was tested under the following environment:

- Ubuntu 20.04
- Python 3.8.x
- (Optional) To use the NVidia GPU for the training: `CUDA Toolkit 11.7 <https://developer.nvidia.com/cuda-11-7-0-download-archive>`_

.. note::

        If using CUDA, make sure you are using a proper driver version. To do so, use ``ls -la /usr/local | grep cuda``. If necessary, `install CUDA 11.7 <https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local>`_ (requires 'sudo' permission) and select it with ``export CUDA_HOME=/usr/local/cuda-11.7``.

**********************
Install OpenVINO™ Training Extensions for users
**********************

1. Clone the training_extensions
repository with the following command:

.. code-block::

    git clone https://github.com/openvinotoolkit/training_extensions.git
    cd training_extensions
    git checkout develop

2. Set up a
virtual environment

.. code-block::

    # Create virtual env.
    python -m venv .otx

    # Activate virtual env.
    source .otx/bin/activate

3. Install prerequisite
dependencies with:

Install PyTorch according to your system environment. Refer to the `official installation guide <https://pytorch.org/get-started/previous-versions/>`_

.. note::

    Currently, only torch==1.13.1 was fully validated. torch==2.x will be supported soon. (Earlier versions are not supported due to security issues)

Example install command for torch==1.13.1+cu117:

.. code-block::

    pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117

4. Then, install
OpenVINO™ Training Extensions package

Install from a local source in development mode:

.. code-block::

    pip install -e .[full]

Or, you can install from PyPI:

.. code-block::

    pip install otx

5. Once the package is installed in the virtual environment, you can use full
`otx` command line functionality.

***************************
Install OpenVINO™ Training Extensions for developers
***************************

Install ``tox`` and create a development environment:

.. code-block::

    pip install tox
    # -- need to fix '38' in tox.ini if another python version needed
    tox devenv venv/otx -e pre-merge
    source venv/otx/bin/activate

Then you may change code and all fixes will be directly applied to the editable package

***************************
Run unit tests
***************************

.. code-block::

    pytest tests/unit

***************
Troubleshooting
***************

1. If you have problems when you try to use ``pip install`` command, please update pip version by following command:

.. code-block::

    python -m pip install --upgrade pip

2. If you use Anaconda environment, you should consider that OpenVINO has limited `Conda support <https://docs.openvino.ai/2021.4/openvino_docs_install_guides_installing_openvino_conda.html>`_ for Python 3.6 and 3.7 versions only. But the demo package requires python 3.8.
So please use other tools to create the environment (like ``venv`` or ``virtualenv``) and use ``pip`` as a package manager.

3. If you have access to the Internet through the proxy server only, please use pip with proxy call as demonstrated by command below:

.. code-block::

    python -m pip install --proxy http://<usr_name>:<password>@<proxyserver_name>:<port#> <pkg_name>





