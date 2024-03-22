Installation
=============

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

***********************************************
Install OpenVINO™ Training Extensions for users
***********************************************

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

    Currently, only torch==1.13.1 ~ 2.0.1 was fully validated. (older versions are not supported due to security issues).

.. code-block::

    # Install command for torch==2.0.1 for CUDA 11.7:
    pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu117

    # Or, install command for torch==1.13.1 for CUDA 11.7:
    pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117

4. Install OpenVINO™ Training Extensions package from either:

* A local source in development mode

.. code-block::

    pip install -e .[full]

* PyPI

.. code-block::

    pip install otx[full]

5. Once the package is installed in the virtual environment, you can use full
OpenVINO™ Training Extensions command line functionality.

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

****************************************************
Install OpenVINO™ Training Extensions by using Docker
****************************************************

.. code-block::

    $ docker build \
        -t trainer \ # image tag, required
        --build-arg UBUNTU_VER=20.04 \ # default Ubunutu version, optional
        --build-arg PYTHON_VER=3.9 \ # default Python version, optional
        --build-arg SOURCE=https://download.pytorch.org/whl/cpu \ # default (CPU) deps, optional
        . # training_extensions/
    $ docker run \
        -it \ # enter interactive terminal
        --rm \ # remove container after use
        -v "$(pwd)/shared:/mnt/shared:rw" \ # shared volume to host machine
        --shm-size=4g \ # increase mounted shared memory
        trainer
    trainer$ otx # ... installed on Ubuntu 20.04 with /mnt/shared as shared directory

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
