Install OTX
===========
This section demonstrates how to install `otx`.

.. contents:: Table of Contents
    :class: this-will-duplicate-information-and-it-is-still-useful-here
    :depth: 2
    :local:
    :backlinks: none


Create a Virtual Environment
----------------------------
We highly recommend using the library with a fresh virtual environment since the
library might require certain versions of packages, which could conflict with
the existing packages on your system.

Via Conda
~~~~~~~~~
To install the library via conda, the following snippet could be used.

.. code-block:: bash

    # Create a virtual env via Conda.
    yes | conda create -n otx python=3.8
    conda activate otx

Via Pyenv
~~~~~~~~~
Pyenv is another virtual environment manager to handle various virtual
environments with different Python versions.

.. code-block:: bash

    # Create a virtual env via pyenv
    pyenv virtualenv 3.8.13 otx
    pyenv local otx

It is also possible a install a virtual environment via other tools such as
Python's ``virtualenv`` and ``venv``.

Install OTX
-----------
The library can be installed via either PyPI or editable local install.

Local Install
~~~~~~~~~~~~~
Local install is currently the best approach to install the library since it
will also allow the user to edit the changes within the codebase.

.. code-block:: bash

    git clone https://github.com/openvinotoolkit/training_extensions.git
    cd training_extensions
    pip install -e .[full]

.. note::
    If you use zsh, ``pip install .[full]`` may not work properly. In this case,
    try installing the library via ``pip install ".[full]"``

PyPI Install
~~~~~~~~~~~~

.. warning::

    PyPI install is currently not stable, and we advise the users to use the
    local install instead. With future otx versions, this feature will be more
    stable, such that the user could install the library via,

.. code-block:: bash

    pip install otx[full]


Options
~~~~~~~
Unlike the well-known libraries and frameworks, ``otx`` provides number of
task types such as classification, detection, segmentation and anomaly in a
single library. Even though this is a rather unique feature, some users may want
to install a specific task instead. ``otx`` therefore makes this installation
completely optional, depending on the algorithmic needs.

Basic Installation
""""""""""""""""""
This option is the most basic option, where pip only installs ``otx`` api and
cli modules. For those who are only interested in the api and cli capabilities
could install this option.

.. code-block:: bash

    # Install API and CLI only.
    pip install otx

Classification Task Installation
""""""""""""""""""""""""""""""""
Classification option would install ``otx`` with classification algorithms only.

.. code-block:: bash

    # Install only otx classification library
    pip install otx[classification]

Detection Task Installation
"""""""""""""""""""""""""""
Detection would install the ``otx`` with detection algorithms, which are based
on `mmdetection <https://github.com/open-mmlab/mmdetection>`_ library but with a
lot more features.

.. code-block:: bash

    # Install otx detection library
    pip install otx[detection]

Segmentation Task Installation
""""""""""""""""""""""""""""""
Similar to the detection task, ``segmentation`` option would install segmentation
algorithms that utilizes `mmsegmentation <https://github.com/open-mmlab/mmsegmentation>`_
library, but again with more end-to-end functionality such as OpenVINO export.

.. code-block:: bash

    # Install segmentation library
    pip install otx[segmentation]

Anomaly Task Installation
"""""""""""""""""""""""""
``anomaly`` option would install the anomaly classification, detection and
segmentation tasks that uses `anomalib <https://github.com/openvinotoolkit/anomalib>`_.
Users would train models within a one-class classification fashion by utilizing
only the ``normal`` class during training to find any ``abnormality`` during the
validation, test or inference.

.. code-block:: bash

    # Install otx anomaly library
    pip install otx[anomaly]

Full Installation
"""""""""""""""""
``full``` option would install all of the tasks together, overall making ``otx``
a complete library that a user could train a supervised, semi-supervised or
self-supervised classification, detection or segmentation algorithms with full
OpenVINO capabilities.

.. code-block:: bash

    # Install full otx library
    pip install otx[full]