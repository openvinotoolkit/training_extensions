How to run the demo with exportable code
========================================

In this tutorial we will show how to infer the model deployed in the previous step on images, videos, and webcam in order and see how it works on custom data.

1. Unzip the ``openvino.zip`` archive.

.. code-block::

    unzip  outputs/deploy/openvino.zip -d outputs/deploy/

2. To run the demo in exportable code, we can use a brand-new virtual environment, where we need to install the minimum packages from exportable code.

.. code-block::

    python3 -m venv demo_venv --prompt="demo"
    source demo_venv/bin/activate
    python -m pip install -r requirements.txt


3. The following line will run the demo on your input source, using the model in the ``model`` folder. You can pass as ``input`` a single image, a folder of images, a video file, or a web camera id.

.. code-block::

    (demo) ...$ python3 outputs/deploy/python/demo.py --input docs/utils/images/wgisd_dataset_sample.jpg \
                                                      --models outputs/deploy/model

You can press ``Q`` to stop inference during the demo running.

The model inference on your custom image will look like this:

.. image:: ../../../utils/images/wgisd_pr_sample.jpg
  :width: 600
  :alt: this image shows the inference results on the WGISD dataset

.. note::

    If you provide a single image as input, the demo processes and renders it quickly, then exits. To continuously
    visualize inference results on the screen, apply the ``loop`` option, which enforces processing a single image in a loop.

    You can change ``config.json`` that specifies the confidence threshold and color for each class visualization, but any changes should be made with caution.

To learn how to run the demo on Windows and MacOS, please refer to the ``README.md`` file in exportable code.

Congratulations! Now you have learned how to use base OTX functionality. For the advanced features, please refer to the next section called :doc:`../advanced/index`.

***************
Troubleshooting
***************

1. If you have access to the Internet through the proxy server only, please use pip with a proxy call as demonstrated by the command below:

.. code-block::

    python -m pip install --proxy http://<usr_name>:<password>@<proxyserver_name>:<port#> <pkg_name>


2. If you use Anaconda environment, you should consider that OpenVINO has limited `Conda support <https://docs.openvino.ai/2021.4/openvino_docs_install_guides_installing_openvino_conda.html>`_ for Python 3.6 and 3.7 versions only. But the demo package requires python 3.8.

So please use other tools to create the environment (like ``venv`` or ``virtualenv``) and use ``pip`` as a package manager.

3. If you have problems when you try to use ``pip install`` command, please update the pip version by the following command:

.. code-block::
   
    python -m pip install --upgrade pip
