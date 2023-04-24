How to run the demonstration mode with OpenVINO™ Training Extensions CLI
========================================================================

This tutorial shows how to run :doc:`trained <how_to_train/index>` model inside OTX repository in demonstration mode.
It allows you to apply the model on the custom data or the online footage from a web camera and see how it will work in the real-life scenario.

.. note::

    This tutorial uses an object detection model for example, however for other tasks the functionality remains the same - you just need to replace the input dataset with your own.

For visualization you use images from WGISD dataset from the :doc:`object detection tutorial <how_to_train/detection>`.

1. Activate the virtual environment 
created in the previous step.

.. code-block::

    source .otx/bin/activate

2. As an ``input`` we can use a single image, 
a folder of images, a video file, or a web camera id. We can run the demo on PyTorch (.pth) model and IR (.xml) model.

The following command will run the demo on your input source, using PyTorch ``outputs/weights.pth``. 

.. code-block::

    (demo) ...$ otx demo --input docs/utils/images/wgisd_dataset_sample.jpg \
                         --load-weights outputs/weights.pth

But if we'll provide a single image the demo processes and renders it quickly, then exits. To continuously visualize inference results on the screen, apply the ``loop`` option, which enforces the processing a single image in a loop.

.. code-block::

    (demo) ...$ otx demo --input docs/utils/images/wgisd_dataset_sample.jpg \
                         --load-weights outputs/weights.pth --loop

In this case, you can stop the demo by pressing `Q` button or killing the process in the terminal (``Ctrl+C`` for Linux).

3. If we want to pass an images folder, it's better to specify the delay parameter, that defines, how much millisecond pause will be held between showing the next image.
For example ``--delay 100`` will make this pause 0.1 ms.
If you want to skip showing the resulting image and instead see the number of predictions and time spent on each image inference, specify ``--delay 0``.


4. In WGISD dataset we have high-resolution images, 
so the ``--fit-to-size`` parameter would be quite useful. It resizes the resulting image to a specified:

.. code-block::

    (demo) ...$ otx demo --input docs/utils/images/wgisd_dataset_sample.jpg \
                         --load-weights outputs/weights.pth --loop --fit-to-size 800 600


5. To save inferenced results with predictions on it, we can specify the folder path, using ``--output``. 
It works for images, videos, image folders and web cameras. To prevent issues, do not specify it together with a ``--loop`` parameter.

.. code-block::

    (demo) ...$ otx demo --input docs/utils/images/wgisd_dataset_sample.jpg \
                         --load-weights outputs/weights.pth \
                         --output resulted_images

6. If we want to show inference speed right on images, 
we can run the following line:

.. code-block::

    (demo) ...$ otx demo --input docs/utils/images/wgisd_dataset_sample.jpg \
                         --load-weights outputs/weights.pth --loop \
                         --fit-to-size 800 600 --display-perf


6. To run a demo on a web camera, you need to know its ID. 
You can check a list of camera devices by running the command line below on Linux system:

.. code-block::

    (demo) ...$ sudo apt-get install v4l-utils
    (demo) ...$ v4l2-ctl --list-devices

The output will look like this:

.. code-block::

    Integrated Camera (usb-0000:00:1a.0-1.6):
        /dev/video0

After that, you can use this ``/dev/video0`` as a camera ID for ``--input``.

Congratulations! Now you have learned how to use base OpenVINO™ Training Extensions functionality. For the advanced features, refer to the next section called :doc:`../advanced/index`.

***************
Troubleshooting
***************

If you use Anaconda environment, keep in mind that OpenVINO has limited `Conda support <https://docs.openvino.ai/2021.4/openvino_docs_install_guides_installing_openvino_conda.html>`_ for Python 3.6 and 3.7 versions only. The demo package requires python 3.8, though.
Therefore, use other tools to create the environment (like ``venv`` or ``virtualenv``) and use ``pip`` as a package manager.