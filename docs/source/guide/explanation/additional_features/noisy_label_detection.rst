Noisy label detection
=====================

OpenVINO™ Training Extensions provide a feature for detecting noisy labels during model training.
With this feature, you can identify noisy labeled samples in your training dataset.
Our algorithm accumulates the training loss dynamics during the model training
and exports it to `Datumaro <https://github.com/openvinotoolkit/datumaro>`_.
The training loss dynamics are then post-processed by exponential moving average (EMA),
a strong criterion for detecting noisy label samples [1]_.
Finally, Datumaro ranks the top-k samples, which can be considered as noisy labeled candidates.
We provide an end-to-end example written in a Jupyter notebook, which you can find at the link in the note below.

In OpenVINO™ Training Extensions CLI, you can enable this feature
by adding the argument ``--algo_backend.enable_noisy_label_detection true`` as follows.

.. code-block::

    $ otx train ... params --algo_backend.enable_noisy_label_detection true

.. note::
    Currently, it only supports multi-class classification task and single GPU training.

.. note:: **Important!**
    The post-processing step to analyze the training loss dynamics requires `Datumaro <https://github.com/openvinotoolkit/datumaro>`_.
    Please see `this end-to-end Jupyter notebook example <https://github.com/openvinotoolkit/datumaro/blob/develop/notebooks/10_noisy_label_detection.ipynb>`_.

.. [1] Zhou, Tianyi, Shengjie Wang, and Jeff Bilmes. "Robust curriculum learning: from clean label detection to noisy label self-correction." International Conference on Learning Representations. 2021.
