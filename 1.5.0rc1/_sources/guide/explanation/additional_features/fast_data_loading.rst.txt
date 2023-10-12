Fast Data Loading
=================

OpenVINO™ Training Extensions provides several ways to boost model training speed,
one of which is fast data loading.


===================
Faster Augmentation
===================


******
AugMix
******
AugMix [1]_ is a simple yet powerful augmentation technique
to improve robustness and uncertainty estimates of image classification task.
OpenVINO™ Training Extensions implemented it in `Cython <https://cython.org/>`_ for faster augmentation.
Users do not need to configure anything as cythonized AugMix is used by default.



=======
Caching
=======


*****************
In-Memory Caching
*****************
OpenVINO™ Training Extensions provides in-memory caching for decoded images in main memory.
If the batch size is large, such as for classification tasks, or if dataset contains
high-resolution images, image decoding can account for a non-negligible overhead
in data pre-processing.
One can enable in-memory caching for maximizing GPU utilization and reducing model
training time in those cases.


.. code-block::

   $ otx train --mem-cache-size=8GB ..



***************
Storage Caching
***************

OpenVINO™ Training Extensions uses `Datumaro <https://github.com/openvinotoolkit/datumaro>`_
under the hood for dataset managements.
Since Datumaro `supports <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/arrow.html>`_
`Apache Arrow <https://arrow.apache.org/overview/>`_, OpenVINO™ Training Extensions
can exploit fast data loading using memory-mapped arrow file at the expanse of storage consumtion.


.. code-block::

   $ otx train .. params --algo_backend.storage_cache_scheme JPEG/75


The cache would be saved in ``$HOME/.cache/otx`` by default.
One could change it by modifying ``OTX_CACHE`` environment variable.


.. code-block::

   $ OTX_CACHE=/path/to/cache otx train .. params --algo_backend.storage_cache_scheme JPEG/75


Please refere `Datumaro document <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/arrow.html#export-to-arrow>`_
for available schemes to choose but we recommend ``JPEG/75`` for fast data loaidng.

.. [1] Dan Hendrycks, Norman Mu, Ekin D. Cubuk, Barret Zoph, Justin Gilmer, and Balaji Lakshminarayanan. "AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty" International Conference on Learning Representations. 2020.
