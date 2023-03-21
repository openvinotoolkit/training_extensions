Explainable AI (XAI)
====================

**Explainable AI (XAI)** is a field of research that aims to make machine learning models more transparent and interpretable to humans.
The goal is to help users understand how and why AI systems make decisions and provide insight into their inner workings. It allows us to detect, analyze, and prevent common mistakes like the lack of data diversity for certain objects. 
XAI can help to build trust in AI, make sure that the model is safe for development and increase its adoption in various domains.

Most XAI tools generate **saliency maps** as a part of the process. It is a visual representation, suitable for human comprehension, that highlights the most important parts of the image that the network has focused on the most. 
It looks like a heatmap, where warm-colored areas represent the areas with main focuses.


.. image:: ../../../../utils/images/xai_example.jpg
  :width: 600
  :alt: this image shows the result of XAI algorithm


We can generate saliency maps for a certain model that was trained in OpenVINO™ Training Extensions, using ``otx explain`` command line. Learn more about its usage in  :doc:`../../tutorials/base/explain` tutorial.

*************************
Classification algorithms
*************************

.. image:: ../../../../utils/images/xai_cls.jpg
  :width: 600
  :alt: this image shows the comparison of XAI classification algorithms


For classification networks these algorithms are used to generate saliency maps:

- **Activation Map​** - this is the most basic and naive approach. It takes the outputs of the model's feature extractor (backbone) and averages it in channel dimension. The results highly rely on the backbone and ignore neck and head computations. Basically, it gives a relatively good and fast result.

- `Eigen-Cam <https://arxiv.org/abs/2008.00299​>`_ uses Principal Component Analysis (PCA).  It returns the first principal component of the feature extractor output, which most of the time corresponds to the dominant object. The results highly rely on the backbone as well and ignore neck and head computations.

- `Recipro-CAM​ <https://arxiv.org/pdf/2209.14074>`_ uses Class Activation Mapping (CAM) to weigh the activation map for each class, so it can generate different saliency per class. Recipro-CAM is a fast gradient-free Reciprocal CAM method. The method involves spatially masking the extracted feature maps to exploit the correlation between activation maps and network predictions for target classes. 


Below we show the comparison of described algorithms:

+-------------------------------------------+----------------+----------------+-------------------------------------------------------------------------+
| Classification algorithm                  | Activation Map | Eigen-Cam      | Recipro-CAM                                                             |
+===========================================+================+================+=========================================================================+
| Need access to model internal state       | Yes            | Yes            |  Yes                                                                    |
+-------------------------------------------+----------------+----------------+-------------------------------------------------------------------------+
| Gradient-free                             | Yes            | Yes            |  Yes                                                                    |
+-------------------------------------------+----------------+----------------+-------------------------------------------------------------------------+
| Single-shot                               | Yes            | Yes            |  No (re-infer neck + head H*W times, where HxW – feature map size)      |                                                          
+-------------------------------------------+----------------+----------------+-------------------------------------------------------------------------+
| Class discrimination                      | No             | No             | Yes                                                                     |
+-------------------------------------------+----------------+----------------+-------------------------------------------------------------------------+
| Execution speed                           | Fast           | Fast           | Medium                                                                  |  
+-------------------------------------------+----------------+----------------+-------------------------------------------------------------------------+


*************************
Detection algorithms
*************************

To generate a saliency map for the detection task, we use the **DetClassProbabilityMap** algorithm.
It's the naive approach for detection that takes the raw classification head output and uses class probability maps to calculate regions of interest for each class. So, it creates different salience maps for each class.
For now, this algorithm is implemented for single-stage detectors only.​

.. image:: ../../../../utils/images/xai_det.jpg
  :width: 600
  :alt: this image shows the detailed description of XAI detection algorithm


The main limitation of this method is that, due to training loss design, activation values drift towards the center of the object. It limits the getting of clear explanations in the near-edge image areas.​

+-------------------------------------------+-------------------------------------------------------------------------+
| Detection algorithm                       | DetClassProbabilityMap                                                  |
+===========================================+=========================================================================+
| Need access to model internal state       | Yes                                                                     |           
+-------------------------------------------+-------------------------------------------------------------------------+
| Gradient-free                             | Yes                                                                     |         
+-------------------------------------------+-------------------------------------------------------------------------+
| Single-shot                               | Yes                                                                     |         
+-------------------------------------------+-------------------------------------------------------------------------+
| Class discrimination                      | No                                                                      |          
+-------------------------------------------+-------------------------------------------------------------------------+
| Box discrimination                        | No                                                                      |          
+-------------------------------------------+-------------------------------------------------------------------------+
| Execution speed                           | Fast                                                                    |           
+-------------------------------------------+-------------------------------------------------------------------------+
