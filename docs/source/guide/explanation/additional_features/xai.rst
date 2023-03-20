Explainable AI
==================

Explainable AI is a set of tools 

Explainable AI (XAI) is a field of research that aims to make machine learning models more transparent and interpretable to humans.
The goal is to help users understand how and why AI systems make decisions, to provide insight into their inner workings, detect, analyze and prevent common mistakes like the lack of data diversity for certain objects. 
XAI can help to build trust in AI, make sure that the model is safe for development and increase its adoption in various domains.

The most XAI tools generate **saliency maps** as a part of the process. It's a visual representation suitable for human understanding that highlights the most important parts of an image, there network focused the most. It looks like a heatmap there warm-colored areas represent the areas with main focuses.

|

.. image:: ../../../../../utils/images/xai_example.jpg
  :width: 600
  :alt: this image shows the result of XAI algorithm

|




*************************
Classification algorithms
*************************

|

.. image:: ../../../../../utils/images/xai_cls.jpg
  :width: 600
  :alt: this image shows the comparison of XAI classification algorithms

|



For classification networks these algorithms are used to generate saliency maps:

- **Activation Map​** - this is the most basic and naive approach. It takes the outputs of the model's feature extractor (backbone) and averages it in channel dimension.
The results are highly rely on backbone and ignore neck and head computations. Basically, it gives a relatively good and fast result.

- `Eigen-Cam <https://arxiv.org/abs/2008.00299​>`_ uses Principal Compomponent Analisys (PCA).  It returns the first principal component of the feature extractor output, which in most of the time corresponds to the dominant object.
The results are highly rely on backbone as well and ignore neck and head computations.

- `Recipro-CAM​ <https://arxiv.org/pdf/2209.14074>`_ uses Class Activation Mapping (CAM) to weight the activation map for each class, so it can generate different saliency per class.
Recipro-CAM is a fast gradient-free Reciprocal CAM method. Method involves spatially masking the extracted feature maps to exploit the correlation between activation maps and network predictions for target classes. 


Below we show the comparison of described algorithms:

+-------------------------------------------+----------------+----------------+-------------------------------------------------------------------------+
| Cls algo                                  | Activation Map | Eigen-Cam      | Recipro-CAM                                                             |
+===========================================+================+================+=========================================================================+
| Need access to model internal state       | Yes            | Yes            |  Yes 
+-------------------------------------------+----------------+----------------+-------------------------------------------------------------------------+
| Gradient-free                             | Yes            | Yes            | Yes
+-------------------------------------------+---------------------------------+-------------------------------------------------------------------------+
| Single-shot                               | Yes            | Yes            | No (re-infer neck + head H*W times, where HxW – feature map size)
+-------------------------------------------+----------------+----------------+-------------------------------------------------------------------------+
| Class discrimination                      | No             | No             | Yes
+-------------------------------------------+----------------+----------------+-------------------------------------------------------------------------+
| Execution speed                           | Fast           | Fast           | Medium
+-------------------------------------------+----------------+----------------+-------------------------------------------------------------------------+


*************************
Detection algorithms
*************************

To generate saliency map for detection task we use only **DetClassProbabilityMap** algorithm.
It's the naive approach for detection that takes raw classification head output, and uses class probability maps to calculate region of interests for each class. For know, this algorithm is implemented for single-stage detectors only.​

The main limitation of this method is that ,due to training loss design, activation values drift towards the center of the object. It limits the getting of clear explanations in the near-edge image areas.​

​+-------------------------------------------+----------------+----------------+-------------------------------------------------------------------------+
| Cls algo                                  | Activation Map | Eigen-Cam      | Recipro-CAM                                                             |
+===========================================+================+================+=========================================================================+
| Need access to model internal state       | Yes            | Yes            |  Yes 
+-------------------------------------------+----------------+----------------+-------------------------------------------------------------------------+
| Gradient-free                             | Yes            | Yes            | Yes
+-------------------------------------------+---------------------------------+-------------------------------------------------------------------------+
| Single-shot                               | Yes            | Yes            | No (re-infer neck + head H*W times, where HxW – feature map size)
+-------------------------------------------+----------------+----------------+-------------------------------------------------------------------------+
| Class discrimination                      | No             | No             | Yes
+-------------------------------------------+----------------+----------------+-------------------------------------------------------------------------+
| Execution speed                           | Fast           | Fast           | Medium
+-------------------------------------------+----------------+----------------+-------------------------------------------------------------------------+



.. .. code-block::

..     $ otx train
