Improve Small Object Detection with Image Tiling
================================================

The OpenVINO Training Extension introduces the concept of image tiling to improve the accuracy of detection algorithms and instance segmentation algorithms, particularly for small and densely packed objects in high-resolution images.

Image tiling involves dividing the original full-resolution image into multiple smaller tiles or patches. This division allows objects within the tiles to appear larger in relation to the tile size, effectively addressing the challenge of objects becoming nearly invisible in deeper layers of feature maps due to downsampling operations. For datasets where objects can be as small as 20 by 20 pixels in a 4K image, image tiling proves especially beneficial.

However, it's important to consider the trade-off associated with image tiling. By dividing a single image sample into several tiles, the number of samples for training, evaluation, and testing increases. This trade-off impacts the speed of execution, as processing more images requires additional computational resources. To strike a balance between patch size and computational efficiency, the OpenVINO Training Extension incorporates tile dataset samples and adaptive tiling parameter optimization. These features enable the proper tuning of tile size and other tiling-related parameters to ensure efficient execution without compromising accuracy.

By leveraging image tiling, the OpenVINO Training Extension empowers detection and instance segmentation algorithms to effectively detect and localize small and crowded objects in large-resolution images, ultimately leading to improved overall performance and accuracy.

Tiling Strategies 
**********************************
Below we provided an example of tiling used on one of the image from `DOTA https://captain-whu.github.io/DOTA/dataset.html`_

.. image:: ../../../../utils/images/dota_tiling_example.jpg
  :width: 800
  :alt: this image uploaded from this `source <https://captain-whu.github.io/DOTA/dataset.html>`_


In this example, the full image is cropped into 9 tiles. During training, only the tiles with annotations (bounding boxes or masks) are used for training.

During evaluation in training, only the tiles with annotations are used for evaluation, and evaluation is performed at the tile level.

During testing, each tile is processed and predicted separately. The tiles are then stitched back together to form the full image, and the tile predictions are merged to form the full image prediction.

The tiling strategy is implemented in the OpenVINO Training Extension through the following steps:

.. code-block:: 

    * Training: Create an ImageTilingDataset with annotated tiles -> Train with annotated tile images -> Evaluate on annotated tiles
    * Testing: Create an ImageTilingDataset including all tiles -> Test with all tile images -> Stitching -> Merge tile-level predictions -> Full Image Prediction

.. note::

    While running `ote eval` on models trained with tiling enabled, the evaluation will be performed on all tiles, this process including mergeing all the tile-level prediction. 
    The below context will be provided during evaluation:

    .. code-block:: 
        
        [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 650/650, 17.2 task/s, elapsed: 38s, ETA:     0s
        ==== merge: 7.326097726821899 sec ====




Enable Tiling via OTX Training CLI 
**********************************

Currently tiling is supported for both detection and instance segmentation models.

.. code-block:: 

    +-----------+-----------------------------------+-------+---------------------------------------------------------------------------+
    |    TASK   |                 ID                |  NAME |                                 BASE PATH                                 |
    +-----------+-----------------------------------+-------+---------------------------------------------------------------------------+
    | DETECTION |  Custom_Object_Detection_Gen3_SSD |  SSD  |  otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml |
    | DETECTION | Custom_Object_Detection_Gen3_ATSS |  ATSS | otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml |
    | DETECTION |   Custom_Object_Detection_YOLOX   | YOLOX | otx/algorithms/detection/configs/detection/cspdarknet_yolox/template.yaml |
    +-----------+-----------------------------------+-------+---------------------------------------------------------------------------+

    +-----------------------+----------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------------------------------+
    |          TASK         |                               ID                               |           NAME           |                                           BASE PATH                                           |
    +-----------------------+----------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------------------------------+
    | INSTANCE_SEGMENTATION |    Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50     |    MaskRCNN-ResNet50     |     otx/algorithms/detection/configs/instance_segmentation/resnet50_maskrcnn/template.yaml    |
    | INSTANCE_SEGMENTATION | Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B | MaskRCNN-EfficientNetB2B | otx/algorithms/detection/configs/instance_segmentation/efficientnetb2b_maskrcnn/template.yaml |
    +-----------------------+----------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------------------------------+

To enable tiling in OTX training, set the tiling_parameters.enable_tiling parameter to 1. Here's an example of enabling tiling for the SSD model template:

.. code-block::

    otx train Custom_Object_Detection_Gen3_SSD --train-data-roots tests/assets/small_objects --val-data-roots tests/assets/small_objects params --tiling_parameters.enable_tiling 1

.. note::

    To learn how to deploy the trained model and run the exported demo, refer to :doc:`../../tutorials/base/deploy`.

    To learn how to run the demo in CLI and visualize results, refer to :doc:`../../tutorials/base/demo`.


Customizing Tiling Parameters
*****************************

Users have the flexibility to customize the tiling process by disabling adaptive tiling parameters and setting the following parameters:

.. code-block:: 

    params --tiling_parameters.enable_adaptive_params 0 \  # disable automatic tiling parameter optimization
           --tiling_parameters.tile_size 512 \             # tile size configured to 512x512
           --tiling_parameters.tile_overlap 0.1 \          # 10% overlap between tiles

By specifying these parameters, you can disable automatic tiling parameter optimization, configure the tile size to 512x512, and set a 10% overlap between tiles.