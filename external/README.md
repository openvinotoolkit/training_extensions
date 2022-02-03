# OTE Algorithms

This folder contains sub-projects implementing [OTE SDK](../ote_sdk) Task interfaces for different algorithms.
Every sub-project is fully indepedent from each other, and each of them has its own dependencies and `init_venv.sh` script to initialize virtual environment.

## Anomaly Classification
ID | Name | Complexity (GFlops) | Model size (MB) | Path
------- | ------- | ------- | ------- | -------
ote_anomaly_classification_padim | PADIM | 3.9 | 168.4 | anomaly/anomaly_classification/configs/padim/template.yaml
ote_anomaly_classification_stfpm | STFPM | 5.6 | 21.1 | anomaly/anomaly_classification/configs/stfpm/template.yaml

## Image Classification
ID | Name | Complexity (GFlops) | Model size (MB) | Path
------- | ------- | ------- | ------- | -------
MobileNet-V3-large-0.75x | MobileNet-V3-large-0.75x | 0.32 | 2.76 | deep-object-reid/configs/ote_custom_classification/mobilenet_v3_large_075/template.yaml
Custom_Image_Classification_EfficinetNet-B0 | EfficientNet-B0 | 0.81 | 4.09 | deep-object-reid/configs/ote_custom_classification/efficientnet_b0/template.yaml

## Object Detection
ID | Name | Complexity (GFlops) | Model size (MB) | Path
------- | ------- | ------- | ------- | -------
Custom_Object_Detection_Gen3_SSD | SSD | 9.4 | 7.6 | mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_SSD/template.yaml
Custom_Object_Detection_Gen3_ATSS | ATSS | 20.6 | 9.1 | mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml
Custom_Object_Detection_Gen3_VFNet | VFNet | 457.4 | 126.0 | mmdetection/configs/ote/custom-object-detection/gen3_resnet50_VFNet/template.yaml

## Object Counting
ID | Name | Complexity (GFlops) | Model size (MB) | Path
------- | ------- | ------- | ------- | -------
Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50 | MaskRCNN-ResNet50 | 533.8 | 177.9 | mmdetection/configs/ote/custom-counting-instance-seg/resnet50_maskrcnn/template.yaml

## Semantic Segmentaion
ID | Name | Complexity (GFlops) | Model size (MB) | Path
------- | ------- | ------- | ------- | -------
Custom_Semantic_Segmentation_Lite-HRNet-18_OCR | Lite-HRNet-18 OCR | 3.45 | 4.5 | mmsegmentation/configs/ote/custom-sematic-segmentation/ocr-lite-hrnet-18/template.yaml
