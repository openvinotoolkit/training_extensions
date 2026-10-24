# Ultralytics Backend

This code is based on the [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO package.

Licensing Information: Ultralytics YOLO models are distributed under the AGPL-3.0 license, an OSI approved license ideal for open-source research, academic, and personal projects. For commercial use, enhanced support, and tailored licensing terms, please explore flexible Ultralytics licensing options at https://www.ultralytics.com/license.

## Supported Tasks

| Task                       | Description                                         |
| -------------------------- | --------------------------------------------------- |
| Detection                  | Object detection with bounding boxes                |
| Instance Segmentation      | Instance segmentation with masks and bounding boxes |
| Multi-class Classification | Single-label image classification                   |
| Multi-label Classification | Multi-label image classification                    |
| Semantic Segmentation      | Dense pixel-wise classification                     |

## Supported Models

| Model                                                                              | Task                       | Variants      |
| ---------------------------------------------------------------------------------- | -------------------------- | ------------- |
| [YOLO26-N](./../../recipe/detection/yolo26_n.yaml)                                 | Detection                  | `yolo26n`     |
| [YOLO26-S](./../../recipe/detection/yolo26_s.yaml)                                 | Detection                  | `yolo26s`     |
| [YOLO26-M](./../../recipe/detection/yolo26_m.yaml)                                 | Detection                  | `yolo26m`     |
| [YOLO26-N Seg](./../../recipe/instance_segmentation/yolo26_n_seg.yaml)             | Instance Segmentation      | `yolo26n-seg` |
| [YOLO26-S Seg](./../../recipe/instance_segmentation/yolo26_s_seg.yaml)             | Instance Segmentation      | `yolo26s-seg` |
| [YOLO26-M Seg](./../../recipe/instance_segmentation/yolo26_m_seg.yaml)             | Instance Segmentation      | `yolo26m-seg` |
| [YOLO26-N Cls](./../../recipe/classification/multi_class_cls/yolo26_n_cls.yaml)    | Multi-class Classification | `yolo26n-cls` |
| [YOLO26-S Cls](./../../recipe/classification/multi_class_cls/yolo26_s_cls.yaml)    | Multi-class Classification | `yolo26s-cls` |
| [YOLO26-M Cls](./../../recipe/classification/multi_class_cls/yolo26_m_cls.yaml)    | Multi-class Classification | `yolo26m-cls` |
| [YOLO26-L Cls](./../../recipe/classification/multi_class_cls/yolo26_l_cls.yaml)    | Multi-class Classification | `yolo26l-cls` |
| [YOLO26-X Cls](./../../recipe/classification/multi_class_cls/yolo26_x_cls.yaml)    | Multi-class Classification | `yolo26x-cls` |
| [YOLO26-N ML Cls](./../../recipe/classification/multi_label_cls/yolo26_n_cls.yaml) | Multi-label Classification | `yolo26n-cls` |
| [YOLO26-S ML Cls](./../../recipe/classification/multi_label_cls/yolo26_s_cls.yaml) | Multi-label Classification | `yolo26s-cls` |
| [YOLO26-M ML Cls](./../../recipe/classification/multi_label_cls/yolo26_m_cls.yaml) | Multi-label Classification | `yolo26m-cls` |
| [YOLO26-L ML Cls](./../../recipe/classification/multi_label_cls/yolo26_l_cls.yaml) | Multi-label Classification | `yolo26l-cls` |
| [YOLO26-X ML Cls](./../../recipe/classification/multi_label_cls/yolo26_x_cls.yaml) | Multi-label Classification | `yolo26x-cls` |
| [YOLO26-N Sem](./../../recipe/semantic_segmentation/yolo26_n_sem.yaml)             | Semantic Segmentation      | `yolo26n-sem` |
| [YOLO26-S Sem](./../../recipe/semantic_segmentation/yolo26_s_sem.yaml)             | Semantic Segmentation      | `yolo26s-sem` |
| [YOLO26-M Sem](./../../recipe/semantic_segmentation/yolo26_m_sem.yaml)             | Semantic Segmentation      | `yolo26m-sem` |
| [YOLO26-L Sem](./../../recipe/semantic_segmentation/yolo26_l_sem.yaml)             | Semantic Segmentation      | `yolo26l-sem` |
| [YOLO26-X Sem](./../../recipe/semantic_segmentation/yolo26_x_sem.yaml)             | Semantic Segmentation      | `yolo26x-sem` |
