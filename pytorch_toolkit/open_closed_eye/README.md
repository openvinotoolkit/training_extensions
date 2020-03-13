### Eye State classification

This repository contains training scripts for the eye state classifier.

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* 3.6
* PyTorch\* 1.0.1
* OpenVINO™ 2019 R1 with Python API


## Training and Evaluation

1. To train and evalute net, run

    ```bash
         python3 train.py ../../data/open_closed_eye/ 1 1 0.001 --pretrained open_closed_eye.pth
    ```

Net will be finetuned (1 epoch) on sample dataset and evalute then.

    ```bash
        epoch=0, lr=0.001
        iter:     1 loss: 0.031
        iter:     2 loss: 0.031
        iter:     3 loss: 0.031
        iter:     4 loss: 0.031
        iter:     5 loss: 0.031
        iter:     6 loss: 0.031
        iter:     7 loss: 0.031
        iter:     8 loss: 0.048
        iter:     9 loss: 0.066
        iter:    10 loss: 0.031
        iter:    11 loss: 0.076
        iter:    12 loss: 0.031
        iter:    13 loss: 0.080
        iter:    14 loss: 0.131
        iter:    15 loss: 0.031
        iter:    16 loss: 0.031
        iter:    17 loss: 0.031
        iter:    18 loss: 0.032
        iter:    19 loss: 0.114
        iter:    20 loss: 0.031
        Test accuracy: 1.0
    ```
### Convert to OpenVino 
    
    ```bash
    python3 <OpenVINO_INSTALL_DIR>/deployment_tools/model_optimizer/mo.py \
        --input_model open_close_eyes_epoch_0.onnx \
        --input_shape [1,3,32,32] \
        --mean_values [128.0,128.0,128.0] \
        --scale_values [255,255,255] 
    ```

### Demo

    ```bash
    python3 demo.py open_close_eyes_epoch_0.xml ../../data/open_closed_eye/val/
    ```