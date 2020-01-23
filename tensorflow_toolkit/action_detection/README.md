# Smart Classroom Scenario

This repository contains the TensorFlow\* code for deployment of person detection (PD) and action recognition (AR) models for the smart classroom use case. You can define your own list of possible actions (see annotation file [format](./README_DATA.md) and steps for model training to change the list of actions), but this repository shows example for 6 action classes: stand, sit, raise a hand, write, turn around, and lay on the desk.

## Prerequisites

- Ubuntu\* 16.04 / 18.04
- Python\* 2.7

## Installation

 1. Create virtual environment
    ```bash
    virtualenv venv -p python2 --prompt="(action)"
    ```

 2. Activate virtual environment and setup the OpenVINO™ variables:
    ```bash
    . venv/bin/activate
    . /opt/intel/openvino/bin/setupvars.sh
    ```
    > **TIP:** Good practice is adding `. /opt/intel/openvino/bin/setupvars.sh` to the end of the `venv/bin/activate`.
    ```
    echo ". /opt/intel/openvino/bin/setupvars.sh" >> venv/bin/activate
    ```

 3. Install modules
    ```bash
    pip2 install -e .
    ```

## Train a Model

There are two ways to get a high-accuracy model:
 - Fine-tune from the proposed [initial weights](https://download.01.org/opencv/openvino_training_extensions/models/action_detection/person-detection-action-recognition-0006.tar.gz). This way is the most simple and the fastest due to the reduction of training stages to a single one - training PD&AR model directly.
 - Full cycle model pretraining on classification and detection datasets and final PD&AR model training. To get the most accurate model, pretrain a model on the next tasks:
   1. Classification on the ImageNet\* dataset (see classifier training [instruction](./README_CLASSIFIER.md))
   2. Detection on Pascal VOC0712 dataset (see the detector training [instruction](./README_DETECTOR.md))
   3. Detection on the MS COCO dataset

## Data Preparation

To prepare a dataset, follow the [instructions](./README_DATA.md)

## Action List Definition

Current repository is configured to work with a 6-class action detection task, but you can easily define your own set of actions. After the [data preparation](#data-preparation) step you should have the configured class mapping file. We will use the class `IDs` from there. Then change the `configs/action/pedestriandb_twinnet_actionnet.yml` file according to the set of actions:
 1. Field `ACTIONS_MAP` maps class `IDs` of input data into final set of actions.
    > **NOTE**: If you have an `undefined` class, place it at the end of the action list to exclude it during training.
 2. Field `VALID_ACTION_NAMES` stores names of valid actions that you want to recognize (excluding the `undefined` action).
 4. If you have the `undefined` class, set the `UNDEFINED_ACTION_ID` field to `ID` of this class from the `ACTIONS_MAP` map and add this `ID` to the `IGNORE_CLASSES` list.
 4. If you plan to use the demo mode (see [header](#action-detection-model-demostration)) change colors of the actions by setting the `ACTION_COLORS_MAP` and `UNDEFINED_ACTION_COLOR` fields.
 5. You can exclude some actions from the training procedure by including them into the `IGNORE_CLASSES` list. However, to achieve the best performance, label all boxes with persons even if the target action is undefined for them, because these boxes are still useful to train the person-detector model part.

Example of the valid field definition:
```yaml
"ACTIONS_MAP": {0:  0,   # sitting --> sitting
                1:  3,   # standing --> standing
                2:  2,   # raising_hand --> raising_hand
                3:  0,   # listening --> sitting
                4:  0,   # reading --> sitting
                5:  1,   # writing --> writing
                6:  5,   # lie_on_the_desk --> lie_on_the_desk
                7:  0,   # busy --> sitting
                8:  0,   # in_group_discussions --> sitting
                9:  4,   # turned_around --> turned_around
                10: 6}   # __undefined__ --> __undefined__
"VALID_ACTION_NAMES": ["sitting", "writing", "raising_hand", "standing", "turned_around", "lie_on_the_desk"]
"UNDEFINED_ACTION_NAME": "undefined"
"UNDEFINED_ACTION_ID": 6
"IGNORE_CLASSES": [6]
"ACTION_COLORS_MAP": {0: [0, 255, 0],
                      1: [255, 0, 255],
                      2: [0, 0, 255],
                      3: [255, 0, 0],
                      4: [0, 153, 255],
                      5: [153, 153, 255]}
"UNDEFINED_ACTION_COLOR": [255, 255, 255]
```

## Person-Detection and Action-Recognition Model Training

Assume we have a pretrained model and want to fine-tune a PD&AR model. In this case, the train procedure consists of the consistent stages:
 1. [Model training](#action-detection-model-training)
 2. [Model evaluation](#action-detection-model-evaluation)
 3. [Model demonstration](#action-detection-model-demonstration)
 4. [Graph optimization](#action-detection-model-optimization)
 5. [Export to IR format](#export-to-ir-format)


### Action-Detection Model Training

If you want to fine-tune the model with a custom set of actions, use the provided init weights by running the command:
```Shell
python2 tools/models/train.py -c configs/action/pedestriandb_twinnet_actionnet.yml \ # path to config file
                              -t <PATH_TO_DATA_FILE> \                               # file with train data paths
                              -l <PATH_TO_LOG_DIR> \                                 # directory for logging
                              -b 4 \                                                 # batch size
                              -n 1 \                                                 # number of target GPU devices
                              -i <PATH_TO_INIT_WEIGHTS> \                            # initialize model weights
                              --src_scope "ActionNet/twinnet"                        # name of scope to load weights from
```

> **NOTE**: To continue model training (for example, after stopping) from your snapshot, run the same command but with the `-s <PATH_TO_SNAPSHOT>` key and without specifying the `--src_scope` key:
```Shell
python2 tools/models/train.py -c configs/action/pedestriandb_twinnet_actionnet.yml \ # path to config file
                              -t <PATH_TO_DATA_FILE> \                               # file with train data paths
                              -l <PATH_TO_LOG_DIR> \                                 # directory for logging
                              -b 4 \                                                 # batch size
                              -n 1 \                                                 # number of target GPU devices
                              -s <PATH_TO_SNAPSHOT> \                                # snapshot model weights
```

If you want to initialize the model from the weights different from the provided ones, set the valid `--src_scope` key value:
 - To initialize the model after pretraining on ImageNet classification dataset, set `--src_scope "ImageNetModel/rmnet"`
 - To initialize the model after pretraining on Pascal VOC or COCO detection dataset, set `--src_scope "SSD/rmnet"`

### Action-Detection Model Evaluation

To evaluate the quality of the trained Action Detection model, prepare the test data according to the [instructions](./README_DATA.md).

```Shell
python2 tools/models/eval.py -c configs/action/pedestriandb_twinnet_actionnet.yml \ # path to config file
                             -v <PATH_TO_DATA_FILE> \                               # file with test data paths
                             -b 4 \                                                 # batch size
                             -s <PATH_TO_SNAPSHOT> \                                # snapshot model weights
```


### Action-Detection Model Demonstration

```Shell
python2 tools/models/demo.py -c configs/action/pedestriandb_twinnet_actionnet.yml \ # path to config file
                             -i <PATH_TO_VIDEO_FILE> \                              # file with video
                             -s <PATH_TO_SNAPSHOT> \                                # snapshot model weights
```

>**NOTE**: To scale the output screen size, specify the `--out_scale` key with the desirable scale factor: `--out_scale 0.5`

### Action-Detection Model Optimization

```Shell
python2 tools/models/export.py -c configs/action/pedestriandb_twinnet_actionnet.yml \ # path to config file
                                 -s <PATH_TO_SNAPSHOT> \                                # snapshot model weights
                                 -o <PATH_TO_OUTPUT_DIR> \                              # directory for the output model
```

>**NOTE**: The frozen graph is stored at `<PATH_TO_OUTPUT_DIR>/frozen.pb`.

### Export to OpenVINO™ Intermediate Representation (IR) format

Run the Model Optimizer for the trained Actio- Detection model (OpenVINO™ should be installed before):
```Shell
python mo_tf.py --input_model <PATH_TO_FROZEN_GRAPH> \
                --output_dir <OUTPUT_DIR> \
                --model_name SmartClassroomActionNet
```
