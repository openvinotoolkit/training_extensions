# Smart classroom scenario
This repository contains TensorFlow code for deployment of person detection (PD) and action recognition (AR) models for smart classroom use-case. You can define own list of possible actions (see annotation file [format](./README_DATA.md) and steps for model training to change the list of actions) but this repository shows example for 6 action classes: standing, sitting, raising hand, writing, turned-around and lie-on-the-desk.

## Pre-requisites
- Ubuntu 16.04 / 18.04
- Python 2.7
- For Python pre-requisites refer to `requirements.txt`

## Installation
 1. Create virtual environment
 ```bash
 virtualenv venv -p python2 --prompt="(action)"
 ```

 2. Activate virtual environment and setup OpenVINO variables
 ```bash
 . venv/bin/activate
 . /opt/intel/openvino/bin/setupvars.sh
 ```
 **NOTE** Good practice is adding `. /opt/intel/openvino/bin/setupvars.sh` to the end of the `venv/bin/activate`.
 ```
 echo ". /opt/intel/openvino/bin/setupvars.sh" >> venv/bin/activate
 ```

 3. Install modules
 ```bash
 pip2 install -r requirements.txt
 ```

## Model training
Proposed repository allows to carry out the full cycle model training procedure. There are two ways to get the high accurate model:
 - Fine-tune from the proposed initial weights: `$REPO_ROOT/weights`. This way is most simple and fast due to reducing training stages to single one - training PD&AR model directly.
 - Full cycle model pre-training on classification and detection datasets and final PD&AR model training. To get most accurate model we recommend to pre-train model on the next tasks:
   1. Classification on ImageNet dataset (see classifier training [instruction](./README_CLASSIFIER.md))
   2. Detection on Pascal VOC0712 dataset (see detector training [instruction](./README_DETECTOR.md))
   3. Detection on MS COCO dataset

## Data preparation
To prepare a dataset follow the [instruction](./README_DATA.md)

## Action list definition
Current repository is configured to work with 6-class action detection task but you can easily define own set of actions. After the [data preparation](#data-preparation) step you should have the configured class mapping file. Next we will use class `IDs` from there. Then change `$REPO_ROOT/configs/action/pedestriandb_twinnet_actionnet.yml` file according you set of actions:
 1. Field `ACTIONS_MAP` maps class `IDs` of input data into final set of actions. Note, that some kind of `undefined` class (if you have it) should be placed at he end of action list (to exclude it during training).
 2. Field `VALID_ACTION_NAMES` stores names of valid actions, which you want to recognize (excluding `undefined` action).
 4. If you have the `undefined` class set field `UNDEFINED_ACTION_ID` to `ID` of this class from `ACTIONS_MAP` map. Also add this `ID` to list: `IGNORE_CLASSES`.
 4. If you plan to use the demo mode (see [header](#action-detection-model-demostration)) change colors of the actions by setting fields: `ACTION_COLORS_MAP` and `UNDEFINED_ACTION_COLOR`.
 5. You can exclude some actions from the training procedure by including them into the list `IGNORE_CLASSES` but to achieve best performance it's recommended to label all boxes with persons even the target action is undefined for them (this boxes is still useful to train person detector model part).

Bellow you can see the example of the valid field definition:
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

## Person Detection and Action Recognition model training
Assume we have a pre-trained model and want to fine-tune PD&AR model. In this case the the train procedure consists of next consistent stages:
 1. [Model training](#action-detection-model-training)
 2. (Optional) [Model evaluation](#action-detection-model-evaluation)
 3. (Optional) [Model demostration](#action-detection-model-demonstration)
 4. [Graph optimization](#action-detection-model-optimization)
 5. [Export to IR format](#export-to-ir-format)


### Action Detection model training
If you want to fine-tune the model with custom set of actions you can use the provided init weights. To do this run the command:
```Shell
PYTHONPATH=$PYTHONPATH:$REPO_ROOT \
python2 tools/models/train.py -c configs/action/pedestriandb_twinnet_actionnet.yml \ # path to config file
                              -t <PATH_TO_DATA_FILE> \                               # file with train data paths
                              -l <PATH_TO_LOG_DIR> \                                 # directory for logging
                              -b 4 \                                                 # batch size
                              -n 1 \                                                 # number of target GPU devices
                              -i <PATH_TO_INIT_WEIGHTS> \                            # initialize model weights
                              --src_scope "ActionNet/twinnet"                        # name of scope to load weights from
```

Note to continue model training (e.g. after stopping) from your snapshot you should run the same command but with key `-s <PATH_TO_SNAPSHOT>` and without specifying `--src_scope` key:
```Shell
PYTHONPATH=$PYTHONPATH:$REPO_ROOT \
python2 tools/models/train.py -c configs/action/pedestriandb_twinnet_actionnet.yml \ # path to config file
                              -t <PATH_TO_DATA_FILE> \                               # file with train data paths
                              -l <PATH_TO_LOG_DIR> \                                 # directory for logging
                              -b 4 \                                                 # batch size
                              -n 1 \                                                 # number of target GPU devices
                              -s <PATH_TO_SNAPSHOT> \                                # snapshot model weights
```

If you want to initialize the model from the weights differ than provided you should set the valid `--src_scope` key value:
 - To initialize the model after pre-training on ImageNet classification dataset set `--src_scope "ImageNetModel/rmnet"`
 - To initialize the model after pre-training on Pascal or COCO detection dataset set `--src_scope "SSD/rmnet"`

### Action Detection model evaluation
To evaluate the quality of the trained Action Detection model you should prepare the test data according [instruction](./README_DATA.md).

```Shell
PYTHONPATH=$PYTHONPATH:$REPO_ROOT \
python2 tools/models/eval.py -c configs/action/pedestriandb_twinnet_actionnet.yml \ # path to config file
                             -v <PATH_TO_DATA_FILE> \                               # file with test data paths
                             -b 4 \                                                 # batch size
                             -s <PATH_TO_SNAPSHOT> \                                # snapshot model weights
```


### Action Detection model demonstration

```Shell
PYTHONPATH=$PYTHONPATH:$REPO_ROOT \
python2 tools/models/demo.py -c configs/action/pedestriandb_twinnet_actionnet.yml \ # path to config file
                             -i <PATH_TO_VIDEO_FILE> \                              # file with video
                             -s <PATH_TO_SNAPSHOT> \                                # snapshot model weights
```

Note to scale the output screen size you can specify the `--out_scale` key with desirable scale factor: `--out_scale 0.5`

### Action Detection model optimization

```Shell
PYTHONPATH=$PYTHONPATH:$REPO_ROOT \
python2 tools/models/optimize.py -c configs/action/pedestriandb_twinnet_actionnet.yml \ # path to config file
                                 -s <PATH_TO_SNAPSHOT> \                                # snapshot model weights
                                 -o <PATH_TO_OUTPUT_DIR> \                              # directory for the output model
```

Note that the frozen graph will be stored in: `<PATH_TO_OUTPUT_DIR>/frozen.pb`.

### Export to IR format

Run model optimizer for the trained Action Detection model (OpenVINO should be installed before):
```Shell
python mo_tf.py --input_model <PATH_TO_FROZEN_GRAPH> \
                --output_dir <OUTPUT_DIR> \
                --model_name SmartClassroomActionNet
```
