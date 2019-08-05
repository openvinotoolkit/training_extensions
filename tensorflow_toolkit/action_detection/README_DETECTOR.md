# Object detection
Current repository includes training and evaluation tools for general object detection task. You can use one of the prepared config files to train the model on:
 - Pascal VOC0712 dataset (`configs/detection/pascal_rmnet_ssd.yml`)
 - MS COCO dataset (`configs/detection/coco_rmnet_ssd.yml`)
 - Pedestrian DB dataset (`configs/detection/pedestriandb_rmnet_ssd.yml`)

## Data preparation
To prepare a dataset follow the [instruction](./README_DATA.md)

## Model training
To train object detection model from scratch run the command:
```Shell
PYTHONPATH=$PYTHONPATH:$REPO_ROOT \
python2 tools/models/train.py -c configs/detection/pedestriandb_rmnet_ssd.yml \ # path to config file
                              -t <PATH_TO_DATA_FILE> \                          # file with train data paths
                              -l <PATH_TO_LOG_DIR> \                            # directory for logging
                              -b 4 \                                            # batch size
                              -n 1 \                                            # number of target GPU devices
```

**Note** If you want to initialize the model from the pre-trained model weights you should specify `-i` key as a path to init weights and set the valid `--src_scope` key value:
 - To initialize the model after pre-training on ImageNet classification dataset set `--src_scope "ImageNetModel/rmnet"`
 - To initialize the model after pre-training on Pascal or COCO detection dataset set `--src_scope "SSD/rmnet"`

Bellow the command to run the training procedure from the pre-trained model:
```Shell
PYTHONPATH=$PYTHONPATH:$REPO_ROOT \
python2 tools/models/train.py -c configs/detection/pedestriandb_rmnet_ssd.yml \ # path to config file
                              -t <PATH_TO_DATA_FILE> \                          # file with train data paths
                              -l <PATH_TO_LOG_DIR> \                            # directory for logging
                              -b 4 \                                            # batch size
                              -n 1 \                                            # number of target GPU devices
                              -i <PATH_TO_INIT_WEIGHTS> \                       # initialize model weights
                              --src_scope "ImageNetModel/rmnet"                 # name of scope to load weights from
```

## Model evaluation
To evaluate the quality of the trained Object Detection model you should prepare the test data according [instruction](./README_DATA.md).

```Shell
PYTHONPATH=$PYTHONPATH:$REPO_ROOT \
python2 tools/models/eval.py -c configs/detection/pedestriandb_rmnet_ssd.yml \ # path to config file
                             -v <PATH_TO_DATA_FILE> \                          # file with test data paths
                             -b 4 \                                            # batch size
                             -s <PATH_TO_SNAPSHOT> \                           # snapshot model weights
```
