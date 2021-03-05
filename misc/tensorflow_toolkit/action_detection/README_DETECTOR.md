# Object Detection

Current repository includes training and evaluation tools for general object-detection task. You can use one of the prepared config files to train the model on:
 - Pascal VOC0712 dataset (`configs/detection/pascal_rmnet_ssd.yml`)
 - MS COCO dataset (`configs/detection/coco_rmnet_ssd.yml`)
 - Pedestrian DB dataset (`configs/detection/pedestriandb_rmnet_ssd.yml`)

## Data Preparation

To prepare a dataset, follow the [instructions](./README_DATA.md)

## Model Training

To train an object-detection model from scratch, run the command:
```Shell
python2 tools/models/train.py -c configs/detection/pedestriandb_rmnet_ssd.yml \ # path to config file
                              -t <PATH_TO_DATA_FILE> \                          # file with train data paths
                              -l <PATH_TO_LOG_DIR> \                            # directory for logging
                              -b 4 \                                            # batch size
                              -n 1 \                                            # number of target GPU devices
```

>**NOTE**: If you want to initialize the model from the pretrained model weights,specify the `-i` key as a path to init weights and set the valid `--src_scope` key value:
 - To initialize the model after pretraining on ImageNet classification dataset, set `--src_scope "ImageNetModel/rmnet"`
 - To initialize the model after pretraining on Pascal VOC or COCO detection dataset, set `--src_scope "SSD/rmnet"`

The command to run the training procedure from the pretrained model:
```Shell
python2 tools/models/train.py -c configs/detection/pedestriandb_rmnet_ssd.yml \ # path to config file
                              -t <PATH_TO_DATA_FILE> \                          # file with train data paths
                              -l <PATH_TO_LOG_DIR> \                            # directory for logging
                              -b 4 \                                            # batch size
                              -n 1 \                                            # number of target GPU devices
                              -i <PATH_TO_INIT_WEIGHTS> \                       # initialize model weights
                              --src_scope "ImageNetModel/rmnet"                 # name of scope to load weights from
```

## Model Evaluation

To evaluate the quality of the trained Object-Detection model, prepare the test data according to the [instruction](./README_DATA.md).

```Shell
python2 tools/models/eval.py -c configs/detection/pedestriandb_rmnet_ssd.yml \ # path to config file
                             -v <PATH_TO_DATA_FILE> \                          # file with test data paths
                             -b 4 \                                            # batch size
                             -s <PATH_TO_SNAPSHOT> \                           # snapshot model weights
```
