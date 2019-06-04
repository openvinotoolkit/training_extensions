# Image classification
Current repository includes training and evaluation tools for image classification task. You can use one of the prepared config files to train the model on:
 - ImageNet dataset (`$REPO_ROOT/configs/classification/imagenet_rmnet.yml`)

## Data preparation
Assume next structure of data:
<pre>
    |-- data_dir
         |-- images
            image_000000.png
            image_000001.png
         train_data.txt
         test_data.txt
</pre>
Each data file (`train_data.txt` and `test_data.txt`) describes a data to train/eval model. Each row in data file represents a single source in next format: `path_to_image image_label`.


## Model training
To train object detection model from scratch run the command:
```Shell
PYTHONPATH=$PYTHONPATH:$REPO_ROOT \
python2 tools/models/train.py -c configs/classification/imagenet_rmnet.yml \ # path to config file
                              -t <PATH_TO_DATA_FILE> \                       # file with train data paths
                              -l <PATH_TO_LOG_DIR> \                         # directory for logging
                              -b 4 \                                         # batch size
                              -n 1 \                                         # number of target GPU devices
```

**Note** If you want to initialize the model from the pre-trained model weights you should specify `-i` key as a path to init weights and set the valid `--src_scope` key value:
 - To initialize the model after pre-training on any other classification dataset with `DATASET_NAME` name set `--src_scope "DATASET_NAME/rmnet"`

Bellow the command to run the training procedure from the pre-trained model:
```Shell
PYTHONPATH=$PYTHONPATH:$REPO_ROOT \
python2 tools/models/train.py -c configs/classification/imagenet_rmnet.yml \ # path to config file
                              -t <PATH_TO_DATA_FILE> \                       # file with train data paths
                              -l <PATH_TO_LOG_DIR> \                         # directory for logging
                              -b 4 \                                         # batch size
                              -n 1 \                                         # number of target GPU devices
                              -i <PATH_TO_INIT_WEIGHTS> \                    # initialize model weights
                              --src_scope "DATASET_NAME/rmnet"               # name of scope to load weights from
```

## Model evaluation
To evaluate the quality of the trained Image Classification model you should prepare the test data according [instruction](#data-preparation).

```Shell
PYTHONPATH=$PYTHONPATH:$REPO_ROOT \
python2 tools/models/eval.py -c configs/classification/imagenet_rmnet.yml \ # path to config file
                             -v <PATH_TO_DATA_FILE> \                       # file with test data paths
                             -b 4 \                                         # batch size
                             -s <PATH_TO_SNAPSHOT> \                        # snapshot model weights
```
