# Image Classification

Current repository includes training and evaluation tools for the image classification task. Use one of the prepared config files to train the model on:
 - ImageNet dataset (`configs/classification/imagenet_rmnet.yml`)

## Data Preparation

Assume the following structure of data:

```
    |-- data_dir
         |-- images
            image_000000.png
            image_000001.png
         train_data.txt
         test_data.txt
```

Each data file (`train_data.txt` and `test_data.txt`) describes data to train and evaluate model. Each row in a data file represents a single source in the next format: `path_to_image image_label`.


## Model Training

To train an object-detection model from scratch, run the command:
```Shell
python2 tools/models/train.py -c configs/classification/imagenet_rmnet.yml \ # path to config file
                              -t <PATH_TO_DATA_FILE> \                       # file with train data paths
                              -l <PATH_TO_LOG_DIR> \                         # directory for logging
                              -b 4 \                                         # batch size
                              -n 1 \                                         # number of target GPU devices
```

> **NOTE**: To initialize the model from the pretrained model weights, specify the `-i` key as a path to init weights and set the valid `--src_scope` key value:
>  to initialize the model after pretraining on any other classification dataset with the `DATASET_NAME` name, set `--src_scope "DATASET_NAME/rmnet"`.

The command to run the training procedure from the pretrained model:
```Shell
python2 tools/models/train.py -c configs/classification/imagenet_rmnet.yml \ # path to config file
                              -t <PATH_TO_DATA_FILE> \                       # file with train data paths
                              -l <PATH_TO_LOG_DIR> \                         # directory for logging
                              -b 4 \                                         # batch size
                              -n 1 \                                         # number of target GPU devices
                              -i <PATH_TO_INIT_WEIGHTS> \                    # initialize model weights
                              --src_scope "DATASET_NAME/rmnet"               # name of scope to load weights from
```

## Model Evaluation
To evaluate the quality of the trained Image-Classification model, prepare the test data according the  [instruction](#data-preparation).

```Shell
python2 tools/models/eval.py -c configs/classification/imagenet_rmnet.yml \ # path to config file
                             -v <PATH_TO_DATA_FILE> \                       # file with test data paths
                             -b 4 \                                         # batch size
                             -s <PATH_TO_SNAPSHOT> \                        # snapshot model weights
```
