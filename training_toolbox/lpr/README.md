# LPRNet: License Plate Recognition 


## Train LPRNet model

To train a [LPRNet](https://arxiv.org/abs/1806.10447), jump to
training_toolbox/lpr directory. You'll see the folder with sample code
demonstrating how to train a LPRNet model.

We provide predefined configuration for: 
* Chinese license plates recognition.
  - Configuration file: [training_toolbox/lpr/chinese_lp/config.py](chinese_lp/config.py).
  - Trained model: [LPRNet 94x24](https://download.01.org/openvinotoolkit/training_toolbox_tensorflow/models/lpr/chinese_lp/license-plate-recognition-barrier-0007.zip).

As training dataset for this model [Synthetic Chinese License Plates](https://download.01.org/openvinotoolkit/training_toolbox_tensorflow/datasets/Synthetic_Chinese_License_Plates.tar.gz) dataset was used.
  
To train a model, go through the following steps:

1. Download training data and extract it in `data/synthetic_chinese_license_plates` folder. After extracting it will 
    consist from folder with training images named `crops` and text file with annotations named `annotation`. 
    
    After extracting training data archive run python script from 
    `data/synthetic_chinese_license_plates/make_train_val_split.py` to make split of 
    the whole annotations into `train` and `val` feeding him path to `data/synthetic_chinese_license_plates/annotation` 
    file from archive as an input. As a result you'll find `data/synthetic_chinese_license_plates/train`, 
    `data/synthetic_chinese_license_plates/val` annotation files with full path to images and labels in the folder 
    with extracted data.

    Then edit `training_toolbox/lpr/chinese_lp/config.py` by pointing out 
    `train.file_list_path` and `eval.file_list_path` 
    parameters in train section to paths of obtained `train` and `val` 
    annotation files accordingly.

2. To start training go to `training_toolbox/lpr` directory and type in command line:

    ```
    python3 train.py chinese_lp/config.py
    ```

3. To start evaluation process go to `training_toolbox/lpr` directory and type
    in command line:

    ```
    python3 eval.py vlp/config.py
    ```

    Before doing step 4, make sure that parameter `eval.file_list_path` in 
    `training_toolbox/lpr/chinese_lp/config.py` pointing out to file with 
    annotations to test on. Do step 4 in another terminal, so training and 
    evaluation are performed simultaneously.
    

4. Training and evaluation artifacts will be stored by default in
    `training_toolbox/lpr/chinese_lp/model`.  To visualize training and evaluation, go to
    `training_toolbox/lpr/chinese_lp` and run tensorboard with:

    ```
    tensorboard --logdir=./model
    ```

    And view results in a browser: [http://localhost:6006](http://localhost:6006).

6. When training is complete, model from the checkpoint could be infered on
    input data by running `training_toolbox/lpr/chinese_lp/infer.py`:

    ```
    python3 infer.py chinese_lp/config.py
    ```
    Input data for infer should be set via parameter `infer.file_list_path` in 
    training_toolbox/lpr/chinese_lp/config.py` and must be look like text file 
    with list of path to license plates images in format:

    ```
    path_to_lp_image1
    path_to_lp_image2
    ...
    ```

## Citation

If you find *LPRNet* useful in your research, please, consider to cite the following paper:

```
@article{icv2018lprnet,
title={LPRNet: License Plate Recognition via Deep Neural Networks},
author={Sergey Zherzdev and Alexey Gruzdev},
journal={arXiv:1806.10447},
year={2018}
}
```
