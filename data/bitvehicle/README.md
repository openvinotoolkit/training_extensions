BIT-Vehicle Dataset: http://iitlab.bit.edu.cn/mcislab/vehicledb/

Prepare the dataset:
1. Download the dataset from link above.
2. Unpack dataset in the directory `images`.
    ```
     $ tree
     .
     ├── bitvehicle_test.json
     ├── bitvehicle_train.json
     ├── images
     │   ├── vehicle_0000001.jpg
     │   ├── vehicle_0000002.jpg
     |   ...
     └── README.md
    ```
3. Downscale images to increase training speed.
    ```
    python <training_toolbox_tensorflow>/tools/downscale_images.py -target_size 512 <training_toolbox_tensorflow>/data/bitvehicle/images 

    ```