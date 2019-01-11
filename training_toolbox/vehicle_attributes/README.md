# Vehicle attributes recognition: type and color


## Train Vehicle Attributes model

To train a Vehicle Attributes model, jump to
training_toolbox/vehicle_attributes directory. You'll see the folder with a sample code
demonstrating how to train a VA model.

We provide predefined configuration for: 
* Vehicle type and color recognition.
  - Configuration file: [training_toolbox/vehicle_attributes/cars_100/config.py](cars_100/config.py).
  - Trained model: [Vehicle Attributes](https://download.01.org/openvinotoolkit/training_toolbox_tensorflow/models/vehicle_attributes/type_color/vehicle-attributes-barrier-0103.tar.gz).
    A model has two output nodes:
    . type is 4 components: one for each of 4 types - car, van, truck, bus. The resulting type is argmax of these values.
    . color is 3 components: LAB format normalized from 0 to 1

For demonstration purpose only a small dataset is uploaded [cars_100](https://download.01.org/openvinotoolkit/training_toolbox_tensorflow/datasets/cars_100.tar.gz).
It is divided by two subsets of images for training and evaluation and serves just as an example of flow.
The resulting model is trained on the internal dataset of more than 50000 images of front-facing vehicles in different
lighting and weather conditions where number of white/black/gray vehicle colors more than twice exceeds others.
It has about 74% car images, 15% truck images, 2% bus images and 9% van images.
It was evaluated on [BitVehicle](http://iitlab.bit.edu.cn/mcislab/vehicledb/) dataset:

| Metric                                    | Value    |
|-------------------------------------------|----------|
| Color mean absolute error                 | 8.68     |
| Mean absolute error of l color component  | 19.1     |
| Mean absolute error of a color component  | 3.22     |
| Mean absolute error of b color component  | 3.69     |
| Type accuracy - average                   | 0.957    |
| Type accuracy - car                       | 0.950    |
| Type accuracy - van                       | 0.967    |
| Type accuracy - truck                     | 0.953    |
| Type accuracy - bus                       | 0.961    |

To train a model, go through the following steps:

1. Download training data and extract it in `data/cars_100/images` folder. There are two files with annotations
    named `cars_100_test.json` and `cars_100_train.json` in the 'data/cars_100' directory.

    Please make sure that configuration fila `training_toolbox/vehicle_attributes/cars_100/config.py`
    contains correct paths to the annotation files.

    To use the trained model weights, download it to `training_toolbox/vehicle_attributes` folder, unpack it and set
    the corresponding configuration flag 'use_pretrained_weights' to True. Please make sure that the correct path
    to pretrained model is set in the configuration file.

2. To start training go to `training_toolbox/vehicle_attributes` directory and type in command line:

    ```
    python3 train.py cars_100/config.py
    ```
    Do step 3 in another terminal, so training and 
    evaluation are performed simultaneously.
    
3. To start evaluation process go to `training_toolbox/vehicle_attributes` directory and type
    in command line:

    ```
    python3 eval.py cars_100/config.py
    ```
4. Training and evaluation artifacts will be stored by default in
    `training_toolbox/vehicle_attributes/model`.  To visualize training and evaluation, go to
    `training_toolbox/vehicle_attributes` and run tensorboard with:

    ```
    tensorboard --logdir=./model
    ```

    And view results in a browser: [http://localhost:6006](http://localhost:6006).

6. When training is complete, model from the checkpoint could be infered on
    input data by running `training_toolbox/vehicle_attributes/infer.py`:

    ```
    python3 infer.py cars_100/config.py
    ```
7. Finally the trained model can be converted to Inference Engine format for
    optimized inference. To export, go to `training_toolbox/vehicle_attributes` folder and run:
    
    ```
    python3 export.py <path_to_model_optimizer>/mo.py cars_100/config.py
    ```
    As a result, you'll find three new artifacts in `training_toolbox/vehicle_attributes/model/ie_model` folder:
    - `graph.pb` - TensorFlow frozen graph,
    - `graph.xml` and `graph.bin` - Inference Engine representation of the model.
    
8. Model in IR format can be inferred using the following python script:
  
    ```
    python3 infer_ie.py -m model/ie_model/graph.xml -l <path_to_computer_vision_sdk>/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so <image_path>
    ```
