# Textile Recognition

## Description

The textile detection and recognition demo receives a video stream from a camera mounted on a textile factory
near the machine where a pattern is printed onto a cloth.
The camera should be directed onto the cloth with printed pattern, going out from the machine.

The demo receives from the camera frames one-by-one and works as follows:
* detects the "textile area" -- the area on the frame where the cloth is going out from the printing machine
* compares the pattern in the area with patterns from the pre-defined gallery
* recognizes the pattern that is printed by the machine at the moment

## Setup

### Prerequisites

* Ubuntu 16.04
* Python 3.5 or 3.6
* TensorFlow 2.0.0a0 (for training only)
* OpenVINO 2019 R1 with Python API (to infer pre-trained model only)

### Installation

1. Create virtual environment
```bash
virtualenv venv -p python3 --prompt="(textile)"
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

3. Install the modules

```
pip install -r requirements.txt
```

## Train the textile recognition model

To train the textile recognition model run the script `train.py` as follows:
```
python3 textile_recognition/train.py \
        --gallery_folder <path_to_a_file_with_list_of_gallery_images> \
        --train_dir <path to folder to save the model> \
        --loss triplet \
        --model mobilenet_v2 \
        --input_size 224 \
        --augmentation_config textile_recognition/configs/augmentation_config.json
```

The file `augmentation_config.json` contains the following parameters on gallery images augmentation:
```
{
    "apply_gray_noise": true,
    "fit_to_max_size": 0,
    "max_tiling": 4,
    "vertical_flip": true,
    "blur": true,
    "add_rot_angle": 0.1,
    "rot90": true,
    "horizontal_flip": true
}
```

Each line in the file with list of gallery images should have format
```
<path_to_image_folder>/<image_file_name> <id_of_gallery_group>
```
where `<id_of_gallery_group>` should be a number identifier to join similar (almost identical) gallery images
into groups (but in the simplest case it may be different for each line).

## Demo

### Demo Input and Output

As the input the demo receives a trained recognition model, a list of gallery images, and a list of video files
to run.

When demo works it reads frames from video files one-by-one and runs pattern recognition on the input frames
to find the gallery images that are similar to the pattern that is being printed at the moment.

The demo shows the current frame from the video in the top-left part of the demo screen.
If the textile area detection is successful, the demo
* shows the result of textile area detection as a magenta rectangle on the frame
* estimates top 10 of the gallery images that should be close to the pattern on the textile area;
    the found 10 gallery images are shown in the bottom part of the demo screen,
    the groundtruth pattern is highlighted by additional green border.
* prints info on time required for the textile pattern recognition on the top-right part of the demo screen

Hot keys:
* 'n' -- go to the next video
* 'Escape' -- exit from the demo.

### Running
To run the demo run the script `test.py` as follows:
```
python textile_recognition/test.py \
        --model_weights <path_to_model_weights> \
        --gallery <path_to_a_file_with_list_of_gallery_images> \
        --input_size 224  \
        --ie ie \
        --imshow_delay 1 \
        --test_data_type videos \
        --test_annotation_path <path_to_a_file_with_list_of_videos> \
        --test_data_path <path_to_the_root_folder_for_videos>
```
Note that
* each line in the file with list of gallery images should have format
    ```
    <path to image folder>/<image file name> <id_of_gallery_group>
    ```
    where `<image file name>` without extension will be used as a text identifier of gallery image
    (so, if `<image file name>` is `44863229-df45-4044-8f6c-d38301a5680b.png`, then the text identifier of
    the image is `44863229-df45-4044-8f6c-d38301a5680b`);    
    `<id_of_gallery_group>` should be a number identifier to join similar (almost identical) gallery images
    into groups (but in the simplest case it may be different for each line).
* each line in the file with list of videos should have format
    ```
    <path to video file> <identifier_of_gallery_image>
    ```
    where `<identifier_of_gallery_image>` is the groundtruth label of the pattern printed on cloth in the
    video -- it should be a text identifier of one of gallery images
    (e.g. `44863229-df45-4044-8f6c-d38301a5680b` from the example above),    
    and `<path to video file>` may be either absolute or relative, in the case of relative path the parameter
    `--test_data_path` is used to point the root folder for videos.
