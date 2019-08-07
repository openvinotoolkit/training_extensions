# Image Retrieval

## Description

This code is intended to be used for image retrieval algorithm training when probe image is transformed by following transformation:
* Random cropping
* Rotation
* Repetition
* Color distortion

## Setup

### Prerequisites

* Ubuntu 16.04
* Python 3.5 or 3.6
* TensorFlow 2.0.0a0 (for training only)
* OpenVINO 2019 R1 with Python API (to infer pre-trained model only)

### Installation

1. Create virtual environment
```bash
virtualenv venv -p python3 --prompt="(image_retrieval)"
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

## Training

To train the image retrieval model run the script `train.py` as follows:
```
python tools/train.py \
--gallery train.txt \
--test_images_folder crops_test/ \
--test_gallery test.txt \
--train_dir model \
--model mobilenet_v2 \
--augmentation_config augmentation_configs/augmentation_config.json \
--loss triplet_1.0 \
--steps_per_epoch 10000 \
--batch_size 128 \
--input_size 224 \
--dump_hard_examples \
--lr_drop_step 50000

```

The file `augmentation_config.json` contains the following parameters on gallery images augmentation:
```
{
	"duplicate_n_times":2,
	"weighted_sampling": true,
	"apply_gray_noise": true,
	"fit_to_max_size": 0,
	"max_tiling": 2,
	"vertical_flip": true,
	"blur": true,
	"add_rot_angle": 0.1,
	"rot90": true,
	"horizontal_flip": true,
	"repeat": true,
	"shuffle": true
}
```

Each line in the file with list of gallery images should have format
```
<path_to_image_folder>/<image_file_name> <id_of_gallery_group>
```
where `<id_of_gallery_group>` should be a number identifier to join similar (almost identical) gallery images
into groups (but in the simplest case it may be different for each line).

## Evaluation

To test the image retrieval model run the script `test.py` as follows:
```
python tools/test.py \
--model_weights model/weights-300000 \
--gallery test.txt \
--test_images_fodler crops_test/ \
--input_size 224 \
--model mobilenet_v2 \
--ie tf
```

## Export
To export to OpenVINO-compatible format (IR) the image retrieval model run the script `export.py` as follows:

```
python tools/export.py --model mobilenet_v2 --model_weights 20190831/triplet_1.0_train.txt/weights-115550
```

tensorflow 1.12 installed is needed instead of tensorflow 2.0. One can run export.py from another virtual environment with tensorflow 1.12