#!/usr/bin/env bash

echo "Convert image train/val dataset to tf records."
export PYTHONPATH=${PYTHONPATH}:../../external/models/research:../../external/models/research/slim
python create_crossroad_extra_tf_records.py --train_image_dir=./dataset/ssd_mbv2_data_train --val_image_dir=./dataset/ssd_mbv2_data_val/ --train_annotations_file=./dataset/annotation_example_train.json --val_annotations_file=./dataset/annotation_example_val.json --output_dir=tfrecords
