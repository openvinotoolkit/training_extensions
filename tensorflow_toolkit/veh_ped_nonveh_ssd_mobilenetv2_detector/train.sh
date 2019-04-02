#!/usr/bin/env bash

echo "Run training"
export PYTHONPATH=${PYTHONPATH}:../../external/models/research:../../external/models/research/slim
python ../../external/models/research/object_detection/legacy/train.py --train_dir=./model --pipeline_config_path=pipeline.config --logtostderr
