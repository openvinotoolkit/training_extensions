#!/usr/bin/env bash

echo "Run evaluation"
export PYTHONPATH=${PYTHONPATH}:../../external/models/research:../../external/models/research/slim
python ../../external/models/research/object_detection/legacy/eval.py --eval_dir=./eval --checkpoint_dir=./model --logtostderr --pipeline_config_path=pipeline.config --run_once
