echo "Run training"
python ../../external/models/research/object_detection/legacy/train.py --train_dir=./model --pipeline_config_path=pipeline.config --logtostderr