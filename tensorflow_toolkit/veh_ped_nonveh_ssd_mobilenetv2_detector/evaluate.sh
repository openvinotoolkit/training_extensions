echo "Run evaluation"
python ../../external/models/research/object_detection/legacy/eval.py --eval_dir=./eval --checkpoint_dir=./model --logtostderr --pipeline_config_path=pipeline.config --run_once