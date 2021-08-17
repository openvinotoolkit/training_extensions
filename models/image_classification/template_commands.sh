export MODEL_TEMPLATE=`realpath ./model_templates/custom-classification/mobilenet_v3_small/template.yaml`
export WORK_DIR=`realpath ./tmp`
python ../../tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR} --do-not-load-snapshot

export DATA_DIR=`realpath ./data/cf/`
export TRAIN_DATA_ROOT=${DATA_DIR}/train
export VAL_DATA_ROOT=${DATA_DIR}/val
export TEST_DATA_ROOT=${DATA_DIR}/val

cd ${WORK_DIR}

python train.py --train-ann-files '' --train-data-roots ${TRAIN_DATA_ROOT} --val-ann-files '' --val-data-roots ${VAL_DATA_ROOT} --save-checkpoints-to ${WORK_DIR}/outputs

#export SNAP=${WORK_DIR}/outputs/best.pth
export SNAP=${WORK_DIR}/outputs/main_model/main_model.pth.tar-131

# pytorch validation
python eval.py --load-weights ${SNAP} --test-ann-files '' --test-data-roots ${TEST_DATA_ROOT} --save-metrics-to ${WORK_DIR}/metrics.yaml

#setupvars.sh should be called in advance to produce IR
python export.py --openvino --load-weights ${SNAP} --save-model-to ${WORK_DIR}/export

# IR validation
python eval.py --load-weights ${WORK_DIR}/export/main_model.xml --test-ann-files '' --test-data-roots ${TEST_DATA_ROOT} --save-metrics-to ${WORK_DIR}/metrics_ir.yaml
