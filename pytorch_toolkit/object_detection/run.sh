CONFIG=`find $1 -name "*.py"`
CHECKPOINT=`find $1 -name latest.pth`
echo "CONFIGURATION FILE" $CONFIG
echo "CHEKPOINT         " $CHECKPOINT
TMP_DIR=$(mktemp -d -t ci-XXXXXXXXXX)
echo $TMP_DIR
#PyTorch
echo ""
echo "Testing model..."
OUTPUT_FILE=$TMP_DIR/res.pkl
python ../../external/mmdetection/tools/test.py $CONFIG $CHECKPOINT --out $OUTPUT_FILE
echo ""
echo "Computing Wider custom metrics values..."
python ./tools/task_specific/face_detection/wider_custom_eval.py $CONFIG $OUTPUT_FILE
#IR with detection_output
echo ""
echo "Exporting model to onnx (export_ssd.py)..."
ONNX_FILE=$TMP_DIR/model_ssd.onnx
python ../../external/mmdetection/tools/export_ssd.py $CONFIG $CHECKPOINT $ONNX_FILE > $TMP_DIR/log.export_ssd.py 2>&1
if [[ -f $ONNX_FILE ]]
then
  echo ""
  echo "Converting onnx model to IR (export_ssd.py)..."
  IR_DIR=$TMP_DIR/IR
  python ../../external/mmdetection/tools/convert_to_ir.py $CONFIG $ONNX_FILE $IR_DIR
  echo ""
  echo "Testing exported IR (export_ssd.py)..."
  IR_MODEL=$IR_DIR/model_ssd.xml
  python ~/git/model-zoo-tools/model-analyzer/model_analyzer.py -m$IR_MODEL
  python /opt/intel/openvino/deployment_tools/tools/benchmark_tool/benchmark_app.py -m $IR_MODEL -api sync
  IR_OUTPUT_FILE=$TMP_DIR/res_ir_ssd.pkl
  python ../../external/mmdetection/tools/test_exported.py $CONFIG $IR_MODEL --out $IR_OUTPUT_FILE --backend openvino --with_detection_output --do_not_normalize
  echo ""
  echo "Computing Wider custom metrics values (export_ssd.py) ..."
  python ./tools/task_specific/face_detection/wider_custom_eval.py $CONFIG $IR_OUTPUT_FILE
else
  echo "failed to export with export_ssd.py, see logs" $TMP_DIR/log.export_ssd.py
fi
##IR without detection_output
echo ""
echo "Exporting model to onnx (export.py)..."
ONNX_FILE=$TMP_DIR/model.onnx
python ../../external/mmdetection/tools/export.py $CONFIG $CHECKPOINT $ONNX_FILE > $TMP_DIR/log.export.py 2>&1
if [[ -f $ONNX_FILE ]]
then
  echo ""
  echo "Converting onnx model to IR (export.py)..."
  IR_DIR=$TMP_DIR/IR
  python ../../external/mmdetection/tools/convert_to_ir.py $CONFIG $ONNX_FILE $IR_DIR
  echo ""
  echo "Testing exported IR (export.py)..."
  IR_MODEL=$IR_DIR/model.xml
  python ~/git/model-zoo-tools/model-analyzer/model_analyzer.py -m $IR_MODEL
  python /opt/intel/openvino/deployment_tools/tools/benchmark_tool/benchmark_app.py -m $IR_MODEL -api sync
  IR_OUTPUT_FILE=$TMP_DIR/res_ir.pkl
  python ../../external/mmdetection/tools/test_exported.py $CONFIG $IR_MODEL --out $IR_OUTPUT_FILE --backend openvino --do_not_normalize
  echo ""
  echo "Computing Wider custom metrics values (export.py) ..."
  python ./tools/task_specific/face_detection/wider_custom_eval.py $CONFIG $IR_OUTPUT_FILE
else
  echo "failed to export with export_ssd.py, see logs" $TMP_DIR/log.export.py
fi
