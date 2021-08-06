export OTE_CLI_WD=/tmp/ote_cli
export OTE_REPO_DIR=`pwd`
rm -rf ${OTE_CLI_WD}
virtualenv ${OTE_CLI_WD}/venv
source ${OTE_CLI_WD}/venv/bin/activate
pip3 install -e ote_cli/
python3 tools/instantiate.py --destination ${OTE_CLI_WD}/model_templates --verbose --init-venv
source ${OTE_CLI_WD}/model_templates/OTEDetection_v2.9.1/venv/bin/activate
pytest --nbmake ote_cli/notebooks/train.ipynb -s || exit 1

cd ${OTE_CLI_WD}/model_templates/OTEDetection_v2.9.1/Object_Detection/Custom_Object_Detection_256x256
ote_train \
  --train-ann-file ${OTE_REPO_DIR}/data/airport/annotation_example_train.json \
  --train-data-roots ${OTE_REPO_DIR}/data/airport/train \
  --val-ann-file ${OTE_REPO_DIR}/data/airport/annotation_example_train.json \
  --val-data-roots ${OTE_REPO_DIR}/data/airport/train \
  --save-weights trained.pth || exit 1
ote_eval \
  --test-ann-files ${OTE_REPO_DIR}/data/airport/annotation_example_train.json \
  --test-data-roots ${OTE_REPO_DIR}/data/airport/train \
  --load-weights trained.pth || exit 1
ote_export \
  --ann-files ${OTE_REPO_DIR}/data/airport/annotation_example_train.json \
  --load-weights trained.pth \
  --save-model-to exported || exit 1
ote_export \
  --labels vehilce person non-vehicle \
  --load-weights trained.pth \
  --save-model-to exported || exit 1
cd -
