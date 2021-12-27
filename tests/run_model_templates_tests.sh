python3 -m venv venv || exit 1
. venv/bin/activate || exit 1
pip install --upgrade pip || exit 1
pip install -e ote_cli || exit 1
pip install -e $OTE_SDK_PATH || exit 1
echo ""
echo ""

export PYTHONPATH=${PYTHONPATH}:`pwd`
pytest tests/ote_cli/ -v --durations=0 || exit 1
deactivate
echo ""
echo ""

cd external/anomaly
. /tmp/ote_cli/ANOMALY_CLASSIFICATION/venv/bin/activate
pytest tests/ -v || exit 1
