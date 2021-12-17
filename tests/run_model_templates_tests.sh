python3 -m venv venv || exit 1
. venv/bin/activate || exit 1
pip install --upgrade pip || exit 1
pip install -e ote_cli || exit 1
pip install -e $OTE_SDK_PATH || exit 1
echo ""
echo ""

export PYTHONPATH=${PYTHONPATH}:`pwd`
pytest tests/ote_cli/ -v -s --durations=0
