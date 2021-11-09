virtualenv venv || exit 1
. venv/bin/activate || exit 1
pip install -e ote_cli || exit 1
pip install -e $SC_SDK_REPO/src/ote_sdk || exit 1
echo ""
echo ""

export PYTHONPATH=${PYTHONPATH}:`pwd`
pytest tests/ote_cli/ -v
