virtualenv venv || exit 1
. venv/bin/activate || exit 1
pip install -e ote_cli || exit 1
pip install -e $SC_SDK_REPO/src/ote_sdk || exit 1
echo ""
echo ""
pytest tests/ote_cli/ --collect-only || exit 1
echo "Sleep 5 sec before actually running tests."
sleep 5
pytest tests/ote_cli/ -v
