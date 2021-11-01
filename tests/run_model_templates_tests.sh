virtualenv venv || exit 1
. venv/bin/activate || exit 1
pip install -e ote_cli || exit 1
pip install -e $SC_SDK_REPO/src/ote_sdk || exit 1
echo ""
echo ""
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pytest tests/ote_cli/ --collect-only || exit 1
echo "Sleep 5 sec before actually running tests."
sleep 5
pytest tests/ote_cli/test_ote_cli_tools_segmentation.py -v -s
