#!/bin/bash

WORK_DIR=`mktemp -d`
python3 -m venv $WORK_DIR
source $WORK_DIR/bin/activate
pip install pip --upgrade
pip install wheel
pip install ote_sdk/
pip install ote_cli/
pip install pre-commit
pip install -r ote_sdk/ote_sdk/tests/requirements.txt
echo ""
echo ""
echo ""
echo "          ##############################################"
echo "          ########                              ########"
echo "          ########  ./tests/run_code_checks.sh  ########"
echo "          ########                              ########"
echo "          ##############################################"
echo ""
pre-commit run --all-files
