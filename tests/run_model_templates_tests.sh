./tests/run_code_checks.sh || exit 1

python3 -m venv venv || exit 1
. venv/bin/activate || exit 1
pip install --upgrade pip || exit 1
pip install -e ote_cli || exit 1
pip install -e $OTE_SDK_PATH || exit 1

export PYTHONPATH=${PYTHONPATH}:`pwd`

python tests/run_model_templates_tests.py `pwd` $@ || exit 1
