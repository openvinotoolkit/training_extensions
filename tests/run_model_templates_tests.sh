#!/usr/bin/env bash

python3 -m venv venv || exit 1
# shellcheck source=/dev/null
. venv/bin/activate || exit 1
pip install --upgrade pip || exit 1
pip install -e ote_cli || exit 1
pip install -e ote_sdk || exit 1

python tests/run_model_templates_tests.py "$(pwd)" "$@" || exit 1
