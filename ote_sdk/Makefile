include ../../Makefile.variable

# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and your use of them is governed by
# the express license under which they were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express or implied warranties,
# other than those that are expressly stated in the License.

# Redefine MyPy Config location because ote_sdk has some custom configurations
MYPY_CONFIG = .mypy.ini
PYLINT_CONFIG= .pylintrc
OTE_SDK = ./ote_sdk
VIRTUALENV_DIR = .venv-ote_sdk
VIRTUALENV_BIN = $(VIRTUALENV_DIR)/bin
ACTIVATE = $(VIRTUALENV_BIN)/activate
DEV_VIRTUALENV_MARK = $(VIRTUALENV_DIR)/dev

.PHONY: clean venv venv-dev style test test-component

clean:
	@rm -rf __pycache__ .cache .pytest_cache $(VIRTUALENV_DIR) *.egg-info build dist .mypy_cache .coverage
	@find . -name .pytest_cache -type d -prune -exec rm -rf {} \;

venv: $(ACTIVATE)
$(ACTIVATE):
	@$(PYTHON_VERSION) -m venv $(VIRTUALENV_DIR)
	@. $(ACTIVATE); python -m pip install -U pip==${PIP_VERSION} setuptools==${SETUPTOOLS_VERSION} wheel==${WHEEL_VERSION}
	@. $(ACTIVATE); pip3 install -r requirements.txt


venv-dev: $(DEV_VIRTUALENV_MARK)
$(DEV_VIRTUALENV_MARK): $(ACTIVATE)
	@touch $(DEV_VIRTUALENV_MARK)
	@. $(ACTIVATE); cd $(OTE_SDK)/tests && pip3 install -r requirements.txt

style: $(DEV_VIRTUALENV_MARK)

	@echo $(ECHO_FLAKE_START)
	@. $(ACTIVATE); flake8 $(OTE_SDK) --exclude $(OTE_SDK)/tests,.venv -v --config=$(FLAKE_CONFIG) || echo "fixes needed"
	@echo $(ECHO_FLAKE_STOP)

	@echo $(ECHO_MYPY_START)
	@. $(ACTIVATE); mypy $(OTE_SDK) --exclude ".venv/.*" --config-file=$(MYPY_CONFIG)
	@echo $(ECHO_MYPY_STOP)

	@echo $(ECHO_BANDIT_START)
	@. $(ACTIVATE); bandit -r $(OTE_SDK) --exclude $(OTE_SDK)/tests,.venv
	@echo $(ECHO_BANDIT_STOP)

	@echo $(ECHO_PYLINT_START)
	@. $(ACTIVATE); pylint $(OTE_SDK) --ignore=tests,.venv,protobuf,configuration,exportable_code,reporting,augmentations,logging --rcfile=$(PYLINT_CONFIG)
	@echo $(ECHO_PYLINT_STOP)

test: test-component

test-component: $(DEV_VIRTUALENV_MARK)
	@echo $(ECHO_COMPONENT_START)
	@. $(ACTIVATE); $(PYTEST) --cov-config $(COV_CONFIG) --cov $(OTE_SDK) $(OTE_SDK)/tests
	@echo $(ECHO_COMPONENT_STOP)
