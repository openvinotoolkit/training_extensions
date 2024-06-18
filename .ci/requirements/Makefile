# INTEL CONFIDENTIAL
#
# Copyright (C) 2024 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and your use of them is governed by
# the express license under which they were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express or implied warranties,
# other than those that are expressly stated in the License.

.PHONY:	check

check:
ifeq (, $(shell command -v pip-compile))
$(error "No pip-compile in current Python environment, activate correct env or install pip-compile")
endif

all: check benchmark piptools publish tox
	@echo "Update requirements.txt for all"

benchmark: check
	@echo "Update requirements.txt for benchmark"
	@rm ./benchmark/requirements.txt
	@pip-compile --generate-hashes --output-file=./benchmark/requirements.txt ./benchmark/requirements.in

piptools: check
	@echo "Update requirements.txt for piptools"
	@rm ./piptools/requirements.txt
	@pip-compile --generate-hashes --output-file=./piptools/requirements.txt ./piptools/requirements.in

publish: check
	@echo "Update requirements.txt for publish"
	@rm ./publish/requirements.txt
	@pip-compile --generate-hashes --output-file=./publish/requirements.txt ./publish/requirements.in

tox: check
	@echo "Update requirements.txt for tox"
	@rm ./tox/requirements.txt
	@pip-compile --generate-hashes --output-file=./tox/requirements.txt ./tox/requirements.in
