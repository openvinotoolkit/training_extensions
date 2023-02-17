"""Unit tests for OTX Cli"""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import sys

import pytest

from otx.cli.tools import cli
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCli:
    @e2e_pytest_unit
    def test_cli(self):
        backup_argv = sys.argv
        # invalid inputs
        sys.argv = ["otx"]
        with pytest.raises(SystemExit) as e:
            cli.main()
        assert e.type == SystemExit, f"{e}"
        sys.argv = ["otx", None]
        with pytest.raises(SystemExit) as e:
            cli.main()
        assert e.type == SystemExit, f"{e}"
        sys.argv = ["otx", 0]
        with pytest.raises(SystemExit) as e:
            cli.main()
        assert e.type == SystemExit, f"{e}"
        sys.argv = ["otx", ""]
        with pytest.raises(SystemExit) as e:
            cli.main()
        assert e.type == SystemExit, f"{e}"
        # not a string value
        sys.argv = ["otx", -1]
        with pytest.raises(Exception) as e:
            cli.main()
        assert e.type == TypeError, f"{e}"
        sys.argv = ["otx", b"\x00"]
        with pytest.raises(Exception) as e:
            cli.main()
        assert e.type == TypeError, f"{e}"
        sys.argv = backup_argv
