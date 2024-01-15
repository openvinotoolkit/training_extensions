# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import signal

from otx.utils import utils as target_file
from otx.utils.utils import append_signal_handler
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_append_signal_handler(mocker):
    # prepare
    mocker.patch("signal.raise_signal")
    spy_signal = mocker.spy(target_file.signal, "signal")
    sig_hand_1 = mocker.MagicMock()
    sig_hand_2 = mocker.MagicMock()

    # run
    append_signal_handler(signal.SIGTERM, sig_hand_1)
    append_signal_handler(signal.SIGTERM, sig_hand_2)

    old_sig_handler = signal.getsignal(signal.SIGTERM)
    old_sig_handler()

    # check
    sig_hand_1.assert_called_once()
    sig_hand_2.assert_called_once()
    assert spy_signal.call_args == ((signal.SIGTERM, signal.SIG_DFL),)
