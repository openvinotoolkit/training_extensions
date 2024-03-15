from __future__ import annotations

import signal
from contextlib import contextmanager
from copy import copy

from otx.utils import signal as target_file
from otx.utils.signal import append_main_proc_signal_handler, append_signal_handler


@contextmanager
def register_signal_temporally(sig_num: signal.Signals):
    old_sig_handler = signal.getsignal(sig_num)
    ori_handler_arr = copy(target_file._SIGNAL_HANDLERS)
    yield
    signal.signal(sig_num, old_sig_handler)
    target_file._SIGNAL_HANDLERS = ori_handler_arr


def test_append_signal_handler(mocker):
    with register_signal_temporally(signal.SIGTERM):
        # prepare
        mocker.patch("signal.raise_signal")
        spy_signal = mocker.spy(target_file.signal, "signal")
        sig_hand_1 = mocker.MagicMock()
        sig_hand_2 = mocker.MagicMock()

        # run
        append_signal_handler(signal.SIGTERM, sig_hand_1)
        append_signal_handler(signal.SIGTERM, sig_hand_2)

        old_sig_handler = signal.getsignal(signal.SIGTERM)
        old_sig_handler(signal.SIGTERM, mocker.MagicMock())

        # check
        sig_hand_1.assert_called_once()
        sig_hand_2.assert_called_once()
        assert spy_signal.call_args == ((signal.SIGTERM, signal.SIG_DFL),)


def test_append_main_proc_signal_handler(mocker):
    with register_signal_temporally(signal.SIGTERM):
        # prepare
        mocker.patch("os.getpid", return_value=1)
        mocker.patch("signal.raise_signal")
        spy_signal = mocker.spy(target_file.signal, "signal")
        sig_hand_1 = mocker.MagicMock()
        sig_hand_2 = mocker.MagicMock()

        # run
        append_main_proc_signal_handler(signal.SIGTERM, sig_hand_1)
        append_main_proc_signal_handler(signal.SIGTERM, sig_hand_2)

        mocker.patch("os.getpid", return_value=2)
        old_sig_handler = signal.getsignal(signal.SIGTERM)
        old_sig_handler(signal.SIGTERM, mocker.MagicMock())

        # check
        sig_hand_1.assert_not_called()
        sig_hand_2.assert_not_called()
        assert spy_signal.call_args == ((signal.SIGTERM, signal.SIG_DFL),)
