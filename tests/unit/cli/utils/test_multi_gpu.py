import datetime
import os
import socket
from contextlib import closing
from copy import deepcopy

import pytest

from otx.cli.utils import multi_gpu
from otx.cli.utils.multi_gpu import (
    MultiGPUManager,
    _get_free_port,
    get_gpu_ids,
    is_multigpu_child_process,
    set_arguments_to_argv,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit

NUM_AVAILABLE_GPU = 4


@pytest.fixture(autouse=True)
def mocking_torch_device_count(mocker):
    mock_torch = mocker.patch.object(multi_gpu, "torch")
    mock_torch.cuda.device_count.return_value = NUM_AVAILABLE_GPU


@e2e_pytest_unit
def test_get_free_port():
    free_port = _get_free_port()

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", free_port))


@e2e_pytest_unit
def test_get_gpu_ids():
    gpus = []
    for i in range(0, NUM_AVAILABLE_GPU, 2):
        gpus.append(i)

    expected_result = deepcopy(gpus)
    gpus.append(NUM_AVAILABLE_GPU + 2)

    assert get_gpu_ids(",".join([str(val) for val in gpus])) == expected_result


@e2e_pytest_unit
def test_get_gpu_ids_with_wrong_args():
    with pytest.raises(ValueError):
        get_gpu_ids("abcd")


@pytest.fixture
def mock_argv_without_params(mocker):
    mock_sys = mocker.patch("otx.cli.utils.multi_gpu.sys")
    mock_sys.argv = ["--a_key", "a_val", "--b_key"]
    return mock_sys.argv


@pytest.fixture
def mock_argv_with_params(mock_argv_without_params):
    mock_argv_without_params.extend(["params", "--c_key", "c_val", "--d_key"])
    return mock_argv_without_params


@e2e_pytest_unit
def test_set_arguments_to_argv_key_exist(mock_argv_without_params):
    """Test a case where key already exists and value exists."""
    other_val = "other_val"
    set_arguments_to_argv("--a_key", other_val)

    assert mock_argv_without_params[1] == other_val


@e2e_pytest_unit
def test_set_arguments_to_argv_keys_exist(mock_argv_without_params):
    """Test a case where key already exists and value exists."""
    other_val = "other_val"
    set_arguments_to_argv(["--a_key", "-a"], other_val)

    assert mock_argv_without_params[1] == other_val


@e2e_pytest_unit
def test_set_arguments_to_argv_key_exist_none_val(mock_argv_without_params):
    """Test a case where key already exists in argv and value doesn't exists."""
    expected_result = deepcopy(mock_argv_without_params)
    set_arguments_to_argv("--a_key")

    assert mock_argv_without_params == expected_result


@e2e_pytest_unit
def test_set_arguments_to_argv_key(mock_argv_with_params):
    """Test a case where key to set doesn't exists in argv and order of key is before params and vlaue exists."""
    set_arguments_to_argv("--other_key", "other_val")

    param_idx = mock_argv_with_params.index("params")
    new_key_idx = mock_argv_with_params.index("--other_key")

    assert new_key_idx < param_idx
    assert mock_argv_with_params[new_key_idx + 1] == "other_val"


@e2e_pytest_unit
def test_set_arguments_to_argv_key_none_val(mock_argv_with_params):
    """Test a case where key to set doesn't exists in argv and order of key is before params and vlaue doesn't exist."""
    set_arguments_to_argv("--other_key")

    param_idx = mock_argv_with_params.index("params")
    new_key_idx = mock_argv_with_params.index("--other_key")

    assert new_key_idx < param_idx
    assert "--other_key" in mock_argv_with_params


@e2e_pytest_unit
def test_set_arguments_to_argv_key_after_param(mock_argv_with_params):
    """Test a case where key to set doesn't exists in argv and order of key is after params and vlaue exists."""
    set_arguments_to_argv("--other_key", "other_val", True)

    param_idx = mock_argv_with_params.index("params")
    new_key_idx = mock_argv_with_params.index("--other_key")

    assert new_key_idx > param_idx
    assert mock_argv_with_params[new_key_idx + 1] == "other_val"


@e2e_pytest_unit
def test_set_arguments_to_argv_key_after_param_non_val(mock_argv_with_params):
    """Test a case where key to set doesn't exists in argv and order of key is after params and vlaue doesn't exist."""
    set_arguments_to_argv("--other_key", after_params=True)

    param_idx = mock_argv_with_params.index("params")
    new_key_idx = mock_argv_with_params.index("--other_key")

    assert new_key_idx > param_idx
    assert "--other_key" in mock_argv_with_params


@e2e_pytest_unit
def test_is_multigpu_child_process(mocker):
    mocker.patch.object(multi_gpu.dist, "is_initialized", return_value=True)
    os.environ["LOCAL_RANK"] = "1"
    assert is_multigpu_child_process()


@e2e_pytest_unit
def test_is_multigpu_child_process_no_initialized(mocker):
    mocker.patch.object(multi_gpu.dist, "is_initialized", return_value=False)
    os.environ["LOCAL_RANK"] = "1"
    assert not is_multigpu_child_process()


@e2e_pytest_unit
def test_is_multigpu_child_process_rank0(mocker):
    mocker.patch.object(multi_gpu.dist, "is_initialized", return_value=True)
    os.environ["LOCAL_RANK"] = "0"
    assert not is_multigpu_child_process()


class TestMultiGPUManager:
    @pytest.fixture(autouse=True)
    def _set_up(self, mocker):
        self.mock_singal = mocker.patch.object(multi_gpu, "signal")
        self.mock_thread = mocker.patch.object(multi_gpu.threading, "Thread")
        self.mock_train_func = mocker.MagicMock()
        self.mock_mp = mocker.patch.object(multi_gpu, "mp")
        self.mock_process = mocker.MagicMock()
        self.mock_mp.get_context.return_value.Process = self.mock_process
        self.num_gpu = NUM_AVAILABLE_GPU
        self.mock_os = mocker.patch.object(multi_gpu, "os")
        self.mock_os.environ = {}
        self.mock_os.getpid.return_value = os.getpid()

        self.multigpu_manager = MultiGPUManager(self.mock_train_func, ",".join([str(i) for i in range(self.num_gpu)]))

    @pytest.fixture
    def process_arr(self, mocker):
        """List consists of normal process excpet last one. Last element is a process which exit abnormally."""
        normal_process = mocker.MagicMock()
        normal_process.is_alive.return_value = True
        wrong_process = mocker.MagicMock()
        wrong_process.is_alive.return_value = False
        wrong_process.exitcode = 1
        process_arr = []
        for _ in range(self.num_gpu - 2):
            process_arr.append(deepcopy(normal_process))
        process_arr.append(wrong_process)

        return process_arr

    @e2e_pytest_unit
    def test_init(self, mocker):
        elapsed_second = 180
        start_time = datetime.datetime.now() - datetime.timedelta(seconds=elapsed_second)
        MultiGPUManager(mocker.MagicMock(), "0,1", "localhost:0", start_time=start_time)

        # check torch.dist.init_process_group timeout value is adapted if elapsed time is bigger than criteria.
        assert int(self.mock_os.environ.get("TORCH_DIST_TIMEOUT", 60)) >= int(elapsed_second * 1.5)

    @e2e_pytest_unit
    @pytest.mark.parametrize("num_gpu", [4, 10])
    def test_is_available(self, mocker, num_gpu):
        multigpu_manager = MultiGPUManager(
            mocker.MagicMock(), ",".join([str(i) for i in range(num_gpu)]), "localhost:0"
        )

        assert multigpu_manager.is_available()

    @e2e_pytest_unit
    def test_is_unavailable(self, mocker):
        mock_torch = mocker.patch.object(multi_gpu, "torch")
        mock_torch.cuda.device_count.return_value = 0
        multigpu_manager = MultiGPUManager(mocker.MagicMock(), ",".join([str(i) for i in range(4)]), "localhost:0")

        assert not multigpu_manager.is_available()

    @e2e_pytest_unit
    def test_is_unavailable_by_torchrun(self, mocker):
        self.mock_os.environ = {"TORCHELASTIC_RUN_ID": "1234"}
        multigpu_manager = MultiGPUManager(mocker.MagicMock(), ",".join([str(i) for i in range(4)]), "localhost:0")

        assert not multigpu_manager.is_available()

    @e2e_pytest_unit
    def test_setup_multi_gpu_train(self, mocker):
        # prepare
        mock_initialize_multigpu_train = mocker.patch.object(MultiGPUManager, "initialize_multigpu_train")
        mock_hyper_parameters = mocker.MagicMock()
        mock_hyper_parameters.learning_parameters.learning_rate = 0.01
        mock_hyper_parameters.learning_parameters.batch_size = 8
        mock_sys = mocker.patch.object(multi_gpu, "sys")
        mock_sys.argv = []
        fake_output_path = "fake"

        # run
        self.multigpu_manager.setup_multi_gpu_train(fake_output_path, mock_hyper_parameters)

        # check spwaning child process
        assert self.mock_process.call_count == self.num_gpu - 1
        assert self.mock_process.return_value.start.call_count == self.num_gpu - 1
        assert self.mock_process.call_args.kwargs["target"] == MultiGPUManager.run_child_process
        assert self.mock_process.call_args.kwargs["args"][0] == self.mock_train_func  # train_func
        assert self.mock_process.call_args.kwargs["args"][1] == fake_output_path  # output_path
        assert self.mock_process.call_args.kwargs["args"][-1] == self.num_gpu  # num_gpu

        # check initialize multigpu trian
        mock_initialize_multigpu_train.assert_called_once()
        assert mock_initialize_multigpu_train.call_args.args[-1] == self.num_gpu  # world_size
        assert mock_initialize_multigpu_train.call_args.args[-2] == list(range(self.num_gpu))  # gpu_ids

        # check that making a thread to check child process is alive
        self.mock_thread.assert_called_once_with(target=self.multigpu_manager._check_child_processes_alive, daemon=True)
        self.mock_thread.return_value.start.assert_called_once()

        # check that register signal callback
        assert self.mock_singal.signal.call_count == 2
        mock_singal_args = self.mock_singal.signal.call_args_list
        assert mock_singal_args[0][0][0] in (self.mock_singal.SIGINT, self.mock_singal.SIGTERM)
        assert mock_singal_args[1][0][0] in (self.mock_singal.SIGINT, self.mock_singal.SIGTERM)
        assert mock_singal_args[0][0][1] == self.multigpu_manager._terminate_signal_handler
        assert mock_singal_args[1][0][1] == self.multigpu_manager._terminate_signal_handler

        # check that optimized hyper parameters are in sys.argv to pass them to child process
        assert "--learning_parameters.learning_rate" in mock_sys.argv
        assert mock_sys.argv[mock_sys.argv.index("--learning_parameters.learning_rate") + 1] == "0.01"
        assert "--learning_parameters.batch_size" in mock_sys.argv
        assert mock_sys.argv[mock_sys.argv.index("--learning_parameters.batch_size") + 1] == "8"

    @e2e_pytest_unit
    def test_check_child_processes_alive(self, mocker, process_arr):
        # prepare
        mock_kill = mocker.patch.object(multi_gpu.os, "kill")
        mocker.patch.object(multi_gpu.time, "sleep")
        mocker.patch.object(MultiGPUManager, "initialize_multigpu_train")
        self.mock_process.side_effect = process_arr

        # run
        self.multigpu_manager.setup_multi_gpu_train("fake")
        self.multigpu_manager._check_child_processes_alive()

        # check
        for p in process_arr[: self.num_gpu - 2]:
            p.kill.assert_called_once()
        self.mock_os.kill.assert_called_once_with(os.getpid(), self.mock_singal.SIGKILL)

    @e2e_pytest_unit
    def test_terminate_signal_handler(self, mocker, process_arr):
        # prepare
        mock_exit = mocker.patch.object(multi_gpu.sys, "exit")
        mocker.patch.object(MultiGPUManager, "initialize_multigpu_train")
        self.mock_process.side_effect = process_arr

        # run
        self.multigpu_manager.setup_multi_gpu_train("fake")
        self.multigpu_manager._terminate_signal_handler(2, mocker.MagicMock())

        # check
        for p in process_arr[: self.num_gpu - 2]:
            p.kill.assert_called_once()
        mock_exit.assert_called_once()

    @e2e_pytest_unit
    def test_terminate_signal_handler_not_main_thread(self, mocker, process_arr):
        # prepare
        def raise_error(*args, **kwargs):
            raise RuntimeError

        mock_exit = mocker.patch.object(multi_gpu.sys, "exit")
        mock_exit.side_effect = raise_error
        mocker.patch.object(MultiGPUManager, "initialize_multigpu_train")
        mocker.patch.object(multi_gpu.os, "getpid").return_value = os.getpid() + 1
        self.mock_process.side_effect = process_arr

        # run
        self.multigpu_manager.setup_multi_gpu_train("fake")
        with pytest.raises(RuntimeError):
            self.multigpu_manager._terminate_signal_handler(2, mocker.MagicMock())

        # check
        for p in process_arr[: self.num_gpu - 2]:
            p.kill.assert_not_called()

    @e2e_pytest_unit
    def test_finalize(self, mocker, process_arr):
        # prepare
        mocker.patch.object(MultiGPUManager, "initialize_multigpu_train")
        self.mock_process.side_effect = process_arr

        # run
        self.multigpu_manager.setup_multi_gpu_train("fake")
        self.multigpu_manager.finalize()

        # check
        for p in process_arr:
            p.join.assert_called_once()

    @e2e_pytest_unit
    def test_finalize_still_running_child_process(self, mocker, process_arr):
        # prepare
        mocker.patch.object(MultiGPUManager, "initialize_multigpu_train")
        self.mock_process.side_effect = process_arr
        for p in process_arr:
            p.exitcode = None
            p.join.return_value = None

        # run
        self.multigpu_manager.setup_multi_gpu_train("fake")
        self.multigpu_manager.finalize()

        # check
        for p in process_arr:
            p.join.assert_called_once()
            p.kill.assert_called_once()

    @e2e_pytest_unit
    def test_finalize_before_spawn(self, mocker, process_arr):
        # prepare
        mocker.patch.object(MultiGPUManager, "initialize_multigpu_train")
        self.mock_process.side_effect = process_arr

        # run
        self.multigpu_manager.setup_multi_gpu_train("fake")
        self.multigpu_manager.finalize()

    @e2e_pytest_unit
    def test_initialize_multigpu_train(self, mocker):
        # prepare
        mocker.patch.object(multi_gpu.dist, "get_world_size", return_value=2)
        mocker.patch.object(multi_gpu.dist, "get_rank", return_value=0)

        # run
        MultiGPUManager.initialize_multigpu_train(
            rdzv_endpoint="localhost:1234",
            rank=0,
            local_rank=0,
            gpu_ids=[0, 1],
            world_size=2,
        )

        # check
        assert self.mock_os.environ["MASTER_ADDR"] == "localhost"
        assert self.mock_os.environ["MASTER_PORT"] == "1234"
        assert self.mock_os.environ["LOCAL_WORLD_SIZE"] == "2"
        assert self.mock_os.environ["WORLD_SIZE"] == "2"
        assert self.mock_os.environ["LOCAL_RANK"] == "0"
        assert self.mock_os.environ["RANK"] == "0"

    @e2e_pytest_unit
    def test_run_child_process(self, mocker):
        # prepare
        mock_set_start_method = mocker.patch.object(multi_gpu.mp, "set_start_method")
        mock_sys = mocker.patch.object(multi_gpu, "sys")
        mock_sys.argv = ["--gpus", "0,1"]
        output_path = "mock_output_path"
        rdzv_endpoint = "localhost:1234"
        mock_initialize_multigpu_train = mocker.patch.object(MultiGPUManager, "initialize_multigpu_train")
        mock_threading = mocker.patch.object(multi_gpu, "threading")
        mock_train_func = mocker.MagicMock()

        # run
        MultiGPUManager.run_child_process(
            train_func=mock_train_func,
            output_path=output_path,
            rdzv_endpoint=rdzv_endpoint,
            rank=0,
            local_rank=0,
            gpu_ids=[0, 1],
            world_size=4,
        )

        # check
        assert mock_set_start_method.call_args.kwargs["method"] is None
        assert "--gpus" not in mock_sys.argv
        for output_arg_key in ["-o", "--output", False]:
            if output_arg_key in mock_sys.argv:
                break
        assert output_arg_key is not False, "There arn't both '-o' and '--output'."
        assert mock_sys.argv[mock_sys.argv.index(output_arg_key) + 1] == output_path
        assert "--rdzv-endpoint" in mock_sys.argv
        assert mock_sys.argv[mock_sys.argv.index("--rdzv-endpoint") + 1] == rdzv_endpoint
        mock_initialize_multigpu_train.assert_called_once()
        mock_threading.Thread.assert_called_once_with(target=MultiGPUManager.check_parent_processes_alive, daemon=True)
        mock_threading.Thread.call_args.return_value.start.assert_called_once
        mock_train_func.assert_called_once()

    @e2e_pytest_unit
    def test_check_parent_processes_alive(self, mocker):
        # prepare
        mocker.patch.object(multi_gpu, "time")
        mock_cur_process = mocker.MagicMock()
        mocker.patch.object(multi_gpu.psutil, "Process", return_value=mock_cur_process)
        mock_cur_process.parent.return_value.is_running.return_value = False

        # run
        MultiGPUManager.check_parent_processes_alive()

        # check
        mock_cur_process.kill.assert_called_once()
