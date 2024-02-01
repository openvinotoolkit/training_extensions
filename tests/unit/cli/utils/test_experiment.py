from pathlib import Path
from unittest.mock import MagicMock

import pytest

from otx.cli.utils import experiment as target_file
from otx.cli.utils.experiment import ResourceTracker, _check_resource, CpuUsageRecorder, GpuUsageRecorder, GIB
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestResourceTracker:
    @pytest.fixture(autouse=True)
    def _set_up(self, mocker):
        self.mock_mp = mocker.patch.object(target_file, "mp")

        self.mock_proc = mocker.MagicMock()
        self.mock_mp.Process.return_value = self.mock_proc

        self.mock_queue = mocker.MagicMock()
        self.mock_mp.Queue.return_value = self.mock_queue

    @e2e_pytest_unit
    @pytest.mark.parametrize("resource_type", ("cpu", "gpu", "all", "cpu,gpu"))
    @pytest.mark.parametrize("gpu_ids", (None, "0", "0,3"))
    @pytest.mark.parametrize("output_path", ("fake", Path("fake")))
    def test_init(self, output_path, resource_type, gpu_ids):
        ResourceTracker(output_path, resource_type, gpu_ids)

    @e2e_pytest_unit
    @pytest.mark.parametrize("resource_type", ("cpu", "gpu", "all", "cpu,gpu"))
    @pytest.mark.parametrize("gpu_ids", (None, "0", "0,3"))
    def test_start(self, resource_type, gpu_ids):
        # prepare
        if resource_type == "all":
            expected_resource_type = target_file.AVAILABLE_RESOURCE_TYPE
        else:
            expected_resource_type = [val for val in resource_type.split(",")]

        expected_gpu_ids = None
        if gpu_ids is not None:
            expected_gpu_ids = [int(idx) for idx in gpu_ids.split(",")]
            expected_gpu_ids[0] = 0

        # run
        resource_tracker = ResourceTracker("fake_output", resource_type, gpu_ids)
        resource_tracker.start()

        self.mock_proc.start.assert_called_once()  # check that a process to track resource usages starts
        # check proper resource type and gpu_ids vaues are passed to a process to run
        assert self.mock_mp.Process.call_args.kwargs["args"][1] == expected_resource_type
        assert self.mock_mp.Process.call_args.kwargs["args"][2] == expected_gpu_ids

    @e2e_pytest_unit
    def test_start_multiple_times(self):
        resource_tracker = ResourceTracker("fake_output")

        # run multiple times
        resource_tracker.start()
        resource_tracker.start()

        self.mock_proc.start.assert_called_once()  # check that a process starts once

    @e2e_pytest_unit
    def test_stop(self):
        output_path = Path("fake")

        resource_tracker = ResourceTracker(output_path)
        resource_tracker.start()
        resource_tracker.stop()

        # check that code to terminate a process is executed properly
        self.mock_queue.put.assert_called_once_with(output_path)
        self.mock_proc.join.assert_called()
        self.mock_proc.close.assert_called()

    @e2e_pytest_unit
    def test_stop_not_exit_normally(self):
        output_path = Path("fake")
        self.mock_proc.exitcode = None

        resource_tracker = ResourceTracker(output_path)
        resource_tracker.start()
        resource_tracker.stop()

        # check that code to terminate a process is executed properly
        self.mock_queue.put.assert_called_once_with(output_path)
        self.mock_proc.join.assert_called()
        # check that code to terminate a process forcibly if process doesn't exit normally
        self.mock_proc.terminate.assert_called()
        self.mock_proc.close.assert_called()

    @e2e_pytest_unit
    def test_stop_before_start(self):
        resource_tracker = ResourceTracker("fake")
        resource_tracker.stop()

        # check that code to make a process done isn't called
        self.mock_queue.put.assert_not_called()
        self.mock_proc.join.assert_not_called()
        self.mock_proc.close.assert_not_called()


class MockQueue:
    def __init__(self, output_path: str):
        self.output_path = output_path

    def empty(self):
        return False

    def get(self):
        return self.output_path


@pytest.mark.parametrize("resource_types", (["cpu"], ["gpu"], ["cpu", "gpu"]))
@e2e_pytest_unit
def test_check_resource(mocker, resource_types, tmp_path):
    # prepare
    gpu_ids = [0, 1]
    output_file = f"{tmp_path}/fake.yaml"
    mock_queue = MockQueue(output_file)

    mock_cpu_recorder = mocker.MagicMock()
    mocker.patch.object(target_file, "CpuUsageRecorder", return_value=mock_cpu_recorder)
    mock_gpu_recorder = mocker.MagicMock()
    mock_gpu_recorder_cls = mocker.patch.object(target_file, "GpuUsageRecorder", return_value=mock_gpu_recorder)

    mocker.patch.object(target_file, "yaml")
    mocker.patch.object(target_file, "time")

    # run
    _check_resource(mock_queue, resource_types, gpu_ids)

    # check the recorders record properly
    if "cpu" in resource_types:
        mock_cpu_recorder.record.assert_called_once()
    if "gpu" in resource_types:
        mock_gpu_recorder.record.assert_called_once()
        mock_gpu_recorder_cls.assert_called_once_with(gpu_ids)

    assert Path(output_file).exists()  # check a file is saved well


def test_check_resource_wrong_resource_type(mocker, tmp_path):
    # prepare
    resource_types = ["wrong"]
    output_file = f"{tmp_path}/fake.yaml"
    mock_queue = MockQueue(output_file)

    mocker.patch.object(target_file, "CpuUsageRecorder")
    mocker.patch.object(target_file, "GpuUsageRecorder")
    mocker.patch.object(target_file, "yaml")
    mocker.patch.object(target_file, "time")

    # check that ValueError is raised.
    with pytest.raises(ValueError):
        _check_resource(mock_queue, resource_types)


class TestCpuUsageRecorder:
    @pytest.fixture(autouse=True)
    def _set_up(self, mocker):
        self.mock_psutil = mocker.patch.object(target_file, "psutil")
        self.mock_virtual_memory = mocker.MagicMock()
        self.mock_psutil.virtual_memory.return_value = self.mock_virtual_memory
        self.set_mem_usage(0)
        self.set_cpu_util(0)

    def set_mem_usage(self, mem_usage: int):
        self.mock_virtual_memory.total = mem_usage
        self.mock_virtual_memory.available = 0

    def set_cpu_util(self, cpu_util: int):
        self.mock_psutil.cpu_percent.return_value = cpu_util

    @e2e_pytest_unit
    def test_init(self):
        CpuUsageRecorder()

    @e2e_pytest_unit
    def test_record_report(self):
        cpu_usage_recorder = CpuUsageRecorder()

        # record cpu usage
        cpu_usage_recorder.record()
        self.set_mem_usage(4 * GIB)
        self.set_cpu_util(40)
        cpu_usage_recorder.record()
        self.set_mem_usage(6 * GIB)
        self.set_cpu_util(60)
        cpu_usage_recorder.record()
        report = cpu_usage_recorder.report()

        # check right values are returned when calling report
        assert float(report["max_memory_usage"].split()[0]) == pytest.approx(6)
        assert float(report["avg_util"].split()[0]) == pytest.approx(50)

    @e2e_pytest_unit
    def test_report_wo_record(self):
        cpu_usage_recorder = CpuUsageRecorder()
        report = cpu_usage_recorder.report()

        assert report == {}  # if report is called without calling record, empty dict should be returned


class TestGpuUsageRecorder:
    @pytest.fixture(autouse=True)
    def _set_up(self, mocker):
        self.mock_pynvml = mocker.patch.object(target_file, "pynvml")
        self.mock_pynvml.nvmlDeviceGetCount.return_value = 8
        self.mock_nvmlDeviceGetHandleByIndex = mocker.MagicMock(side_effect=lambda val: val)
        self.mock_pynvml.nvmlDeviceGetHandleByIndex = self.mock_nvmlDeviceGetHandleByIndex

        self.gpu_usage = {}
        self.mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = self.mock_nvmlDeviceGetMemoryInfo
        self.mock_pynvml.nvmlDeviceGetUtilizationRates.side_effect = self.mock_nvmlDeviceGetUtilizationRates

        self.mock_os = mocker.patch.object(target_file, "os")
        self.mock_os.environ = {}

    def mock_nvmlDeviceGetMemoryInfo(self, gpu_idx: int):
        gpu_mem = MagicMock()
        gpu_mem.used = self.gpu_usage.get(gpu_idx, {}).get("mem", 0)
        return gpu_mem

    def mock_nvmlDeviceGetUtilizationRates(self, gpu_idx: int):
        gpu_util = MagicMock()
        gpu_util.gpu = self.gpu_usage.get(gpu_idx, {}).get("util", 0)
        return gpu_util

    def set_mem_usage(self, gpu_idx: int, mem_usage: int):
        if gpu_idx in self.gpu_usage:
            self.gpu_usage[gpu_idx]["mem"] = mem_usage
        else:
            self.gpu_usage[gpu_idx] = {"mem": mem_usage}

    def set_gpu_util(self, gpu_idx: int, gpu_util: int):
        if gpu_idx in self.gpu_usage:
            self.gpu_usage[gpu_idx]["util"] = gpu_util
        else:
            self.gpu_usage[gpu_idx] = {"util": gpu_util}

    @e2e_pytest_unit
    @pytest.mark.parametrize("gpu_to_track", ([0], [0, 4]))
    def test_init(self, mocker, gpu_to_track):
        mocker.patch.object(GpuUsageRecorder, "_get_gpu_to_track", return_value=gpu_to_track)

        GpuUsageRecorder()

        self.mock_pynvml.nvmlInit.assert_called_once()  # check nvml is initialized
        # check proper gpu handler is gotten
        for i, gpu_idx in enumerate(gpu_to_track):
            self.mock_nvmlDeviceGetHandleByIndex.call_args_list[i].args == (gpu_idx,)

    @e2e_pytest_unit
    @pytest.mark.parametrize("gpu_ids", ([0], [1, 2, 5]))
    def test_get_gpu_to_track_no_cuda_env_var(self, gpu_ids):
        gpu_usage_recorder = GpuUsageRecorder()

        assert gpu_usage_recorder._get_gpu_to_track(gpu_ids) == gpu_ids  # check right gpu indices are returned

    @e2e_pytest_unit
    @pytest.mark.parametrize("gpu_ids", ([0], [1, 2, 5]))
    def test_get_gpu_to_track_cuda_env_var(self, gpu_ids):
        cuda_visible_devices = [1, 2, 5, 7, 9, 10]
        self.mock_os.environ = {"CUDA_VISIBLE_DEVICES": ",".join(list(map(str, cuda_visible_devices)))}
        gpu_to_track = [cuda_visible_devices[i] for i in gpu_ids]

        gpu_usage_recorder = GpuUsageRecorder()

        assert gpu_usage_recorder._get_gpu_to_track(gpu_ids) == gpu_to_track  # check right gpu indices are returned

    @e2e_pytest_unit
    def test_record_report(self):
        gpu_ids = [0, 1]
        gpu_usage_recorder = GpuUsageRecorder(gpu_ids)

        # first record
        self.set_mem_usage(0, 4 * GIB)
        self.set_mem_usage(1, 6 * GIB)
        self.set_gpu_util(0, 40)
        self.set_gpu_util(1, 60)
        gpu_usage_recorder.record()

        # second record
        self.set_mem_usage(0, 6 * GIB)
        self.set_mem_usage(1, 8 * GIB)
        self.set_gpu_util(0, 60)
        self.set_gpu_util(1, 80)
        gpu_usage_recorder.record()

        report = gpu_usage_recorder.report()

        # check right values are returned
        assert float(report["gpu_0"]["avg_util"].split()[0]) == pytest.approx(50)
        assert float(report["gpu_0"]["max_mem"].split()[0]) == pytest.approx(6)
        assert float(report["gpu_1"]["avg_util"].split()[0]) == pytest.approx(70)
        assert float(report["gpu_1"]["max_mem"].split()[0]) == pytest.approx(8)
        assert float(report["total_avg_util"].split()[0]) == pytest.approx(60)
        assert float(report["total_max_mem"].split()[0]) == pytest.approx(8)

    @e2e_pytest_unit
    def test_report_wo_record(self):
        gpu_usage_recorder = GpuUsageRecorder()
        report = gpu_usage_recorder.report()

        assert report == {}  # if report is called without calling record, empty dict should be returned
