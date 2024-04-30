import pytest
from otx.core.types.device import DeviceType
from otx.hpo import resource_manager as target_file
from otx.hpo.resource_manager import (
    CPUResourceManager,
    GPUResourceManager,
    XPUResourceManager,
    _cvt_comma_delimited_str_to_list,
    _remove_none_from_dict,
    get_resource_manager,
)


@pytest.fixture()
def cpu_resource_manager() -> CPUResourceManager:
    return CPUResourceManager(num_parallel_trial=4)


@pytest.fixture()
def gpu_resource_manager() -> GPUResourceManager:
    return GPUResourceManager(num_devices_per_trial=1, num_parallel_trial=4)


class TestCPUResourceManager:
    @pytest.mark.parametrize("num_parallel_trial", [1, 5, 10])
    def test_init(self, num_parallel_trial):
        CPUResourceManager(num_parallel_trial)

    @pytest.mark.parametrize("num_parallel_trial", [-1, 0])
    def test_init_with_not_positive_num_parallel_trial(self, num_parallel_trial):
        with pytest.raises(ValueError):  # noqa: PT011
            CPUResourceManager(num_parallel_trial)

    def test_reserve_resource(self, cpu_resource_manager: CPUResourceManager):
        num_parallel_trial = cpu_resource_manager._num_parallel_trial

        for i in range(num_parallel_trial):
            assert cpu_resource_manager.reserve_resource(i) == {}

        for i in range(10):
            assert cpu_resource_manager.reserve_resource(i) is None

    def test_reserve_resource_reserved_already(self, cpu_resource_manager: CPUResourceManager):
        cpu_resource_manager.reserve_resource(0)
        with pytest.raises(RuntimeError):
            cpu_resource_manager.reserve_resource(0)

    def test_release_resource(self, cpu_resource_manager: CPUResourceManager):
        cpu_resource_manager.reserve_resource(1)
        cpu_resource_manager.release_resource(1)

    def test_release_unreserved_resource(self, cpu_resource_manager: CPUResourceManager):
        cpu_resource_manager.release_resource(1)

    def test_have_available_resource(self, cpu_resource_manager: CPUResourceManager):
        num_parallel_trial = cpu_resource_manager._num_parallel_trial

        for i in range(num_parallel_trial):
            assert cpu_resource_manager.have_available_resource()
            cpu_resource_manager.reserve_resource(i)

        assert not cpu_resource_manager.have_available_resource()


class TestGPUResourceManager:
    @pytest.fixture(autouse=True)
    def setupt_test(self, mocker):
        mock_torch_cuda = mocker.patch("otx.hpo.resource_manager.torch.cuda")
        mock_torch_cuda.is_available.return_value = True
        mock_torch_cuda.device_count.return_value = 4

    def test_init(self):
        GPUResourceManager(num_devices_per_trial=1, num_parallel_trial=3)

    @pytest.mark.parametrize("num_devices_per_trial", [-1, 0])
    def test_init_not_positive_num_gpu(self, num_devices_per_trial):
        with pytest.raises(ValueError):  # noqa: PT011
            GPUResourceManager(num_devices_per_trial=num_devices_per_trial)

    @pytest.mark.parametrize("num_parallel_trial", [-1, 0])
    def test_init_wrong_available_gpu_value(self, num_parallel_trial):
        with pytest.raises(ValueError):  # noqa: PT011
            GPUResourceManager(num_parallel_trial=num_parallel_trial)

    def test_reserve_resource(self):
        num_devices_per_trial = 2
        gpu_resource_manager = GPUResourceManager(
            num_devices_per_trial=num_devices_per_trial,
            num_parallel_trial=8,
        )
        num_gpus = 4
        max_parallel = num_gpus // num_devices_per_trial

        for i in range(max_parallel):
            env = gpu_resource_manager.reserve_resource(i)
            assert env is not None
            assert "CUDA_VISIBLE_DEVICES" in env
            assert len(env["CUDA_VISIBLE_DEVICES"].split(",")) == num_devices_per_trial

        for i in range(max_parallel, max_parallel + 10):
            assert gpu_resource_manager.reserve_resource(i) is None

    def test_reserve_resource_reserved_already(self, gpu_resource_manager):
        gpu_resource_manager.reserve_resource(0)
        with pytest.raises(RuntimeError):
            gpu_resource_manager.reserve_resource(0)

    def test_release_resource(self, gpu_resource_manager):
        gpu_resource_manager.reserve_resource(1)
        gpu_resource_manager.release_resource(1)

    def test_release_unreserved_resource(self, gpu_resource_manager):
        gpu_resource_manager.release_resource(1)

    def test_have_available_resource(self):
        num_devices_per_trial = 2
        gpu_resource_manager = GPUResourceManager(
            num_devices_per_trial=num_devices_per_trial,
            num_parallel_trial=8,
        )
        num_gpus = 4
        max_parallel = num_gpus // num_devices_per_trial

        for i in range(max_parallel):
            assert gpu_resource_manager.have_available_resource()
            gpu_resource_manager.reserve_resource(i)

        for _i in range(max_parallel, max_parallel + 10):
            assert not gpu_resource_manager.have_available_resource()


class TestXPUResourceManager:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        self.mock_os = mocker.patch.object(target_file, "os")
        self.mock_torch = mocker.patch.object(target_file, "torch")

    def test_init_env_var_exist(self):
        self.mock_os.getenv.return_value = "level_zero:1,2"
        resource_manager = XPUResourceManager(num_devices_per_trial=1)
        for i in range(2):
            resource_manager.reserve_resource(i)
        assert resource_manager.reserve_resource(3) is None

    def test_init_no_env_var(self):
        self.mock_torch.xpu.device_count.return_value = 4
        resource_manager = XPUResourceManager(num_devices_per_trial=1)
        for i in range(4):
            resource_manager.reserve_resource(i)
        assert resource_manager.reserve_resource(3) is None

    def test_reserve_resource(self):
        self.mock_os.getenv.return_value = None
        self.mock_torch.xpu.device_count.return_value = 4
        resource_manager = XPUResourceManager(num_devices_per_trial=1)

        for i in range(4):
            env = resource_manager.reserve_resource(i)
            assert env is not None
            assert "ONEAPI_DEVICE_SELECTOR" in env
            assert env["ONEAPI_DEVICE_SELECTOR"] == f"level_zero:{i}"

        for i in range(4, 10):
            assert resource_manager.reserve_resource(i) is None


def test_get_resource_manager_cpu():
    manager = get_resource_manager(resource_type=DeviceType.cpu, num_parallel_trial=4)
    assert isinstance(manager, CPUResourceManager)


def test_get_resource_manager_gpu(mocker):
    mocker.patch("otx.hpo.resource_manager.torch.cuda.is_available", return_value=True)
    num_devices_per_trial = 1
    num_parallel_trial = 4
    manager = get_resource_manager(
        resource_type=DeviceType.gpu,
        num_devices_per_trial=num_devices_per_trial,
        num_parallel_trial=num_parallel_trial,
    )
    assert isinstance(manager, GPUResourceManager)


def test_get_resource_manager_xpu(mocker):
    mocker.patch.object(target_file, "is_xpu_available", return_value=True)
    mock_torch = mocker.patch.object(target_file, "torch")
    mock_torch.xpu.device_count.return_value = 4
    num_devices_per_trial = 1
    num_parallel_trial = 4
    manager = get_resource_manager(
        resource_type=DeviceType.xpu,
        num_devices_per_trial=num_devices_per_trial,
        num_parallel_trial=num_parallel_trial,
    )
    assert isinstance(manager, XPUResourceManager)


def test_get_resource_manager_wrong_resource_type():
    with pytest.raises(ValueError, match="Available resource type"):
        get_resource_manager("wrong")


def test_get_resource_manager_gpu_without_available_gpu(mocker):
    mocker.patch("otx.hpo.resource_manager.torch.cuda.is_available", return_value=False)

    manager = get_resource_manager(DeviceType.gpu)
    assert isinstance(manager, CPUResourceManager)


def test_remove_none_from_dict():
    some_dict = {"a": 1, "b": None}
    ret = _remove_none_from_dict(some_dict)
    assert ret == {"a": 1}


def test_cvt_comma_delimited_str_to_list():
    assert _cvt_comma_delimited_str_to_list("1,3,5") == [1, 3, 5]


def test_cvt_comma_delimited_str_to_list_wrong_format():
    with pytest.raises(ValueError, match="Wrong format is given"):
        _cvt_comma_delimited_str_to_list("a,3,5")
