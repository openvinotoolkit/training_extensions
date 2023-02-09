import pytest

from hpopt import hpo_runner
from hpopt.hpo_runner import (
    CPUResourceManager,
    GPUResourceManager,
    get_resource_manager,
    _remove_none_from_dict
)

@pytest.fixture
def cpu_resource_manager():
    return CPUResourceManager(num_parallel_trial=4)

@pytest.fixture
def gpu_resource_manager():
    return GPUResourceManager(
        num_gpu_for_single_trial = 1,
        available_gpu = "0,1,2,3"
    )

class TestCPUResourceManager:
    @pytest.mark.parametrize("num_parallel_trial", [1, 5, 10])
    def test_init(self, num_parallel_trial):
        CPUResourceManager(num_parallel_trial)

    @pytest.mark.parametrize("num_parallel_trial", [-1, 0])
    def test_init_with_not_positive_num_parallel_trial(self, num_parallel_trial):
        with pytest.raises(ValueError):
            CPUResourceManager(num_parallel_trial)

    def test_reserve_resource(self, cpu_resource_manager):
        num_parallel_trial = cpu_resource_manager._num_parallel_trial
        
        for i in range(num_parallel_trial):
            assert cpu_resource_manager.reserve_resource(i) == {}

        for i in range(10):
            assert cpu_resource_manager.reserve_resource(i) == None

    def test_reserve_resource_reserved_already(self, cpu_resource_manager):
        cpu_resource_manager.reserve_resource(0)
        with pytest.raises(RuntimeError):
            cpu_resource_manager.reserve_resource(0)

    def test_release_resource(self, cpu_resource_manager):
        cpu_resource_manager.reserve_resource(1)
        cpu_resource_manager.release_resource(1)

    def test_release_unreserved_resource(self, cpu_resource_manager):
        cpu_resource_manager.release_resource(1)

    def test_have_available_resource(self, cpu_resource_manager):
        num_parallel_trial = cpu_resource_manager._num_parallel_trial

        for i in range(num_parallel_trial):
            assert cpu_resource_manager.have_available_resource() == True
            cpu_resource_manager.reserve_resource(i)

        assert cpu_resource_manager.have_available_resource() == False

class TestGPUResourceManager:
    def test_init(self):
        GPUResourceManager(
            num_gpu_for_single_trial = 1,
            available_gpu = "0,1,2"
        )

    @pytest.mark.parametrize("num_gpu_for_single_trial", [-1, 0])
    def test_init_not_positive_num_gpu(self, num_gpu_for_single_trial):
        with pytest.raises(ValueError):
            GPUResourceManager(num_gpu_for_single_trial = num_gpu_for_single_trial)

    @pytest.mark.parametrize("available_gpu", [",", "a,b", "0,a", ""])
    def test_init_wrong_available_gpu_value(self, available_gpu):
        with pytest.raises(ValueError):
            GPUResourceManager(available_gpu = available_gpu)

    def test_reserve_resource(self):
        num_gpu_for_single_trial = 2
        num_gpus = 8
        max_parallel = num_gpus // num_gpu_for_single_trial
        gpu_resource_manager = GPUResourceManager(
            num_gpu_for_single_trial = num_gpu_for_single_trial,
            available_gpu = ",".join([str(val) for val in range(num_gpus)])
        )
        num_gpus = len(gpu_resource_manager._available_gpu)
        
        for i in range(max_parallel):
            env = gpu_resource_manager.reserve_resource(i)
            assert  env is not None
            assert "CUDA_VISIBLE_DEVICES" in env
            assert len(env["CUDA_VISIBLE_DEVICES"].split(',')) == num_gpu_for_single_trial

        for i in range(max_parallel, max_parallel+10):
            assert gpu_resource_manager.reserve_resource(i) == None

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
        num_gpu_for_single_trial = 2
        num_gpus = 8
        max_parallel = num_gpus // num_gpu_for_single_trial
        gpu_resource_manager = GPUResourceManager(
            num_gpu_for_single_trial = num_gpu_for_single_trial,
            available_gpu = ",".join([str(val) for val in range(num_gpus)])
        )
        num_gpus = len(gpu_resource_manager._available_gpu)
        
        for i in range(max_parallel):
            assert  gpu_resource_manager.have_available_resource() == True
            gpu_resource_manager.reserve_resource(i)

        for i in range(max_parallel, max_parallel+10):
            assert  gpu_resource_manager.have_available_resource() == False

def test_get_resource_manager_cpu():
    manager = get_resource_manager(
        resource_type="cpu",
        num_parallel_trial=4
    )
    assert isinstance(manager, CPUResourceManager)

def test_get_resource_manager_gpu():
    num_gpu_for_single_trial = 1
    available_gpu = "0,1,2,3"
    manager = get_resource_manager(
        resource_type="gpu",
        num_gpu_for_single_trial=num_gpu_for_single_trial,
        available_gpu=available_gpu
    )
    assert isinstance(manager, GPUResourceManager)

def test_get_resource_manager_wrong_resource_type():
    with pytest.raises(ValueError):
        get_resource_manager("wrong")

def test_get_resource_manager_gpu_without_available_gpu(mocker):
    mock_is_available = mocker.patch("hpopt.hpo_runner.torch.cuda.is_available")
    mock_is_available.return_value = False

    manager = get_resource_manager("gpu")
    assert isinstance(manager, CPUResourceManager)

def test_remove_none_from_dict():
    some_dict = {"a" : 1, "b" : None}
    ret = _remove_none_from_dict(some_dict)
    assert ret == {"a" : 1}
