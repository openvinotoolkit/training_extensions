import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from unittest.mock import MagicMock

import pytest

import otx
from otx.api.configuration.helper import create as create_conf_hp
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.cli.registry import find_and_parse_model_template
from otx.cli.utils.hpo import (
    HpoCallback,
    HpoDataset,
    HpoRunner,
    TaskEnvironmentManager,
    TaskManager,
    Trainer,
    get_best_hpo_weight,
    run_hpo,
    run_trial,
)
from otx.hpo.hpo_base import TrialStatus
from tests.test_suite.e2e_test_system import e2e_pytest_unit

CLASSIFCATION_TASK = {TaskType.CLASSIFICATION}
DETECTION_TASK = {TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION, TaskType.ROTATED_DETECTION}
SEGMENTATION_TASK = {TaskType.SEGMENTATION}
ANOMALY_TASK = {TaskType.ANOMALY_CLASSIFICATION, TaskType.ANOMALY_DETECTION, TaskType.ANOMALY_SEGMENTATION}
MMCV_TASK = CLASSIFCATION_TASK | DETECTION_TASK | SEGMENTATION_TASK
ALL_TASK = MMCV_TASK | ANOMALY_TASK
OTX_ROOT_PATH = Path(otx.__file__).parent


class TestTaskManager:
    @e2e_pytest_unit
    @pytest.mark.parametrize("task", MMCV_TASK)
    def test_is_mmcv_framework_task(self, task: TaskType):
        task_manager = TaskManager(task)
        assert task_manager.is_mmcv_framework_task()

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", ANOMALY_TASK)
    def test_is_not_mmcv_framework_task(self, task: TaskType):
        task_manager = TaskManager(task)
        assert not task_manager.is_mmcv_framework_task()

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", CLASSIFCATION_TASK)
    def test_is_cls_framework_task(self, task: TaskType):
        task_manager = TaskManager(task)
        assert task_manager.is_cls_framework_task()

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", ALL_TASK - CLASSIFCATION_TASK)
    def test_is_not_cls_framework_task(self, task: TaskType):
        task_manager = TaskManager(task)
        assert not task_manager.is_cls_framework_task()

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", DETECTION_TASK)
    def test_is_det_framework_task(self, task: TaskType):
        task_manager = TaskManager(task)
        assert task_manager.is_det_framework_task()

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", ALL_TASK - DETECTION_TASK)
    def test_is_not_det_framework_task(self, task: TaskType):
        task_manager = TaskManager(task)
        assert not task_manager.is_det_framework_task()

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", SEGMENTATION_TASK)
    def test_is_seg_framework_task(self, task: TaskType):
        task_manager = TaskManager(task)
        assert task_manager.is_seg_framework_task()

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", ALL_TASK - SEGMENTATION_TASK)
    def test_is_not_seg_framework_task(self, task: TaskType):
        task_manager = TaskManager(task)
        assert not task_manager.is_seg_framework_task()

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", ANOMALY_TASK)
    def test_is_anomaly_framework_task(self, task: TaskType):
        task_manager = TaskManager(task)
        assert task_manager.is_anomaly_framework_task()

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", ALL_TASK - ANOMALY_TASK)
    def test_is_not_anomaly_framework_task(self, task: TaskType):
        task_manager = TaskManager(task)
        assert not task_manager.is_anomaly_framework_task()

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", MMCV_TASK)
    def test_get_mmcv_batch_size_name(self, task: TaskType):
        task_manager = TaskManager(task)
        assert task_manager.get_batch_size_name() == "learning_parameters.batch_size"

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", ANOMALY_TASK)
    def test_get_anomaly_batch_size_name(self, task: TaskType):
        task_manager = TaskManager(task)
        assert task_manager.get_batch_size_name() == "learning_parameters.train_batch_size"

    @e2e_pytest_unit
    def test_get_unknown_task_batch_size_name(self, mocker):
        mock_func1 = mocker.patch.object(TaskManager, "is_mmcv_framework_task")
        mock_func1.return_value = False
        mock_func2 = mocker.patch.object(TaskManager, "is_anomaly_framework_task")
        mock_func2.return_value = False

        task_manager = TaskManager(mocker.MagicMock())

        with pytest.raises(RuntimeError):
            task_manager.get_batch_size_name()

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", MMCV_TASK)
    def test_get_mmcv_epoch_name(self, task: TaskType):
        task_manager = TaskManager(task)
        assert task_manager.get_epoch_name() == "num_iters"

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", ANOMALY_TASK)
    def test_get_anomaly_epoch_name(self, task: TaskType):
        task_manager = TaskManager(task)
        assert task_manager.get_epoch_name() == "max_epochs"

    @e2e_pytest_unit
    def test_get_unknown_task_epoch_name(self, mocker):
        mock_func1 = mocker.patch.object(TaskManager, "is_mmcv_framework_task")
        mock_func1.return_value = False
        mock_func2 = mocker.patch.object(TaskManager, "is_anomaly_framework_task")
        mock_func2.return_value = False

        task_manager = TaskManager(mocker.MagicMock())

        with pytest.raises(RuntimeError):
            task_manager.get_epoch_name()

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", MMCV_TASK)
    def test_copy_weight(self, task: TaskType):
        task_manager = TaskManager(task)
        fake_model_weight = Path("temp_epoch_3.pth")
        with TemporaryDirectory() as src_dir, TemporaryDirectory() as det_dir:
            weight_in_src = src_dir / fake_model_weight
            weight_in_det = det_dir / fake_model_weight
            weight_in_src.write_text("fake")
            task_manager.copy_weight(src_dir, det_dir)

            assert weight_in_det.exists()

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", MMCV_TASK)
    def test_get_latest_weight(self, task: TaskType):
        task_manager = TaskManager(task)

        with TemporaryDirectory() as work_dir:
            for i in range(1, 10):
                (work_dir / Path(f"epoch_{i}.pth")).write_text("fake")

            latest_model_weight = work_dir / Path("epoch_10.pth")
            latest_model_weight.write_text("fake")
            assert task_manager.get_latest_weight(work_dir) == str(latest_model_weight)


def get_template_path(template_dir: str) -> Path:
    task_config_dir = OTX_ROOT_PATH / "algorithms" / template_dir
    return list(task_config_dir.glob("**/template.yaml"))[0]


def make_task_env(template_path: str) -> TaskEnvironment:
    template = find_and_parse_model_template(template_path)
    return TaskEnvironment(template, None, create_conf_hp(template.hyper_parameters.data), MagicMock())


@pytest.fixture(scope="module")
def cls_template_path() -> str:
    return str(get_template_path("classification/configs"))


@pytest.fixture(scope="module")
def det_template_path() -> str:
    return str(get_template_path("detection/configs/detection"))


@pytest.fixture(scope="module")
def seg_template_path() -> str:
    return str(get_template_path("segmentation/configs"))


@pytest.fixture(scope="module")
def anomaly_template_path() -> str:
    return str(OTX_ROOT_PATH / "algorithms/anomaly/configs/classification/stfpm/template.yaml")


@pytest.fixture(scope="module")
def cls_task_env(cls_template_path):
    return make_task_env(cls_template_path)


@pytest.fixture(scope="module")
def det_task_env(det_template_path) -> TaskEnvironment:
    return make_task_env(det_template_path)


@pytest.fixture(scope="module")
def seg_task_env(seg_template_path) -> TaskEnvironment:
    return make_task_env(seg_template_path)


@pytest.fixture(scope="module")
def anomaly_task_env(anomaly_template_path) -> TaskEnvironment:
    return make_task_env(anomaly_template_path)


@pytest.fixture
def mmcv_task_env(cls_task_env, det_task_env, seg_task_env) -> List[TaskEnvironment]:
    return [cls_task_env, det_task_env, seg_task_env]


@pytest.fixture
def all_task_env(cls_task_env, det_task_env, seg_task_env, anomaly_task_env) -> List[TaskEnvironment]:
    return [cls_task_env, det_task_env, seg_task_env, anomaly_task_env]


@pytest.fixture
def mock_environment():
    MockTaskEnv = MagicMock(spec=TaskEnvironment)
    return MockTaskEnv()


@pytest.fixture(scope="module")
def action_template_path() -> str:
    return str(get_template_path("action"))


@pytest.fixture(scope="module")
def action_task_env(action_template_path) -> TaskEnvironment:
    return make_task_env(action_template_path)


class TestTaskEnvironmentManager:
    @pytest.fixture(autouse=True)
    def _make_mock_task_env(self, mock_environment):
        self._mock_environment = mock_environment

    @e2e_pytest_unit
    def test_init(self, all_task_env):
        for task_env in all_task_env:
            TaskEnvironmentManager(task_env)

    @e2e_pytest_unit
    def test_get_task(self, cls_task_env, det_task_env, seg_task_env):
        task_env = TaskEnvironmentManager(cls_task_env)
        assert task_env.get_task() == TaskType.CLASSIFICATION

        task_env = TaskEnvironmentManager(det_task_env)
        assert task_env.get_task() == TaskType.DETECTION

        task_env = TaskEnvironmentManager(seg_task_env)
        assert task_env.get_task() == TaskType.SEGMENTATION

    @e2e_pytest_unit
    def test_get_model_template(
        self, cls_task_env, det_task_env, seg_task_env, cls_template_path, det_template_path, seg_template_path
    ):
        task_env = TaskEnvironmentManager(cls_task_env)
        assert task_env.get_model_template() == find_and_parse_model_template(cls_template_path)

        task_env = TaskEnvironmentManager(det_task_env)
        assert task_env.get_model_template() == find_and_parse_model_template(det_template_path)

        task_env = TaskEnvironmentManager(seg_task_env)
        assert task_env.get_model_template() == find_and_parse_model_template(seg_template_path)

    @e2e_pytest_unit
    def test_get_model_template_path(
        self, cls_task_env, det_task_env, seg_task_env, cls_template_path, det_template_path, seg_template_path
    ):
        task_env = TaskEnvironmentManager(cls_task_env)
        assert task_env.get_model_template_path() == cls_template_path

        task_env = TaskEnvironmentManager(det_task_env)
        assert task_env.get_model_template_path() == det_template_path

        task_env = TaskEnvironmentManager(seg_task_env)
        assert task_env.get_model_template_path() == seg_template_path

    @e2e_pytest_unit
    def test_set_hyper_parameter_using_str_key(self):
        task_env = TaskEnvironmentManager(self._mock_environment)
        hyper_parameter = {"a.b.c.d": 1, "e.f.g.h": 2}

        task_env.set_hyper_parameter_using_str_key(hyper_parameter)

        env_hp = self._mock_environment.get_hyper_parameters()

        assert env_hp.a.b.c.d == hyper_parameter["a.b.c.d"]
        assert env_hp.e.f.g.h == hyper_parameter["e.f.g.h"]

    @e2e_pytest_unit
    def test_get_dict_type_hyper_parameter(self):
        learning_parameters = self._mock_environment.get_hyper_parameters().learning_parameters
        learning_parameters.parameters = ["a", "b"]
        learning_parameters.a = 1
        learning_parameters.b = 2

        task_env = TaskEnvironmentManager(self._mock_environment)
        dict_hp = task_env.get_dict_type_hyper_parameter()

        assert dict_hp["learning_parameters.a"] == 1
        assert dict_hp["learning_parameters.b"] == 2

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", ALL_TASK)
    def test_get_max_epoch(self, task):
        max_epoch = 10
        self._mock_environment.model_template.task_type = task
        learning_parameters = self._mock_environment.get_hyper_parameters().learning_parameters
        setattr(learning_parameters, TaskManager(task).get_epoch_name(), max_epoch)

        task_env = TaskEnvironmentManager(self._mock_environment)

        assert task_env.get_max_epoch() == max_epoch

    @e2e_pytest_unit
    def test_save_mmcv_initial_weight(self, mmcv_task_env):
        for task_env in mmcv_task_env:
            task_env.model = None
            task_env = TaskEnvironmentManager(task_env)
            assert not task_env.save_initial_weight("fake_path")

    @e2e_pytest_unit
    def test_save_anomaly_initial_weight(self, mocker, anomaly_task_env):
        def mock_save_model_data(model, save_path: str):
            (Path(save_path) / "weights.pth").write_text("fake")

        mocker.patch.object(TaskEnvironmentManager, "get_train_task")
        mocker.patch("otx.cli.utils.hpo.save_model_data", mock_save_model_data)
        with TemporaryDirectory() as tmp_dir:
            anomaly_task_env.model = None
            task_env = TaskEnvironmentManager(anomaly_task_env)
            save_path = Path(tmp_dir) / "init.pth"
            assert task_env.save_initial_weight(str(save_path))
            assert save_path.exists()

    @e2e_pytest_unit
    def test_loaded_inital_weight(self, mocker, all_task_env):
        def mock_save_model_data(model, save_path: str):
            (Path(save_path) / "weights.pth").write_text("fake")

        mocker.patch.object(TaskEnvironmentManager, "get_train_task")
        mocker.patch("otx.cli.utils.hpo.save_model_data", mock_save_model_data)
        with TemporaryDirectory() as tmp_dir:
            for task_env in all_task_env:
                task_env.model = mocker.MagicMock()
                task_env = TaskEnvironmentManager(task_env)
                save_path = Path(tmp_dir) / "init.pth"
                assert task_env.save_initial_weight(str(save_path))
                assert save_path.exists()

    @e2e_pytest_unit
    def test_get_train_task(self, mocker, all_task_env):
        mock_func = mocker.patch("otx.cli.utils.hpo.get_impl_class")

        for task_env in all_task_env:
            mock_class = mocker.MagicMock()
            mock_func.return_vlaue = mock_class
            task_env = TaskEnvironmentManager(task_env)
            task_env.get_train_task()

            mock_class.assert_not_called()

    @e2e_pytest_unit
    def test_get_mmcv_batch_size_name(self, mmcv_task_env):
        for task_env in mmcv_task_env:
            task_env = TaskEnvironmentManager(task_env)
            assert task_env.get_batch_size_name() == "learning_parameters.batch_size"

    @e2e_pytest_unit
    def test_get_anomaly_batch_size_name(self, anomaly_task_env):
        task_env = TaskEnvironmentManager(anomaly_task_env)
        assert task_env.get_batch_size_name() == "learning_parameters.train_batch_size"

    @e2e_pytest_unit
    def test_load_model_weight(self, mocker, all_task_env):
        mock_func = mocker.patch("otx.cli.utils.hpo.read_model")

        for task_env in all_task_env:
            mock_class = mocker.MagicMock()
            mock_func.return_value = mock_class
            task_manager = TaskEnvironmentManager(task_env)
            task_manager.load_model_weight("fake", mocker.MagicMock())
            assert task_env.model == mock_class

    @e2e_pytest_unit
    def test_resume_model_weight(self, mocker, all_task_env):
        mock_func = mocker.patch("otx.cli.utils.hpo.read_model")

        for task_env in all_task_env:
            mock_class = mocker.MagicMock()
            mock_func.return_value = mock_class
            task_manager = TaskEnvironmentManager(task_env)
            task_manager.resume_model_weight("fake", mocker.MagicMock())
            assert task_env.model == mock_class
            assert mock_class.model_adapters["resume"]

    @e2e_pytest_unit
    def test_get_new_model_entity(self, all_task_env):
        for task_env in all_task_env:
            task_manager = TaskEnvironmentManager(task_env)
            model_entity = task_manager.get_new_model_entity()
            assert isinstance(model_entity, ModelEntity)

    @e2e_pytest_unit
    def test_set_epoch(self, all_task_env):
        epoch = 123
        for task_env in all_task_env:
            task_manager = TaskEnvironmentManager(task_env)
            task_manager.set_epoch(epoch)
            assert task_manager.get_max_epoch() == epoch


class TestHpoRunner:
    @e2e_pytest_unit
    def test_init(self, all_task_env):
        for task_env in all_task_env:
            HpoRunner(task_env, 100, 10, "fake_path")

    @e2e_pytest_unit
    @pytest.mark.parametrize("train_dataset_size,val_dataset_size", [(0, 10), (10, 0), (-1, -1)])
    def test_init_wrong_dataset_size(self, cls_task_env, train_dataset_size, val_dataset_size):
        with pytest.raises(ValueError):
            HpoRunner(cls_task_env, train_dataset_size, val_dataset_size, "fake_path", 4)

    @e2e_pytest_unit
    @pytest.mark.parametrize("hpo_time_ratio", [-3, 0])
    def test_init_wrong_hpo_time_ratio(self, cls_task_env, hpo_time_ratio):
        with pytest.raises(ValueError):
            HpoRunner(cls_task_env, 100, 10, "fake_path", hpo_time_ratio)

    @e2e_pytest_unit
    def test_run_hpo(self, mocker, cls_task_env):
        cls_task_env.model = None
        hpo_runner = HpoRunner(cls_task_env, 100, 10, "fake_path")
        mock_run_hpo_loop = mocker.patch("otx.cli.utils.hpo.run_hpo_loop")
        mock_hb = mocker.patch("otx.cli.utils.hpo.HyperBand")

        hpo_runner.run_hpo(mocker.MagicMock(), {"fake", "fake"})

        mock_run_hpo_loop.assert_called()  # call hpo_loop to run HPO
        mock_hb.assert_called()  # make hyperband

    @e2e_pytest_unit
    def test_run_hpo_w_dataset_smaller_than_batch(self, mocker, cls_task_env):
        cls_task_env.model = None
        hpo_runner = HpoRunner(cls_task_env, 2, 10, "fake_path")
        mock_run_hpo_loop = mocker.patch("otx.cli.utils.hpo.run_hpo_loop")
        mock_hb = mocker.patch("otx.cli.utils.hpo.HyperBand")

        hpo_runner.run_hpo(mocker.MagicMock(), {"fake", "fake"})

        mock_run_hpo_loop.assert_called()  # call hpo_loop to run HPO
        mock_hb.assert_called()  # make hyperband


class TestTrainer:
    @e2e_pytest_unit
    def test_init(self, mocker, cls_template_path):
        Trainer(
            hp_config={"configuration": {"iterations": 10}},
            report_func=mocker.stub(),
            model_template=find_and_parse_model_template(cls_template_path),
            data_roots={"fake": "fake"},
            task_type=TaskType.CLASSIFICATION,
            hpo_workdir="fake",
            initial_weight_name="fake",
            metric="fake",
        )

    @e2e_pytest_unit
    def test_run(self, mocker, cls_template_path):
        with TemporaryDirectory() as tmp_dir:
            # prepare
            trial_id = "1"
            weight_format = "epoch_{}.pth"
            hpo_workdir = Path(tmp_dir) / "hpo_dir"
            fake_project_path = Path(tmp_dir) / "fake_proejct"
            fake_project_path.mkdir(parents=True)
            for i in range(1, 5):
                (fake_project_path / weight_format.format(i)).write_text("fake")

            mock_get_train_task = mocker.patch.object(TaskEnvironmentManager, "get_train_task")
            mock_task = mocker.MagicMock()
            mock_task.project_path = str(fake_project_path)
            mock_get_train_task.return_value = mock_task

            mock_report_func = mocker.MagicMock()

            mocker.patch("otx.cli.utils.hpo.get_dataset_adapter")
            mocker.patch("otx.cli.utils.hpo.HpoDataset")

            # run
            trainer = Trainer(
                hp_config={"configuration": {"iterations": 10}, "id": trial_id},
                report_func=mock_report_func,
                model_template=find_and_parse_model_template(cls_template_path),
                data_roots=mocker.MagicMock(),
                task_type=TaskType.CLASSIFICATION,
                hpo_workdir=hpo_workdir,
                initial_weight_name="fake",
                metric="fake",
            )
            trainer.run()

            # check
            mock_report_func.assert_called_once_with(0, 0, done=True)  # finilize report
            assert hpo_workdir.exists()  # make a directory to copy weight
            for i in range(1, 5):  # check model weights are copied
                assert (hpo_workdir / "weight" / trial_id / weight_format.format(i)).exists()

        mock_task.train.assert_called()  # check task.train() is called


class TestHpoCallback:
    @e2e_pytest_unit
    def test_init(self, mocker):
        HpoCallback(mocker.MagicMock(), "fake", 3, mocker.MagicMock())

    @e2e_pytest_unit
    @pytest.mark.parametrize("max_epoch", [-3, 0])
    def test_init_wrong_max_epoch(self, mocker, max_epoch):
        with pytest.raises(ValueError):
            HpoCallback(mocker.MagicMock(), "fake", max_epoch, mocker.MagicMock())

    @e2e_pytest_unit
    def test_call(self, mocker):
        mock_report_func = mocker.MagicMock()

        hpo_call_back = HpoCallback(report_func=mock_report_func, metric="fake", max_epoch=50, task=mocker.MagicMock())
        hpo_call_back(progress=20, score=100)

        mock_report_func.assert_called_once_with(progress=10, score=100)

    @e2e_pytest_unit
    def test_call_and_get_stop_flag(self, mocker):
        mock_report_func = mocker.MagicMock()
        mock_report_func.return_value = TrialStatus.STOP
        mock_task = mocker.MagicMock()

        hpo_call_back = HpoCallback(report_func=mock_report_func, metric="fake", max_epoch=50, task=mock_task)
        hpo_call_back(progress=20, score=100)

        mock_task.cancel_training.assert_called_once_with()

    @e2e_pytest_unit
    def test_not_copy_report_func(self, mocker):
        mock_report_func = mocker.MagicMock()

        hpo_call_back = HpoCallback(report_func=mock_report_func, metric="fake", max_epoch=50, task=mocker.MagicMock())
        new_hpo_call_back = deepcopy(hpo_call_back)
        new_hpo_call_back(progress=20, score=100)

        mock_report_func.assert_called_once()


class TestHpoDataset:
    @e2e_pytest_unit
    def test_init(self, mocker):
        hpo_dataset = HpoDataset(fullset=mocker.MagicMock(), config={"train_environment": {"subset_ratio": 0.5}})
        assert hpo_dataset.subset_ratio == 0.5

    @e2e_pytest_unit
    @pytest.mark.parametrize("subset_ratio", [0.1, 0.5, 1])
    def test_get_subset(self, mocker, subset_ratio):
        mock_fullset = mocker.MagicMock()
        mock_fullset.get_subset.return_value = [i for i in range(10)]
        config = {"train_environment": {"subset_ratio": subset_ratio}}

        hpo_dataset = HpoDataset(fullset=mock_fullset, config=config)
        hpo_sub_dataset = hpo_dataset.get_subset(Subset.TRAINING)

        num_hpo_sub_dataset = len(hpo_sub_dataset)
        assert num_hpo_sub_dataset == round(10 * subset_ratio)

        for i in range(num_hpo_sub_dataset):
            hpo_sub_dataset[i]

    @e2e_pytest_unit
    def test_len_before_get_subset(self):
        hpo_dataset = HpoDataset(fullset=range(10), config={"train_environment": {"subset_ratio": 0.5}})
        assert len(hpo_dataset) == 10

    @e2e_pytest_unit
    def test_getitem_before_get_subset(self):
        hpo_dataset = HpoDataset(fullset=range(10), config={"train_environment": {"subset_ratio": 0.5}})

        for _ in hpo_dataset:
            pass


@e2e_pytest_unit
def test_run_hpo(mocker, mock_environment):
    with TemporaryDirectory() as tmp_dir:
        # prepare
        save_model_to_path = Path(tmp_dir) / "fake"

        mock_get_best_hpo_weight = mocker.patch("otx.cli.utils.hpo.get_best_hpo_weight")
        mock_get_best_hpo_weight.return_value = "mock_best_weight_path"

        def mock_run_hpo(*args, **kwargs):
            return {"config": {"a.b": 1, "c.d": 2}, "id": "1"}

        mock_hpo_runner_instance = mocker.MagicMock()
        mock_hpo_runner_instance.run_hpo.side_effect = mock_run_hpo
        mock_hpo_runner_class = mocker.patch("otx.cli.utils.hpo.HpoRunner")
        mock_hpo_runner_class.return_value = mock_hpo_runner_instance

        def mock_read_model(args1, path, arg2):
            return path

        mocker.patch("otx.cli.utils.hpo.read_model", mock_read_model)

        mock_args = mocker.MagicMock()
        mock_args.hpo_time_ratio = "4"
        mock_args.save_model_to = save_model_to_path

        mock_environment.model_template.task_type = TaskType.CLASSIFICATION

        # run
        environment = run_hpo(mock_args, mock_environment, mocker.MagicMock(), mocker.MagicMock())

        # check
        mock_hpo_runner_instance.run_hpo.assert_called()  # Check that HpoRunner.run_hpo is called
        env_hp = environment.get_hyper_parameters()  # Check that best HP is applied well.
        assert env_hp.a.b == 1
        assert env_hp.c.d == 2
        assert environment.model == "mock_best_weight_path"  # check that best model weight is used


@e2e_pytest_unit
def test_run_hpo_not_supported_task(mocker, action_task_env):
    mock_hpo_runner_instance = mocker.MagicMock()
    mock_hpo_runner_class = mocker.patch("otx.cli.utils.hpo.HpoRunner")
    mock_hpo_runner_class.return_value = mock_hpo_runner_instance

    run_hpo(mocker.MagicMock(), action_task_env, mocker.MagicMock(), mocker.MagicMock())
    mock_hpo_runner_instance.run_hpo.assert_not_called()


@e2e_pytest_unit
def test_get_best_hpo_weight():
    with TemporaryDirectory() as tmp_dir:
        # prepare
        hpo_dir = Path(tmp_dir) / "hpo"
        weight_path = hpo_dir / "weight"
        weight_path.mkdir(parents=True)

        score = {"score": {str(i): i for i in range(1, 11)}}
        bracket_0_dir = hpo_dir / "0"
        bracket_0_dir.mkdir(parents=True)
        for trial_num in range(2):
            with (bracket_0_dir / f"{trial_num}.json").open("w") as f:
                json.dump(score, f)
            trial_weight_path = weight_path / str(trial_num)
            trial_weight_path.mkdir(parents=True)
            for i in range(1, 11):
                (trial_weight_path / f"epoch_{i}.pth").write_text("fake")

        assert get_best_hpo_weight(hpo_dir, "1") == str(weight_path / "1" / "epoch_10.pth")


@e2e_pytest_unit
def test_get_best_hpo_weight_not_exist():
    with TemporaryDirectory() as tmp_dir:
        # prepare
        hpo_dir = Path(tmp_dir) / "hpo"
        weight_path = hpo_dir / "weight"
        weight_path.mkdir(parents=True)

        score = {"score": {str(i): i for i in range(1, 11)}}
        bracket_0_dir = hpo_dir / "0"
        bracket_0_dir.mkdir(parents=True)
        for trial_num in range(1):
            with (bracket_0_dir / f"{trial_num}.json").open("w") as f:
                json.dump(score, f)
            trial_weight_path = weight_path / str(trial_num)
            trial_weight_path.mkdir(parents=True)
            for i in range(1, 11):
                (trial_weight_path / f"epoch_{i}.pth").write_text("fake")

        assert get_best_hpo_weight(hpo_dir, "1") is None


@e2e_pytest_unit
def test_run_trial(mocker):
    mock_run = mocker.patch.object(Trainer, "run")
    run_trial(
        mocker.MagicMock(),
        mocker.MagicMock(),
        mocker.MagicMock(),
        mocker.MagicMock(),
        mocker.MagicMock(),
        mocker.MagicMock(),
        mocker.MagicMock(),
        mocker.MagicMock(),
    )

    mock_run.assert_called()
