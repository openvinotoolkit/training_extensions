"""Tests for test_distance.py"""
from otx.algorithms.common.utils import dist_utils
from otx.algorithms.common.utils.dist_utils import append_dist_rank_suffix
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_append_dist_rank_suffix_not_distributed_training(mocker):
    mock_os = mocker.patch.object(dist_utils, "os")
    mock_os.environ = {}
    file_name = "temporary.pth"
    new_file_name = append_dist_rank_suffix(file_name)

    assert file_name == new_file_name


@e2e_pytest_unit
def test_append_dist_rank_suffix_distributed_training(mocker):
    mock_os = mocker.patch.object(dist_utils, "os")
    mock_os.environ = {"LOCAL_RANK": "2"}
    new_file_name = append_dist_rank_suffix("temporary.pth")

    assert new_file_name == "temporary_proc2.pth"
