from os import path as osp
from tempfile import TemporaryDirectory

import pytest

from otx.cli.utils.nncf import get_number_of_fakequantizers_in_xml, is_checkpoint_nncf
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_is_checkpoint_nncf_meta_exist(mocker):
    mock_state = {"meta": {"nncf_enable_compression": "fake"}}
    mocker.patch("torch.load").return_value = mock_state

    assert is_checkpoint_nncf("fake")

    # state["meta"]["nncf_enable_compression"]
    # state["nncf_enable_compression"]


@e2e_pytest_unit
def test_is_checkpoint_nncf_metainfo_exist(mocker):
    mock_state = {"nncf_metainfo": "fake"}
    mocker.patch("torch.load").return_value = mock_state

    assert is_checkpoint_nncf("fake")


@e2e_pytest_unit
def test_is_not_checkpoint_nncf_meta_not_exist(mocker):
    mock_state = {"fake": "fake"}
    mocker.patch("torch.load").return_value = mock_state

    assert not is_checkpoint_nncf("fake")


@e2e_pytest_unit
@pytest.mark.parametrize("number_of_fakequantizers_in_xml", [0, 10, 100])
def test_get_number_of_fakequantizers_in_xml(number_of_fakequantizers_in_xml):
    with TemporaryDirectory() as tmp_dir:
        path_to_xml = osp.join(tmp_dir, "fake.xml")
        with open(path_to_xml, "w") as f:
            for _ in range(number_of_fakequantizers_in_xml):
                f.write('type="FakeQuantize"\n')

        assert get_number_of_fakequantizers_in_xml(path_to_xml) == number_of_fakequantizers_in_xml
