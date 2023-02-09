from os import path as osp
from tempfile import TemporaryDirectory
from pathlib import Path
import shutil

import pytest

from otx.cli.utils import io as target_package
from otx.cli.utils.io import (
    save_model_data,
    read_binary,
    read_model,
    read_label_schema,
    generate_label_schema,
    get_image_files,
    save_saliency_output,
    get_explain_dataset_from_filelist,
)
from otx.api.usecases.adapters.model_adapter import ModelAdapter
from otx.api.entities.model import ModelOptimizationType
from tests.test_suite.e2e_test_system import e2e_pytest_unit

@e2e_pytest_unit
def test_save_model_data(mocker):
    mock_model = mocker.MagicMock()
    model_adapter = ModelAdapter(b"fake")
    file_name = "model.pth"
    mock_model.model_adapters = {file_name : model_adapter}

    with TemporaryDirectory() as tmp_dir:
        save_model_data(mock_model, tmp_dir)

        with open(osp.join(tmp_dir, file_name), "rb") as f:
            assert f.readline() == b"fake"

@e2e_pytest_unit
def test_read_binary():
    with TemporaryDirectory() as tmp_dir:
        file_path = osp.join(tmp_dir, "test.txt") 
        with open(file_path, "w") as f:
            f.write("fake")

        assert read_binary(file_path) == b"fake"

@pytest.fixture
def model_adapter_keys():
    return (
        "confidence_threshold",
        "image_threshold",
        "pixel_threshold",
        "min",
        "max",
        "config.json",
    )

@e2e_pytest_unit
def test_read_xml_bin_model(mocker, model_adapter_keys):

    with TemporaryDirectory() as tmp_dir:
        # prepare
        tmp_dir = Path(tmp_dir)
        for key in model_adapter_keys:
            (tmp_dir / key).write_text(key)

        xml_model_path = (tmp_dir / "model.xml")
        xml_model_path.write_text("xml_model")
        bin_model_path = (tmp_dir / "model.bin")
        bin_model_path.write_text("bin_model")

        # run
        model = read_model(mocker.MagicMock(), str(xml_model_path), mocker.MagicMock())

        # check
        model_adapters = model.model_adapters
        assert model_adapters["openvino.xml"].data == b"xml_model"
        assert model_adapters["openvino.bin"].data == b"bin_model"
        for key in model_adapter_keys:
            assert model_adapters[key].data == bytes(key, 'utf-8')

@e2e_pytest_unit
def test_read_pth_model(mocker):
    with TemporaryDirectory() as tmp_dir:
        # prepare
        tmp_dir = Path(tmp_dir)
        model_path = (tmp_dir / "model.pth")
        model_path.write_text("model")
        mock_is_checkpoint_nncf = mocker.patch.object(target_package, "is_checkpoint_nncf")
        mock_is_checkpoint_nncf.return_value = True

        for i in range(1, 5):
            (tmp_dir / f"aux_model_{i}.pth").write_text(f"aux_{i}")

        # run
        model = read_model(mocker.MagicMock(), str(model_path), mocker.MagicMock())

        # check
        model_adapters = model.model_adapters
        assert model_adapters["weights.pth"].data == b"model"
        for i in range(1, 5):
            model_adapters[f"aux_model_{i}.pth"].data == bytes(f"aux_model_{i}.pth", 'utf-8')
        assert model.optimization_type == ModelOptimizationType.NNCF


@e2e_pytest_unit
def test_read_zip_model(mocker, model_adapter_keys):
    """
    model/model.xml
    model/model.bin
    model/config.json
    model_adapter_keys => should have float value
    """
    with TemporaryDirectory() as tmp_dir:
        # prepare
        tmp_dir = Path(tmp_dir)
        model_zip_dir = (tmp_dir / "model_zip")
        model_zip_dir.mkdir()
        for key in model_adapter_keys:
            (model_zip_dir / key).write_text(key)

        xml_model_path = (model_zip_dir / "model" / "model.xml")
        xml_model_path.write_text("xml_model")
        bin_model_path = (model_zip_dir / "model" / "model.bin")
        bin_model_path.write_text("bin_model")
        zip_file_path = (tmp_dir / model.zip)

        shutil.make_archive(zip_file_path, 'zip', model_zip_dir)

        # run
        model = read_model(mocker.MagicMock(), str(zip_file_path), mocker.MagicMock())

        # check
        model_adapters = model.model_adapters

@e2e_pytest_unit
def test_read_unknown_model(mocker):
    with pytest.raises(ValueError):
        read_model(mocker.MagicMock(), "fake.unknown", mocker.MagicMock())
