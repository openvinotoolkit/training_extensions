import json
import shutil
import struct
from os import path as osp
from pathlib import Path

import cv2
import numpy as np
import pytest

from otx.api.entities.model import ModelOptimizationType
from otx.api.usecases.adapters.model_adapter import ModelAdapter
from otx.cli.utils import io as target_package
from otx.cli.utils.io import (
    get_explain_dataset_from_filelist,
    get_image_files,
    read_binary,
    read_label_schema,
    read_model,
    save_model_data,
    save_saliency_output,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit

IMG_DATA_FORMATS = (
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".gif",
    ".GIF",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
    ".png",
    ".PNG",
)


@e2e_pytest_unit
def test_save_model_data(mocker, tmp_dir):
    mock_model = mocker.MagicMock()
    model_adapter = ModelAdapter(b"fake")
    file_name = "model.pth"
    mock_model.model_adapters = {file_name: model_adapter}

    save_model_data(mock_model, tmp_dir)

    with open(osp.join(tmp_dir, file_name), "rb") as f:
        assert f.readline() == b"fake"


@e2e_pytest_unit
def test_read_binary(tmp_dir):
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
def test_read_xml_bin_model(mocker, model_adapter_keys, tmp_dir):
    # prepare
    tmp_dir = Path(tmp_dir)
    for key in model_adapter_keys:
        (tmp_dir / key).write_text(key)

    xml_model_path = tmp_dir / "model.xml"
    xml_model_path.write_text("xml_model")
    bin_model_path = tmp_dir / "model.bin"
    bin_model_path.write_text("bin_model")

    # run
    model = read_model(mocker.MagicMock(), str(xml_model_path), mocker.MagicMock())

    # check
    model_adapters = model.model_adapters
    assert model_adapters["openvino.xml"].data == b"xml_model"
    assert model_adapters["openvino.bin"].data == b"bin_model"
    for key in model_adapter_keys:
        assert model_adapters[key].data == bytes(key, "utf-8")


@e2e_pytest_unit
def test_read_pth_model(mocker, tmp_dir):
    # prepare
    tmp_dir = Path(tmp_dir)
    model_path = tmp_dir / "model.pth"
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
        model_adapters[f"aux_model_{i}.pth"].data == bytes(f"aux_model_{i}.pth", "utf-8")
    assert model.optimization_type == ModelOptimizationType.NNCF


@pytest.fixture
def mock_zip_file(model_adapter_keys, tmp_dir) -> str:
    """Mock zip file.

    zip file structure is as below.
    model/model.xml
    model/model.bin
    model/config.json
    model_adapter_keys => should have float value
    """
    tmp_dir = Path(tmp_dir)
    model_zip_dir = tmp_dir / "model_zip"
    model_zip_dir.mkdir()
    model_dir_in_zip = model_zip_dir / "model"
    model_dir_in_zip.mkdir()
    (model_dir_in_zip / "model.xml").write_text("xml_model")
    (model_dir_in_zip / "model.bin").write_text("bin_model")
    with (model_dir_in_zip / "config.json").open("w") as f:
        model_config = {"model_parameters": {}}
        for key in model_adapter_keys:
            model_config["model_parameters"][key] = 0.1234
        model_config["model_parameters"]["labels"] = {"fake": "fake"}
        json.dump(model_config, f)

    shutil.make_archive(tmp_dir / "model", "zip", model_zip_dir)

    return tmp_dir / "model.zip"


@e2e_pytest_unit
def test_read_zip_model(mocker, model_adapter_keys, mock_zip_file):
    # run
    model = read_model(mocker.MagicMock(), str(mock_zip_file), mocker.MagicMock())

    # check
    model_adapters = model.model_adapters
    assert model_adapters["openvino.xml"].data == b"xml_model"
    assert model_adapters["openvino.bin"].data == b"bin_model"
    for key in model_adapter_keys:
        assert model_adapters[key].data == struct.pack("f", 0.1234)


@e2e_pytest_unit
def test_read_unknown_model(mocker):
    with pytest.raises(ValueError):
        read_model(mocker.MagicMock(), "fake.unknown", mocker.MagicMock())


@pytest.fixture
def mock_lebel_schema_mapper_instance(mocker):
    mock_lebel_schema_mapper_class = mocker.patch.object(target_package, "LabelSchemaMapper")
    mock_lebel_schema_mapper_instance = mocker.MagicMock()
    mock_lebel_schema_mapper_class.return_value = mock_lebel_schema_mapper_instance
    return mock_lebel_schema_mapper_instance


@e2e_pytest_unit
@pytest.mark.parametrize("extension", [".xml", ".bin", ".pth"])
def test_read_label_schema_with_model_file(mock_lebel_schema_mapper_instance, extension, tmp_dir):
    # prepare
    tmp_dir = Path(tmp_dir)

    model_path = tmp_dir / f"model.{extension}"
    model_path.write_text("fake")
    fake_label_schema = {"fake": "fake"}
    with (tmp_dir / "label_schema.json").open("w") as f:
        json.dump(fake_label_schema, f)

    # run
    read_label_schema(str(model_path))

    # check
    mock_lebel_schema_mapper_instance.backward.assert_called_once_with(fake_label_schema)


@e2e_pytest_unit
def test_read_label_schema_with_model_zipfile(mock_lebel_schema_mapper_instance, mock_zip_file):
    # run
    read_label_schema(str(mock_zip_file))

    # check
    mock_lebel_schema_mapper_instance.backward.assert_called_once_with({"fake": "fake"})


@e2e_pytest_unit
@pytest.mark.parametrize("img_data_format", IMG_DATA_FORMATS)
def test_get_single_image_files(img_data_format):
    img_files = get_image_files(f"fake_path/img{img_data_format}")

    assert img_files[0] == ("./", f"fake_path/img{img_data_format}")


@e2e_pytest_unit
def test_get_image_files_in_dir(tmp_dir):
    # prepare
    tmp_dir = Path(tmp_dir)
    sub_dir_1 = tmp_dir / "sub_dir_1"
    sub_dir_1_1 = sub_dir_1 / "sub_dir_1_1"
    sub_dir_2 = tmp_dir / "sub_dir_2"
    sub_dir_1_1.mkdir(parents=True)
    sub_dir_2.mkdir()
    (sub_dir_1_1 / "fake.jpg").write_text("fake")
    (sub_dir_2 / "fake.jpg").write_text("fake")

    # run
    img_files = get_image_files(str(tmp_dir))

    # check
    for img_file in img_files:
        assert img_file in ((str(sub_dir_1_1), "fake.jpg"), (str(sub_dir_2), "fake.jpg"))


@e2e_pytest_unit
def test_get_image_files_empty_dir(tmp_dir):
    assert get_image_files(tmp_dir) is None


@e2e_pytest_unit
@pytest.mark.parametrize(
    "process_saliency_maps",
    [True, False],
    ids=["w_post_processing", "wo_post_processing"],
)
def test_save_saliency_output(tmp_dir, process_saliency_maps):
    # prepare
    img = np.array([[100 for _ in range(3)] for _ in range(3)])
    saliency_map = np.zeros([3, 3], dtype=np.uint8)
    weight = 0.3

    # run
    save_saliency_output(process_saliency_maps, img, saliency_map, tmp_dir, "fake", weight=weight)

    # check
    if process_saliency_maps:
        saliency_map_file = Path(tmp_dir) / "fake_saliency_map.png"
    else:
        saliency_map_file = Path(tmp_dir) / "fake_saliency_map.tiff"
    assert saliency_map_file.exists()
    saved_saliency = cv2.imread(str(saliency_map_file))
    assert (saved_saliency == saliency_map).all()

    if process_saliency_maps:
        overlay_img = Path(tmp_dir) / "fake_overlay_img.png"
        assert overlay_img.exists()
        saved_overlay = cv2.imread(str(overlay_img))
        assert (saved_overlay == np.array([[100 * weight for _ in range(3)] for _ in range(3)])).all()


@e2e_pytest_unit
def test_get_explain_dataset_from_filelist(tmp_dir):
    # prepare
    tmp_dir = Path(tmp_dir)
    fake_img = np.array([[[i for _ in range(3)] for _ in range(3)] for i in range(1, 4)])
    cv2.imwrite(str(tmp_dir / "fake1.jpg"), fake_img)
    cv2.imwrite(str(tmp_dir / "fake2.jpg"), fake_img)
    image_files = [(tmp_dir, "fake1.jpg"), (tmp_dir, "fake2.jpg")]

    # run
    dataset_entity = get_explain_dataset_from_filelist(image_files)

    # check
    assert (dataset_entity[0].media.numpy == fake_img).all()
    assert (dataset_entity[1].media.numpy == fake_img).all()
