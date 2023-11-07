# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from otx.v2.adapters.torch.modules.utils.io import (
    get_image_filenames,
    get_image_height_and_width,
    read_image,
)


def test_get_image_filenames(tmp_dir_path) -> None:
    single_image = tmp_dir_path / "image.jpg"
    single_image.touch()

    # Load Single Image
    result = get_image_filenames(single_image)
    assert len(result) == 1

    # Load Empty Folder
    empty_folder = tmp_dir_path / "empty_folder"
    empty_folder.mkdir()

    with pytest.raises(ValueError, match="Found 0 images"):
        get_image_filenames(empty_folder)

    # Load Image folder
    image_1 = empty_folder / "image1.jpg"
    image_1.touch()
    image_2 = empty_folder / "image2.jpg"
    image_2.touch()

    result = get_image_filenames(empty_folder)
    assert len(result) == 2


def test_get_image_height_and_width() -> None:
    # Load None
    with pytest.raises(TypeError, match="int or tuple"):
        get_image_height_and_width(image_size=None)

    # Load Int
    result = get_image_height_and_width(image_size=256)
    assert result == (256, 256)

    # Load Tuple
    result = get_image_height_and_width(image_size=(256, 256))
    assert result == (256, 256)


def test_read_image(mocker, tmp_dir_path) -> None:
    mock_imread = mocker.patch("otx.v2.adapters.torch.modules.utils.io.cv2.imread")
    mock_cvtcolor = mocker.patch("otx.v2.adapters.torch.modules.utils.io.cv2.cvtColor")
    mock_resize = mocker.patch("otx.v2.adapters.torch.modules.utils.io.cv2.resize")
    image = tmp_dir_path / "image.jpg"
    image.touch()

    # Load Single Image
    result = read_image(image, image_size=256)
    assert result is not None
    mock_imread.assert_called_once_with(str(image))
    import cv2
    mock_cvtcolor.assert_called_once_with(mock_imread.return_value, cv2.COLOR_BGR2RGB)
    mock_resize.assert_called_once_with(mock_cvtcolor.return_value, dsize=(256, 256), interpolation=cv2.INTER_AREA)
