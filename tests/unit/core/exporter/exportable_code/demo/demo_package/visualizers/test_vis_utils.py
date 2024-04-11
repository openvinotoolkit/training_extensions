from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from numpy.random import PCG64, Generator


@pytest.fixture(scope="module", autouse=True)
def fxt_import_module():
    global ColorPalette, get_actmap, dump_frames  # noqa: PLW0603
    from otx.core.exporter.exportable_code.demo.demo_package.visualizers.vis_utils import (
        ColorPalette as _ColorPalette,
    )
    from otx.core.exporter.exportable_code.demo.demo_package.visualizers.vis_utils import (
        dump_frames as _dump_frames,
    )
    from otx.core.exporter.exportable_code.demo.demo_package.visualizers.vis_utils import (
        get_actmap as _get_actmap,
    )

    ColorPalette = _ColorPalette
    get_actmap = _get_actmap
    dump_frames = _dump_frames


def test_activation_map_shape():
    generator = Generator(PCG64())
    saliency_map = (generator.random((100, 100)) * 255).astype(np.uint8)
    output_res = (50, 50)
    result = get_actmap(saliency_map, output_res)
    assert result.shape == (50, 50, 3)


def test_no_saved_frames():
    output = "output"
    input_path = "input"
    capture = MagicMock()
    saved_frames = []
    dump_frames(saved_frames, output, input_path, capture)
    assert not Path(output).exists()


@patch("otx.core.exporter.exportable_code.demo.demo_package.visualizers.vis_utils.cv2.VideoWriter_fourcc")
@patch("otx.core.exporter.exportable_code.demo.demo_package.visualizers.vis_utils.cv2.VideoWriter")
@patch("otx.core.exporter.exportable_code.demo.demo_package.visualizers.vis_utils.get_input_names_list")
def test_video_input(mock_get_input_names_list, mock_video_writer, mock_video_writer_fourcc, tmp_path):
    output = str(tmp_path / "output")
    input_path = "input"
    capture = MagicMock(spec=cv2.VideoCapture)
    capture.get_type = lambda: "VIDEO"
    capture.fps = lambda: 30
    saved_frames = [MagicMock(shape=(100, 100, 3))]
    filenames = ["video.mp4"]
    mock_get_input_names_list.return_value = filenames
    mock_video_writer_fourcc.return_value = "mp4v"
    dump_frames(saved_frames, output, input_path, capture)
    mock_video_writer_fourcc.assert_called_once_with(*"mp4v")
    mock_video_writer.assert_called_once()


@patch("otx.core.exporter.exportable_code.demo.demo_package.visualizers.vis_utils.cv2.imwrite")
@patch("otx.core.exporter.exportable_code.demo.demo_package.visualizers.vis_utils.get_input_names_list")
@patch("otx.core.exporter.exportable_code.demo.demo_package.visualizers.vis_utils.cv2.cvtColor")
def test_image_input(mock_imwrite, mock_get_input_names_list, mock_cvtcolor, tmp_path):
    output = str(tmp_path / "output")
    input_path = "input"
    capture = MagicMock(spec=cv2.VideoCapture)
    capture.get_type = lambda: "IMAGE"
    saved_frames = [MagicMock(), MagicMock()]
    filenames = ["image1.jpeg", "image2.jpeg"]
    mock_get_input_names_list.return_value = filenames
    dump_frames(saved_frames, output, input_path, capture)
    assert mock_cvtcolor.call_count == 2
    assert mock_imwrite.call_count == 2


class TestColorPalette:
    def test_colorpalette_init_with_zero_classes(self):
        expected_msg = "ColorPalette accepts only the positive number of colors"
        with pytest.raises(ValueError, match=expected_msg):
            ColorPalette(num_classes=0)
        with pytest.raises(ValueError, match=expected_msg):
            ColorPalette(num_classes=-5)

    def test_colorpalette_length(self):
        num_classes = 5
        palette = ColorPalette(num_classes)
        assert len(palette) == num_classes

    def test_colorpalette_getitem(self):
        num_classes = 3
        palette = ColorPalette(num_classes)
        color = palette[1]  # assuming 0-based indexing
        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_colorpalette_getitem_out_of_range(self):
        num_classes = 3
        palette = ColorPalette(num_classes)
        color = palette[num_classes + 2]  # out-of-range index
        assert color == palette[2]  # because it should wrap around

    def test_colorpalette_to_numpy_array(self):
        num_classes = 2
        palette = ColorPalette(num_classes)
        np_array = palette.to_numpy_array()
        assert isinstance(np_array, np.ndarray)
        assert np_array.shape == (num_classes, 3)

    def test_colorpalette_hsv2rgb_known_values(self):
        h, s, v = 0.5, 1, 1  # Cyan in HSV
        expected_rgb = (0, 255, 255)  # Cyan in RGB
        assert ColorPalette.hsv2rgb(h, s, v) == expected_rgb

    def test_dist_same_color(self):
        # Colors that are the same should have a distance of 0
        color = (0.5, 0.5, 0.5)
        assert ColorPalette._dist(color, color) == 0

    def test_dist_different_colors(self):
        # Test distance between two different colors
        color1 = (0.1, 0.2, 0.3)
        color2 = (0.4, 0.5, 0.6)
        expected_distance = 0.54
        assert ColorPalette._dist(color1, color2) == expected_distance
