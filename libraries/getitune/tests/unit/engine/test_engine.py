# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from unittest.mock import MagicMock, patch

import pytest

from getitune.backend.lightning.engine import LightningEngine
from getitune.backend.lightning.models.base import LightningModel
from getitune.backend.openvino.engine import OVEngine
from getitune.backend.openvino.models.base import OVModel
from getitune.data.module import DataModule
from getitune.engine import Engine, create_engine
from getitune.engine.utils.create import _RECIPE_PATH, _read_backend, _resolve_recipe


class TestCreateEngine:
    @pytest.fixture
    def mock_engine_subclass(self):
        """Fixture to create a mock Engine subclass."""
        mock_engine_cls = MagicMock(spec=Engine)
        mock_engine_cls.is_supported.return_value = True
        return mock_engine_cls

    @patch("getitune.backend.lightning.engine.LightningEngine.is_supported", return_value=False)
    @patch("getitune.backend.openvino.engine.OVEngine.is_supported", return_value=False)
    @patch("getitune.engine.Engine.__subclasses__", autospec=True)
    def test_create_engine(self, mock___subclasses__, mock_ov_is_supported, mock_lt_is_supported, mock_engine_subclass):
        """Test create_engine with arbitrary Engine."""
        mock___subclasses__.return_value = [mock_engine_subclass]
        mock_model = MagicMock(spec=LightningModel)
        mock_data = MagicMock(spec=DataModule)

        engine_instance = create_engine(mock_model, mock_data)  # pyrefly: ignore[bad-argument-type]

        mock_engine_subclass.is_supported.assert_called_once_with(mock_model, mock_data)
        mock_engine_subclass.assert_called_once_with(model=mock_model, data=mock_data)
        assert engine_instance == mock_engine_subclass.return_value

        # test create_engine when is_supported returns False
        mock_engine_subclass.is_supported.return_value = False
        with pytest.raises(ValueError, match="No engine found for model .* and data .*"):
            create_engine(mock_model, mock_data)  # pyrefly: ignore[bad-argument-type]

        # test create_engine when no subclasses are found
        mock___subclasses__.return_value = []
        mock_model = MagicMock(spec=LightningModel)
        mock_data = MagicMock(spec=DataModule)

        with pytest.raises(ValueError, match="No engine found for model .* and data .*"):
            create_engine(mock_model, mock_data)  # pyrefly: ignore[bad-argument-type]

    def test_create_native_engine(self, mocker):
        mock_model = MagicMock(spec=LightningModel)
        mock_data = MagicMock(spec=DataModule)
        mock_engine_init = mocker.patch("getitune.backend.lightning.engine.LightningEngine.__init__", return_value=None)

        # test LightningEngine creation with LightningModel
        engine_instance = create_engine(mock_model, mock_data)
        assert isinstance(engine_instance, LightningEngine)
        mock_engine_init.assert_called_once_with(model=mock_model, data=mock_data)

        # test with additional kwargs
        engine_instance = create_engine(mock_model, mock_data, work_dir="path/to/workdir", device="CPU")
        assert isinstance(engine_instance, LightningEngine)
        mock_engine_init.assert_called_with(
            model=mock_model,
            data=mock_data,
            work_dir="path/to/workdir",
            device="CPU",
        )

    def test_create_openvino_engine(self, mocker):
        """Test create_engine for OpenVINO Engine."""
        # tests OpenVINO Engine creation with OVModel
        mock_model = MagicMock(spec=OVModel)
        mock_data = MagicMock(spec=DataModule)
        mock_engine_init = mocker.patch("getitune.backend.openvino.engine.OVEngine.__init__", return_value=None)
        engine_instance = create_engine(mock_model, mock_data)
        assert isinstance(engine_instance, OVEngine)
        mock_engine_init.assert_called_once_with(model=mock_model, data=mock_data)

        # test with IR path
        mock_model = "/path/to/model.xml"
        engine_instance = create_engine(mock_model, mock_data, work_dir="path/to/workdir")
        assert isinstance(engine_instance, OVEngine)
        mock_engine_init.assert_called_with(model=mock_model, data=mock_data, work_dir="path/to/workdir")


class TestResolveRecipe:
    def test_recipe_path_direct_yaml(self):
        """Passing an absolute .yaml path resolves to itself."""
        recipe = _RECIPE_PATH / "detection" / "yolox_s.yaml"
        result = _resolve_recipe(str(recipe), task=None)
        assert result == recipe.resolve()

    def test_recipe_path_not_found(self):
        """Nonexistent .yaml path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Recipe file not found"):
            _resolve_recipe("/nonexistent/recipe.yaml", task=None)

    def test_model_name_unique(self):
        """Bare model name with a single match resolves correctly."""
        result = _resolve_recipe("yolo26_n", task=None)
        assert result.name == "yolo26_n.yaml"
        assert result.parent.name == "detection"

    def test_model_name_not_found(self):
        """Nonexistent model name raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No recipe found for model"):
            _resolve_recipe("nonexistent_model_xyz", task=None)

    def test_model_name_ambiguous_without_task(self):
        """Ambiguous name without task raises ValueError listing candidates."""
        with pytest.raises(ValueError, match="matches multiple recipes"):
            _resolve_recipe("dino_v2", task=None)

    def test_model_name_ambiguous_with_task(self):
        """Ambiguous name with task= resolves to the correct subdirectory."""
        result = _resolve_recipe("dino_v2", task="SEMANTIC_SEGMENTATION")
        assert result.name == "dino_v2.yaml"
        assert "semantic_segmentation" in str(result)


class TestReadBackend:
    def test_read_backend_ultralytics(self, tmp_path):
        """Recipe with backend: ultralytics returns 'ultralytics'."""
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text("backend: ultralytics\n")
        assert _read_backend(recipe) == "ultralytics"

    def test_read_backend_lightning_default(self, tmp_path):
        """Recipe without backend field defaults to 'lightning'."""
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text("task: DETECTION\n")
        assert _read_backend(recipe) == "lightning"

    def test_read_backend_empty_file(self, tmp_path):
        """Empty recipe file defaults to 'lightning'."""
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text("")
        assert _read_backend(recipe) == "lightning"


class TestCreateEngineRecipe:
    @patch("getitune.backend.lightning.engine.LightningEngine.from_config")
    def test_recipe_path_dispatches_to_backend(self, mock_from_config, tmp_path):
        """A recipe .yaml path dispatches to the correct backend's from_config."""
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text("backend: lightning\n")
        mock_from_config.return_value = MagicMock(spec=LightningEngine)

        result = create_engine(str(recipe), data="dummy_path")

        mock_from_config.assert_called_once()
        assert isinstance(result, Engine)

    @patch("getitune.backend.lightning.engine.LightningEngine.from_config")
    def test_model_name_resolves_and_dispatches(self, mock_from_config):
        """Bare model name is resolved to a recipe and dispatched."""
        mock_from_config.return_value = MagicMock(spec=LightningEngine)

        result = create_engine("yolox_s", data="dummy_path")

        mock_from_config.assert_called_once()
        assert isinstance(result, Engine)

    def test_unknown_backend_raises(self, tmp_path):
        """Recipe with unknown backend field raises ValueError."""
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text("backend: unknown_backend\n")
        with pytest.raises(ValueError, match="Unknown backend"):
            create_engine(str(recipe), data="dummy_path")

    def test_nonexistent_recipe_path_raises(self):
        """Passing a nonexistent .yaml path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Recipe file not found"):
            create_engine("/nonexistent/path.yaml", data="dummy_path")

    def test_nonexistent_model_name_raises(self):
        """Passing a nonexistent model name raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No recipe found for model"):
            create_engine("nonexistent_model_xyz", data="dummy_path")

    def test_model_name_ambiguous_raises(self):
        """Ambiguous model name without task raises ValueError."""
        with pytest.raises(ValueError, match="matches multiple recipes"):
            create_engine("dino_v2", data="dummy_path")

    def test_task_disambiguation(self, mocker):
        """Ambiguous model name with task= disambiguates successfully."""
        mock_from_config = mocker.patch(
            "getitune.backend.lightning.engine.LightningEngine.from_config",
            return_value=MagicMock(spec=LightningEngine),
        )
        result = create_engine("dino_v2", data="dummy_path", task="SEMANTIC_SEGMENTATION")
        mock_from_config.assert_called_once()
        assert isinstance(result, Engine)
