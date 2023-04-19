import pytest
from otx.algorithms.segmentation.adapters.mmseg.utils import build_scalar_scheduler, build_segmentor
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_build_scalar_scheduler(mocker):
    cfg = mocker.MagicMock()
    builder = mocker.patch("mmseg.models.builder.MODELS.build", return_value=True)
    build_scalar_scheduler(cfg)
    builder.assert_called_once_with(cfg)


@e2e_pytest_unit
def test_build_segmentor(mocker):
    from mmcv.utils import Config

    cfg = Config({"model": {}, "load_from": "foo.pth"})
    mocker.patch("mmseg.models.build_segmentor")
    load_ckpt = mocker.patch("otx.algorithms.segmentation.adapters.mmseg.utils.builder.load_checkpoint")
    build_segmentor(cfg)
    load_ckpt.assert_called()

    build_segmentor(cfg, is_training=True)
    load_ckpt.assert_called()
    assert cfg.load_from is None
