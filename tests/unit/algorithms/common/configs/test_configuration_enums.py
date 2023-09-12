"""Tests for common configuration enums in OTX algorithms."""
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from otx.algorithms.common.configs.configuration_enums import (
    POTQuantizationPreset,
    StorageCacheScheme,
    BatchSizeAdaptType,
    InputSizePreset,
)


@e2e_pytest_unit
def test_pot_quansization_preset():
    assert len(POTQuantizationPreset) == 2


@e2e_pytest_unit
def test_storage_cache_scheme():
    assert len(StorageCacheScheme) == 6


@e2e_pytest_unit
def test_batsh_size_adapt_type():
    assert len(BatchSizeAdaptType) == 3


@e2e_pytest_unit
def test_input_size_preset():
    assert len(InputSizePreset) == 10
    assert InputSizePreset.parse("xxx") == None
    assert InputSizePreset.parse("Default") == None
    assert InputSizePreset.parse("Auto") == (0, 0)
    assert InputSizePreset.parse("1x1") == (1, 1)
    assert InputSizePreset.DEFAULT.tuple == None
    assert InputSizePreset.AUTO.tuple == (0, 0)
    assert InputSizePreset._64x64.tuple == (64, 64)
    input_sizes = InputSizePreset.input_sizes()
    assert len(input_sizes) == 8
    assert input_sizes[-1] == (1024, 1024)
