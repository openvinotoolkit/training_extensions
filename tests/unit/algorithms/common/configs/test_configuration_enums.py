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
    assert len(InputSizePreset) == 9
    assert InputSizePreset.parse(InputSizePreset.DEFAULT.value) == None
    assert InputSizePreset.parse(InputSizePreset.AUTO.value) == (0, 0)
    assert InputSizePreset.parse(InputSizePreset._64x64.value) == (64, 64)
    input_sizes = InputSizePreset.input_sizes()
    assert len(input_sizes) == 7
    assert input_sizes[-1] == (1024, 1024)
