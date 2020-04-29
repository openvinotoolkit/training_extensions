from collections import namedtuple
from pathlib import Path
from typing import List

import jsonschema
import pytest

from nncf.config import Config
from tests.conftest import PROJECT_ROOT, TEST_ROOT

GOOD_CONFIG_SOURCES = [
    PROJECT_ROOT / Path("examples/classification/configs"),
    PROJECT_ROOT / Path("examples/semantic_segmentation/configs"),
    PROJECT_ROOT / Path("examples/object_detection/configs"),
    TEST_ROOT / Path("data/configs")
]

BAD_CONFIG_SOURCES = [
    TEST_ROOT / Path("data/schema_validation_bad_configs")
]

ConfigPathVsPassesSchemaVal = namedtuple('ConfigPathVsPassesSchemaVal', ('path', 'should_pass'))
TEST_STRUCTS = []


def get_all_jsons_from_sources(source_directories_list: List[Path]) -> List[Path]:
    retval = []
    for source_dir in source_directories_list:
        files = source_dir.glob('**/*.json')
        retval += files
    return retval


good_config_files = get_all_jsons_from_sources(GOOD_CONFIG_SOURCES)
for file in good_config_files:
    TEST_STRUCTS.append(ConfigPathVsPassesSchemaVal(file, True))

bad_config_files = get_all_jsons_from_sources(BAD_CONFIG_SOURCES)
for file in bad_config_files:
    TEST_STRUCTS.append(ConfigPathVsPassesSchemaVal(file, False))


@pytest.fixture(name="config_test_struct", params=TEST_STRUCTS,
                ids=[str(struct.path.relative_to(PROJECT_ROOT)) for struct in TEST_STRUCTS])
def _config_test_struct(request):
    return request.param


def test_json_against_nncf_config_schema(config_test_struct):
    config_path, should_pass = config_test_struct
    if should_pass:
        _ = Config.from_json(str(config_path))
    else:
        with pytest.raises(jsonschema.ValidationError):
            _ = Config.from_json(str(config_path))
