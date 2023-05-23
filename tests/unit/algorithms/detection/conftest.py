import pytest

from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.entities.model_template import TaskType
from .test_helpers import generate_det_dataset


@pytest.fixture
def otx_model():
    model_configuration = ModelConfiguration(
        configurable_parameters=ConfigurableParameters(header="header", description="description"),
        label_schema=LabelSchemaEntity(),
    )
    return ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)


@pytest.fixture(scope="session")
def fxt_det_dataset_entity(number_of_images: int = 8) -> DatasetEntity:
    dataset, _ = generate_det_dataset(TaskType.DETECTION, number_of_images)
    return dataset
