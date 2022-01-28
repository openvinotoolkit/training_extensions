from pathlib import Path

import numpy as np
import pytest

from ote_sdk.configuration import ConfigurableParameters
from ote_sdk.configuration.helper.create import create
from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.image import Image
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.label_schema import (
    LabelGraph,
    LabelGroup,
    LabelSchemaEntity,
    LabelTree,
)
from ote_sdk.entities.metadata import MetadataItemEntity
from ote_sdk.entities.model import (
    ModelAdapter,
    ModelConfiguration,
    ModelEntity,
    ModelPrecision,
    OptimizationMethod,
)
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.tensor import TensorEntity
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestParamsValidation:
    @staticmethod
    def random_image() -> Image:
        return Image(data=np.random.randint(low=0, high=255, size=(10, 16, 3)))

    @staticmethod
    def scored_labels() -> list:
        detection_label = LabelEntity(name="detection label", domain=Domain.DETECTION)
        segmentation_label = LabelEntity(
            name="segmentation label", domain=Domain.SEGMENTATION
        )
        return [
            ScoredLabel(label=detection_label),
            ScoredLabel(label=segmentation_label),
        ]

    @staticmethod
    def annotations() -> list:
        full_box_rectangle = Rectangle.generate_full_box()
        annotation = Annotation(shape=full_box_rectangle, labels=[])
        other_annotation = Annotation(shape=full_box_rectangle, labels=[])
        return [annotation, other_annotation]

    def annotation_scene(self) -> AnnotationSceneEntity:
        return AnnotationSceneEntity(
            annotations=self.annotations(), kind=AnnotationSceneKind.ANNOTATION
        )

    @staticmethod
    def metadata() -> list:
        numpy = np.random.uniform(low=0.0, high=255.0, size=(10, 15, 3))
        metadata_item = TensorEntity(name="test_metadata", numpy=numpy)
        other_metadata_item = TensorEntity(name="other_metadata", numpy=numpy)
        return [
            MetadataItemEntity(data=metadata_item),
            MetadataItemEntity(data=other_metadata_item),
        ]

    def dataset_items(self) -> list:
        random_image = self.random_image()
        annotation_scene = self.annotation_scene()
        default_values_dataset_item = DatasetItemEntity(random_image, annotation_scene)
        dataset_item = DatasetItemEntity(
            media=random_image,
            annotation_scene=annotation_scene,
            roi=Annotation(
                shape=Rectangle.generate_full_box(), labels=self.scored_labels()
            ),
            metadata=self.metadata(),
            subset=Subset.TESTING,
        )
        return [default_values_dataset_item, dataset_item]

    @staticmethod
    def exclusivity_groups() -> list:
        label_0_1 = LabelEntity(name="Label 0_1", domain=Domain.DETECTION)
        label_0_2 = LabelEntity(name="Label 0_2", domain=Domain.SEGMENTATION)
        label_0_2_4 = LabelEntity(name="Label_0_2_4", domain=Domain.SEGMENTATION)
        label_0_2_5 = LabelEntity(name="Label_0_2_5", domain=Domain.SEGMENTATION)
        exclusivity_0_1_and_0_2 = LabelGroup(
            name="Exclusivity edges 0_1 and 0_2",
            labels=[label_0_1, label_0_2],
            id=ID("ex_01_02"),
        )
        exclusivity_2_4_and_2_5 = LabelGroup(
            name="Exclusivity edges 0_2_4 and 0_2_5", labels=[label_0_2_4, label_0_2_5]
        )
        return [exclusivity_0_1_and_0_2, exclusivity_2_4_and_2_5]

    @staticmethod
    def check_value_error_exception_raised(
        correct_parameters: dict, unexpected_values: list, class_or_function
    ) -> None:
        for key, value in unexpected_values:
            incorrect_parameters_dict = dict(correct_parameters)
            incorrect_parameters_dict[key] = value
            with pytest.raises(ValueError):
                class_or_function(**incorrect_parameters_dict)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check Annotation object initialization parameters validation

        <b>Input data:</b>
        Annotation object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as Annotation
        initialization parameter
        """
        labels = self.scored_labels()
        correct_values_dict = {"shape": Rectangle.generate_full_box(), "labels": labels}
        unexpected_type_value = "unexpected str"
        unexpected_values = [
            # Unexpected string is specified as "shape" parameter
            ("shape", unexpected_type_value),
            # Unexpected string is specified as "labels" parameter
            ("labels", unexpected_type_value),
            # Unexpected string is specified as nested "label"
            ("labels", labels + [unexpected_type_value]),  # type: ignore
            # Unexpected string is specified as "id" parameter
            ("id", unexpected_type_value),
        ]
        self.check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=Annotation,
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_scene_entity_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check AnnotationSceneEntity object initialization parameters validation

        <b>Input data:</b>
        AnnotationSceneEntity object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as AnnotationSceneEntity
        initialization parameter
        """
        annotations = self.annotations()
        correct_values_dict = {
            "annotations": annotations,
            "kind": AnnotationSceneKind.ANNOTATION,
        }
        unexpected_type_value = "unexpected str"
        unexpected_values = [
            # Unexpected string is specified as "annotations" parameter
            ("annotations", unexpected_type_value),
            # Unexpected string is specified nested annotation
            ("annotations", [annotations[0], unexpected_type_value]),
            # Unexpected string is specified as "kind" parameter
            ("kind", unexpected_type_value),
            # Unexpected integer is specified as "editor" parameter
            ("editor", 1),
            # Unexpected string is specified as "creation_date" parameter
            ("creation_date", unexpected_type_value),
            # Unexpected string is specified as "id" parameter
            ("id", unexpected_type_value),
        ]
        self.check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=AnnotationSceneEntity,
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_item_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check DatasetItemEntity object initialization parameters validation

        <b>Input data:</b>
        DatasetItemEntity object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as DatasetItemEntity
        initialization parameter
        """
        unexpected_type_value = 1
        correct_values_dict = {
            "media": self.random_image(),
            "annotation_scene": self.annotation_scene(),
        }
        unexpected_values = [
            # Unexpected integer is specified as "media" parameter
            ("media", unexpected_type_value),
            # Unexpected integer is specified as "annotation_scene" parameter
            ("annotation_scene", unexpected_type_value),
            # Unexpected integer is specified as "roi" parameter
            ("roi", unexpected_type_value),
            # Unexpected integer is specified as "metadata" parameter
            ("metadata", unexpected_type_value),
            # Unexpected integer is specified as nested "metadata" item
            ("metadata", self.metadata() + [unexpected_type_value]),  # type: ignore
            # Unexpected integer is specified as "subset" parameter
            ("subset", unexpected_type_value),
        ]
        self.check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=DatasetItemEntity,
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_entity_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check DatasetEntity object initialization parameters validation

        <b>Input data:</b>
        DatasetEntity object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as DatasetEntity
        initialization parameter
        """
        items = self.dataset_items()
        unexpected_type_value = {"unexpected_key": False}
        correct_values_dict = {"items": items}
        unexpected_values = [
            # Unexpected dictionary is specified as "items" parameter
            ("items", unexpected_type_value),
            # Unexpected boolean is specified as nested "dataset item" parameter
            ("items", items + [False]),  # type: ignore
            # Unexpected dictionary is specified as "purpose" parameter
            ("purpose", unexpected_type_value),
        ]
        self.check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=DatasetEntity,
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check LabelEntity object initialization parameters validation

        <b>Input data:</b>
        LabelEntity object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when incorrect type object is specified as LabelEntity
        initialization parameter
        """
        correct_values_dict = {"name": "label name", "domain": Domain.SEGMENTATION}
        unexpected_type_value = 1
        unexpected_values = [
            # Unexpected integer is specified as "name" parameter
            ("name", unexpected_type_value),
            # Unexpected integer is specified as "domain" parameter
            ("domain", unexpected_type_value),
            # Unexpected integer is specified as "color" parameter
            ("color", unexpected_type_value),
            # Unexpected integer is specified as "hotkey" parameter
            ("hotkey", unexpected_type_value),
            # Unexpected integer is specified as "creation_date" parameter
            ("creation_date", unexpected_type_value),
            # Unexpected integer is specified as "is_empty" parameter
            ("is_empty", unexpected_type_value),
            # Unexpected string is specified as "id" parameter
            ("id", "unexpected str"),
        ]
        self.check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=LabelEntity,
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_schema_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check LabelSchemaEntity object initialization parameters validation

        <b>Input data:</b>
        LabelSchemaEntity object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as LabelSchemaEntity
        initialization parameter
        """
        correct_values_dict = {
            "exclusivity_graph": LabelGraph(directed=True),
            "label_tree": LabelTree(),
        }
        unexpected_type_value = "unexpected str"
        unexpected_values = [
            # Unexpected string is specified as "exclusivity_graph" parameter
            ("exclusivity_graph", unexpected_type_value),
            # Unexpected string is specified as "label_tree" parameter
            ("label_tree", unexpected_type_value),
            # Unexpected string is specified as "label_groups" parameter
            ("label_groups", unexpected_type_value),
            # Unexpected string is specified as nested "label_group"
            ("label_groups", self.exclusivity_groups() + [unexpected_type_value]),
        ]
        self.check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=LabelSchemaEntity,
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_entity_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check ModelEntity object initialization parameters validation

        <b>Input data:</b>
        ModelEntity object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as ModelEntity
        initialization parameter
        """
        dataset = DatasetEntity()
        configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="Test header"),
            label_schema=LabelSchemaEntity(),
        )
        unexpected_str = "unexpected str"
        unexpected_int = 1
        unexpected_float = 1.1
        model_adapter = ModelAdapter(b"{0: binaryrepo://localhost/repo/data_source/0}")
        correct_values_dict = {
            "train_dataset": dataset,
            "configuration": configuration,
        }
        unexpected_values = [
            # Unexpected string is specified as "train_dataset" parameter
            ("train_dataset", unexpected_str),
            # Unexpected string is specified as "configuration" parameter
            ("configuration", unexpected_str),
            # Unexpected string is specified as "creation_date" parameter
            ("creation_date", unexpected_str),
            # Unexpected string is specified as "performance" parameter
            ("performance", unexpected_str),
            # Unexpected string is specified as "previous_trained_revision" parameter
            ("previous_trained_revision", unexpected_str),
            # Unexpected string is specified as "previous_revision" parameter
            ("previous_revision", unexpected_str),
            # Unexpected string is specified as "version" parameter
            ("version", unexpected_str),
            # Unexpected string is specified as "tags" parameter
            ("tags", unexpected_str),
            # Unexpected integer is specified as nested "tag"
            ("tags", ["tag_1", unexpected_int]),
            # Unexpected string is specified as "model_format" parameter
            ("model_format", unexpected_str),
            # Unexpected string is specified as "training_duration" parameter
            ("training_duration", unexpected_str),
            # Unexpected string is specified as "model_adapters" parameter
            ("model_adapters", unexpected_str),
            # Unexpected integer is specified as "model_adapter" key
            (
                "model_adapters",
                {"model_adapter_1": model_adapter, unexpected_int: model_adapter},
            ),
            # Unexpected string is specified as "model_adapter" value
            (
                "model_adapters",
                {"model_adapter_1": model_adapter, "model_adapter_2": unexpected_str},
            ),
            # Unexpected string is specified as "exportable_code_adapter" parameter
            ("exportable_code_adapter", unexpected_str),
            # Unexpected string is specified as "precision" parameter
            ("precision", unexpected_str),
            # Unexpected integer is specified as nested "precision"
            ("precision", [ModelPrecision.INT8, unexpected_int]),
            # Unexpected float is specified as "latency" parameter
            ("latency", unexpected_float),
            # Unexpected float is specified as "fps_throughput" parameter
            ("fps_throughput", unexpected_float),
            # Unexpected string is specified as "target_device" parameter
            ("target_device", unexpected_str),
            # Unexpected integer is specified as nested "target_device"
            ("target_device_type", unexpected_int),
            # Unexpected string is specified as "optimization_type" parameter
            ("optimization_type", unexpected_str),  # str-type "optimization_type"
            # Unexpected string is specified as "optimization_methods" parameter
            ("optimization_methods", unexpected_str),
            # Unexpected string is specified as nested "optimization_method"
            ("optimization_methods", [OptimizationMethod.QUANTIZATION, unexpected_str]),
            # Unexpected string is specified as "optimization_objectives" parameter
            ("optimization_objectives", unexpected_str),
            # Unexpected integer key is specified in nested "optimization_objective"
            (
                "optimization_objectives",
                {"objective_1": "optimization_1", unexpected_int: "optimization_2"},
            ),
            # Unexpected integer value is specified in nested "optimization_objective"
            (
                "optimization_objectives",
                {"objective_1": "optimization_1", "objective_2": unexpected_int},
            ),
            # Unexpected string is specified as "performance_improvement" parameter
            ("performance_improvement", unexpected_str),
            # Unexpected integer key is specified in nested "performance_improvement"
            ("performance_improvement", {"improvement_1": 1.1, unexpected_int: 1.2}),
            # Unexpected string value is specified in nested "performance_improvement"
            (
                "performance_improvement",
                {"improvement_1": 1.1, "improvement_2": unexpected_str},
            ),
            # Unexpected string is specified as "model_size_reduction" parameter
            ("model_size_reduction", unexpected_str),
            # Unexpected string is specified as "_id" parameter
            ("_id", unexpected_int),
        ]
        self.check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=ModelEntity,
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_rectangle_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check Rectangle object initialization parameters validation

        <b>Input data:</b>
        Rectangle object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as Rectangle
        initialization parameter
        """
        rectangle_label = ScoredLabel(
            label=LabelEntity(name="Rectangle label", domain=Domain.DETECTION)
        )
        unexpected_type_value = "unexpected str"
        correct_values_dict = {"x1": 0.1, "y1": 0.1, "x2": 0.8, "y2": 0.6}
        unexpected_values = [
            # Unexpected string is specified as "x1" parameter
            ("x1", unexpected_type_value),
            # Unexpected string is specified as "y1" parameter
            ("y1", unexpected_type_value),
            # Unexpected string is specified as "x2" parameter
            ("x2", unexpected_type_value),
            # Unexpected string is specified as "y2" parameter
            ("y2", unexpected_type_value),
            # Unexpected string is specified as "labels" parameter
            ("labels", unexpected_type_value),  # str-type "labels"
            # Unexpected string is specified as nested "label"
            ("labels", [rectangle_label, unexpected_type_value]),
            # Unexpected string is specified as "modification_date" parameter
            ("modification_date", unexpected_type_value),
        ]
        self.check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=Rectangle,
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_result_set_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check ResultSetEntity object initialization parameters validation

        <b>Input data:</b>
        ResultSetEntity object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as ResultSetEntity
        initialization parameter
        """
        dataset_entity = DatasetEntity()
        model_configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(
                header="model configurable parameters"
            ),
            label_schema=LabelSchemaEntity(),
        )
        correct_values_dict = {
            "model": ModelEntity(
                train_dataset=dataset_entity, configuration=model_configuration
            ),
            "ground_truth_dataset": dataset_entity,
            "prediction_dataset": dataset_entity,
        }
        unexpected_type_value = 1
        unexpected_values = [
            # Unexpected integer is specified as "model" parameter
            ("model", unexpected_type_value),
            # Unexpected integer is specified as "ground_truth_dataset" parameter
            ("ground_truth_dataset", unexpected_type_value),
            # Unexpected integer is specified as "prediction_dataset" parameter
            ("prediction_dataset", unexpected_type_value),
            # Unexpected integer is specified as "purpose" parameter
            ("purpose", unexpected_type_value),
            # Unexpected integer is specified as "performance" parameter
            ("performance", unexpected_type_value),
            # Unexpected integer is specified as "creation_date" parameter
            ("creation_date", unexpected_type_value),
            # Unexpected integer is specified as "id" parameter
            ("id", unexpected_type_value),
        ]
        self.check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=ResultSetEntity,
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_scored_label_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check ScoredLabel object initialization parameters validation

        <b>Input data:</b>
        ScoredLabel object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        ScoredLabel object initialization parameter
        """
        label = LabelEntity(name="test scored label", domain=Domain.SEGMENTATION)
        correct_values_dict = {"label": label, "probability": 0.1}
        unexpected_type_value = "unexpected_str"
        unexpected_values = [
            # Unexpected string is specified as "label" parameter
            ("label", unexpected_type_value),
            # Unexpected string is specified as "probability" parameter
            ("probability", unexpected_type_value),
        ]
        self.check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=ScoredLabel,
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_task_environment_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check TaskEnvironment object initialization parameters validation

        <b>Input data:</b>
        TaskEnvironment object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        TaskEnvironment initialization parameter
        """
        dummy_template = str(
            Path(__file__).parent / Path("../entities/dummy_template.yaml")
        )
        correct_values_dict = {
            "model_template": parse_model_template(dummy_template),
            "model": None,
            "hyper_parameters": ConfigurableParameters(
                header="hyper configurable parameters"
            ),
            "label_schema": LabelSchemaEntity(),
        }
        unexpected_type_value = "unexpected str"
        unexpected_values = [
            # Unexpected string is specified as "model_template" parameter
            ("model_template", unexpected_type_value),
            # Unexpected string is specified as "model" parameter
            ("model", unexpected_type_value),
            # Unexpected string is specified as "hyper_parameters" parameter
            ("hyper_parameters", unexpected_type_value),
            # Unexpected string is specified as "label_schema" parameter
            ("label_schema", unexpected_type_value),
        ]
        self.check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=TaskEnvironment,
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_create_input_parameters_validation(self):
        """
        <b>Description:</b>
        Check "create" function input parameters validation

        <b>Input data:</b>
        "input_config" parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as "input_config"
        parameter
        """
        for incorrect_parameter in [
            # Unexpected integer is specified as "input_config" parameter
            1,
            # Empty string is specified as "input_config" parameter
            "",
            # Empty dictionary is specified as "input_config" parameter
            {},
            # Path to non-existing file is specified as "input_config" parameter
            str(Path(__file__).parent / Path(r"./non_existing.yaml")),
            # Path to non-yaml file is specified as "input_config" parameter
            str(Path(__file__).parent / Path(r"./unexpected_type.jpg")),
            # Path with null character is specified as "input_config" parameter
            str(Path(__file__).parent / Path(r"./null\0char.yaml")),
        ]:
            with pytest.raises(ValueError):
                create(incorrect_parameter)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_image_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check Image object initialization parameters validation

        <b>Input data:</b>
        Image object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as Image initialization
        parameter
        """
        for key, value in [
            # Unexpected integer is specified as "data" parameter
            ("data", 1),
            # Unexpected integer is specified as "file_path" parameter
            ("file_path", 1),
            # Empty string is specified as "file_path" parameter
            ("file_path", ""),
            # Path to file with unexpected extension is specified as "file_path" parameter
            (
                "file_path",
                str(Path(__file__).parent / Path("./unexpected_extension.yaml")),
            ),
            # Path to non-existing file is specified as "file_path" parameter
            ("file_path", str(Path(__file__).parent / Path("./non_existing.jpg"))),
            # Path with null character is specified as "file_path" parameter
            ("file_path", str(Path(__file__).parent / Path(r"./null\0char.jpg"))),
        ]:
            with pytest.raises(ValueError):
                Image(**{key: value})
