import os
import warnings

import numpy as np

from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.color import Color
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.task_environment import TaskEnvironment
from tests.test_helpers import generate_random_annotated_image

DEFAULT_SEG_TEMPLATE_DIR = os.path.join("otx/algorithms/segmentation/configs", "ocr_lite_hrnet_18_mod2")
DEFAULT_RECIPE_CONFIG_PATH = "otx/recipes/stages/segmentation/incremental.py"

labels_names = ("rectangle", "ellipse", "triangle")


def generate_otx_label_schema(labels_names=labels_names):
    label_domain = Domain.SEGMENTATION
    rgb = [int(i) for i in np.random.randint(0, 256, 3)]
    colors = [Color(*rgb) for _ in range(len(labels_names))]
    not_empty_labels = [
        LabelEntity(name=name, color=colors[i], domain=label_domain, id=i) for i, name in enumerate(labels_names)
    ]
    empty_label = LabelEntity(
        name="Empty label",
        color=Color(42, 43, 46),
        is_empty=True,
        domain=label_domain,
        id=len(not_empty_labels),
    )

    label_schema = LabelSchemaEntity()
    exclusive_group = LabelGroup(name="labels", labels=not_empty_labels, group_type=LabelGroupType.EXCLUSIVE)
    empty_group = LabelGroup(name="empty", labels=[empty_label], group_type=LabelGroupType.EMPTY_LABEL)
    label_schema.add_group(exclusive_group)
    label_schema.add_group(empty_group)
    return label_schema


def init_environment(params, model_template):
    labels_schema = generate_otx_label_schema()
    environment = TaskEnvironment(
        model=None,
        hyper_parameters=params,
        label_schema=labels_schema,
        model_template=model_template,
    )

    return environment


def generate_otx_dataset(number_of_images=5):
    items = []
    labels_schema = generate_otx_label_schema()
    labels_list = labels_schema.get_labels(False)
    for i in range(0, number_of_images):
        image_numpy, shapes = generate_random_annotated_image(
            image_width=640,
            image_height=480,
            labels=labels_list,
            max_shapes=20,
            min_size=50,
            max_size=100,
            random_seed=None,
        )
        # Convert all shapes to polygons
        out_shapes = []
        for shape in shapes:
            shape_labels = shape.get_labels(include_empty=True)

            in_shape = shape.shape
            if isinstance(in_shape, Rectangle):
                points = [
                    Point(in_shape.x1, in_shape.y1),
                    Point(in_shape.x2, in_shape.y1),
                    Point(in_shape.x2, in_shape.y2),
                    Point(in_shape.x1, in_shape.y2),
                ]
            elif isinstance(in_shape, Ellipse):
                points = [Point(x, y) for x, y in in_shape.get_evenly_distributed_ellipse_coordinates()]
            elif isinstance(in_shape, Polygon):
                points = in_shape.points

            out_shapes.append(Annotation(Polygon(points=points), labels=shape_labels))

        image = Image(data=image_numpy)
        annotation = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=out_shapes)
        items.append(DatasetItemEntity(media=image, annotation_scene=annotation))
    warnings.resetwarnings()

    dataset = DatasetEntity(items)

    return dataset
