# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Helper functions for tests."""

from pathlib import Path

import numpy as np
from datumaro import Polygon
from datumaro.components.annotation import (
    AnnotationType,
    LabelCategories,
    MaskCategories,
)
from datumaro.components.dataset import DatasetItem
from datumaro.components.errors import MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.media import Image
from datumaro.plugins.data_formats.common_semantic_segmentation import (
    CommonSemanticSegmentationPath,
)
from datumaro.util.definitions import DEFAULT_SUBSET_NAME
from datumaro.util.image import save_image
from datumaro.util.meta_file_util import save_meta_file
from otx.core.utils.mask_util import polygon_to_bitmap


def generate_random_bboxes(
    image_width: int,
    image_height: int,
    num_boxes: int,
    min_width: int = 10,
    min_height: int = 10,
) -> np.ndarray:
    """Generate random bounding boxes.
    Parameters:
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        num_boxes (int): Number of bounding boxes to generate.
        min_width (int): Minimum width of the bounding box. Default is 10.
        min_height (int): Minimum height of the bounding box. Default is 10.
    Returns:
        ndarray: A NumPy array of shape (num_boxes, 4) representing bounding boxes in format (x_min, y_min, x_max, y_max).
    """
    max_width = image_width - min_width
    max_height = image_height - min_height

    bg = np.random.MT19937(seed=42)
    rg = np.random.Generator(bg)

    x_min = rg.integers(0, max_width, size=num_boxes)
    y_min = rg.integers(0, max_height, size=num_boxes)
    x_max = x_min + rg.integers(min_width, image_width, size=num_boxes)
    y_max = y_min + rg.integers(min_height, image_height, size=num_boxes)

    x_max[x_max > image_width] = image_width
    y_max[y_max > image_height] = image_height
    areas = (x_max - x_min) * (y_max - y_min)
    bboxes = np.column_stack((x_min, y_min, x_max, y_max))
    return bboxes[areas > 0]


def find_folder(base_path: Path, folder_name: str) -> Path:
    """
    Find the folder with the given name within the specified base path.

    Args:
        base_path (Path): The base path to search within.
        folder_name (str): The name of the folder to find.

    Returns:
        Path: The path to the folder.
    """
    for folder_path in base_path.rglob(folder_name):
        if folder_path.is_dir():
            return folder_path
    msg = f"Folder {folder_name} not found in {base_path}."
    raise FileNotFoundError(msg)


class CommonSemanticSegmentationExporter(Exporter):
    """Exporter for common semantic segmentation format."""

    DEFAULT_IMAGE_EXT = ".jpg"

    def _apply_impl(self) -> None:
        """Apply the exporter to the dataset."""
        extractor = self._extractor
        save_dir = Path(self._save_dir)

        if self._extractor.media_type() and not issubclass(
            self._extractor.media_type(),
            Image,
        ):
            msg = "Media type is not an image"
            raise MediaTypeError(msg)

        save_dir.mkdir(parents=True, exist_ok=True)

        categories = extractor.categories()
        label_names = [label.name for label in categories[AnnotationType.label]]
        if "background" not in label_names:
            label_names.insert(0, "background")

        mask_categories = MaskCategories.generate(size=len(label_names))
        label_categories = LabelCategories()
        for label in label_names:
            label_categories.add(label)

        category = {
            AnnotationType.mask: mask_categories,
            AnnotationType.label: label_categories,
        }
        subsets = self._extractor.subsets()
        for subset_name, subset in subsets.items():
            _subset_name = subset_name
            if not _subset_name or _subset_name == DEFAULT_SUBSET_NAME:
                _subset_name = DEFAULT_SUBSET_NAME

            subset_dir = Path(save_dir, _subset_name)
            subset_dir.mkdir(parents=True, exist_ok=True)

            mask_dir = subset_dir / CommonSemanticSegmentationPath.MASKS_DIR
            img_dir = subset_dir / CommonSemanticSegmentationPath.IMAGES_DIR
            for item in subset:
                self._export_item_annotation(item, mask_dir)
                if self._save_media:
                    self._export_media(item, img_dir)
                save_meta_file(subset_dir, category)

    def _export_item_annotation(
        self,
        item: DatasetItem,
        save_dir: str,
    ) -> None:
        """Export the annotations of an item."""
        annotations = item.annotations
        height, width, _ = item.media.data.shape
        if not annotations:
            return

        index_map = np.zeros((height, width), dtype=np.uint8)
        for ann in annotations:
            if ann.type is AnnotationType.polygon:
                bitmask = polygon_to_bitmap([ann], height, width)[0]
                index_map[bitmask] = ann.label + 1
            elif ann.type is AnnotationType.ellipse:
                bitmask = polygon_to_bitmap([Polygon(ann.as_polygon(20))], height, width)[0]
                index_map[bitmask] = ann.label + 1
            elif ann.type is AnnotationType.bbox:
                x1, y1, w, h = (int(v) for v in ann.get_bbox())
                index_map[y1 : y1 + h, x1 : x1 + w] = ann.label + 1
            else:
                raise NotImplementedError(
                    "Exporting %s is not supported" % ann.type,
                )

        # standardise item id
        item_id = item.id.replace("/", "_")
        dst = save_dir / (item_id + ".png")

        # save index map as an image
        save_image(
            dst=str(dst),
            image=index_map,
            create_dir=True,
        )

    def _export_media(self, item, save_dir) -> None:
        """Export the media of an item."""
        item_id = item.id.replace("/", "_")
        dst = save_dir / (item_id + self.DEFAULT_IMAGE_EXT)
        save_image(
            dst=str(dst),
            image=item.media.data,
            create_dir=True,
        )
